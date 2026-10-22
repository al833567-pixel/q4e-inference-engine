#include "q4e-fuse.cuh"
#include "mmvf.cuh"

// q4e: fused kernels for the small ops of a qwen4exp decode step. Each one repeats the arithmetic of the kernels it
// replaces in the same order (rms_norm_f32's vector path, mul_mat_vec_f, dsv4_hc_post_f32, ssm_conv_f32, the unary
// ops), so the results are bit-identical to the separate kernels.

static __device__ __forceinline__ float q4e_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float q4e_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// ---------------------------------------------------------------------------------------------------------------------
// conv state: concat(state, x^T) + the rollback-slot tail copies (+ ssm_conv + silu), one thread per channel

#define Q4E_CONV_MAX_COLS 24
#define Q4E_CONV_MAX_K    8

struct q4e_conv_slots {
    int     off[Q4E_FUSE_MAX_SLOTS];
    float * dst[Q4E_FUSE_MAX_SLOTS];
};

template <bool conv>
static __global__ void q4e_conv_state_kernel(
        const char * src0, const char * src1, float * cat, const q4e_conv_slots slots, const int n_slots, const int64_t slot_s_stride,
        const float * w, float * y, const int S, const int T, const int C,
        const int64_t nb00, const int64_t nb01, const int64_t nb02,
        const int64_t nb10, const int64_t nb11, const int64_t nb12,
        const int64_t w_s1, const int64_t y_s1, const int64_t y_s2) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= C) {
        return;
    }

    const int n = S + T;
    float v[Q4E_CONV_MAX_COLS];
#pragma unroll
    for (int j = 0; j < Q4E_CONV_MAX_COLS; ++j) {
        if (j < S) {
            v[j] = *(const float *) (src0 + j*nb00 + c*nb01 + s*nb02);
        } else if (j < n) {
            v[j] = *(const float *) (src1 + (j - S)*nb10 + c*nb11 + s*nb12);
        }
    }

    float * catr = cat + ((int64_t) s*C + c)*n;
#pragma unroll
    for (int j = 0; j < Q4E_CONV_MAX_COLS; ++j) {
        if (j < n) {
            catr[j] = v[j];
        }
    }

    for (int p = 0; p < n_slots; ++p) {
        float * d = slots.dst[p] + s*slot_s_stride + (int64_t) c*S;
        const int off = slots.off[p];
        for (int j = 0; j < S; ++j) {
            d[j] = v[off + j];
        }
    }

    if constexpr (conv) {
        // as ssm_conv_f32: sum over the kernel taps in order, plus the zero bias, then silu
        const int d_conv = S + 1;
        float wv[Q4E_CONV_MAX_K];
#pragma unroll
        for (int j = 0; j < Q4E_CONV_MAX_K; ++j) {
            if (j < d_conv) {
                wv[j] = w[c*w_s1 + j];
            }
        }
        for (int t = 0; t < T; ++t) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < Q4E_CONV_MAX_K; ++j) {
                if (j < d_conv) {
                    sumf += v[t + j] * wv[j];
                }
            }
            sumf += 0.0f;
            y[s*y_s2 + t*y_s1 + c] = sumf / (1.0f + expf(-sumf));
        }
    }
}

void ggml_cuda_op_q4e_conv_state(ggml_backend_cuda_context & ctx, const q4e_conv_state_args & args) {
    const ggml_tensor * cat  = args.concat;
    const ggml_tensor * src0 = cat->src[0];
    const ggml_tensor * src1 = cat->src[1];

    const int S = (int) src0->ne[0];
    const int T = (int) src1->ne[0];
    const int C = (int) cat->ne[1];
    const int n_seqs = (int) cat->ne[2];

    GGML_ASSERT(S + T <= Q4E_CONV_MAX_COLS && S + 1 <= Q4E_CONV_MAX_K);
    GGML_ASSERT(args.n_slots <= Q4E_FUSE_MAX_SLOTS);

    q4e_conv_slots slots = {};
    for (int p = 0; p < args.n_slots; ++p) {
        slots.off[p] = args.slot_off[p];
        slots.dst[p] = args.slot_dst[p];
    }
    // the slot destinations of one sequence are contiguous, the sequences one cache row apart
    const int64_t slot_s_stride = args.n_slots > 0 ? args.slot_s_stride : 0;

    const int threads = 128;
    const dim3 grid((C + threads - 1)/threads, n_seqs);
    cudaStream_t stream = ctx.stream();

    if (args.conv_w) {
        const ggml_tensor * w = args.conv_w;
        const ggml_tensor * y = args.conv_out;
        q4e_conv_state_kernel<true><<<grid, threads, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (float *) cat->data, slots, args.n_slots, slot_s_stride,
            (const float *) w->data, (float *) y->data, S, T, C,
            src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[0], src1->nb[1], src1->nb[2],
            w->nb[1]/sizeof(float), y->nb[1]/sizeof(float), y->nb[2]/sizeof(float));
    } else {
        q4e_conv_state_kernel<false><<<grid, threads, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (float *) cat->data, slots, args.n_slots, slot_s_stride,
            nullptr, nullptr, S, T, C,
            src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[0], src1->nb[1], src1->nb[2],
            0, 0, 0);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// gated delta net gates: the alpha and beta rows of mul_mat_vec_f<nv_bfloat16, float, ncols_dst, block_size> in one grid

template <int ncols_dst, int block_size>
static __global__ void q4e_gdn_gates_kernel(
        const nv_bfloat16 * wa, const nv_bfloat16 * wb, const float * y, const float * dt, const float * a,
        float * gate, float * beta, const int ncols2, const int64_t stride_row_a, const int64_t stride_row_b,
        const int stride_col_y2, const int H, const int64_t gate_stride_col, const int64_t beta_stride_col) {
    const int  row  = blockIdx.x;
    const bool is_b = row >= H;
    const int  r    = is_b ? row - H : row;
    const int  tid  = threadIdx.x;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const nv_bfloat162 * x2 = (const nv_bfloat162 *) (is_b ? wb + r*stride_row_b : wa + r*stride_row_a);
    const float2       * y2 = (const float2 *) y;

    __shared__ float buf_iw[warp_size];

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};

    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    const float value = sumf[tid];
    if (is_b) {
        beta[tid*beta_stride_col + r] = q4e_sigmoid(value);
    } else {
        const float v = q4e_softplus(value + dt[r]);
        gate[tid*gate_stride_col + r] = v * a[r];
    }
}

// the block size mul_mat_vec_f picks for these rows
static int q4e_mmvf_block_size(const int64_t ncols, const int warp_size) {
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    for (int64_t block_size = 2*warp_size; block_size <= 256; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return (int) block_size_best;
}

bool ggml_cuda_q4e_gdn_gates_supported(const ggml_tensor * w_a, const ggml_tensor * y) {
    const int device = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || warp_size != 32) {
        return false;
    }
    if (w_a->type != GGML_TYPE_BF16 || y->type != GGML_TYPE_F32 || y->ne[1] < 1 || y->ne[1] > 8 || y->ne[2] != 1 || y->ne[3] != 1) {
        return false;
    }
    if (!ggml_cuda_should_use_mmvf(w_a->type, cc, warp_size, w_a->ne, w_a->nb, y->ne[1])) {
        return false;
    }
    return w_a->ne[2] == 1 && w_a->ne[3] == 1 && y->nb[0] == sizeof(float) && y->nb[1] % (2*sizeof(float)) == 0 &&
           ((uintptr_t) y->data % 8) == 0 && ((uintptr_t) w_a->data % 4) == 0;
}

template <int ncols_dst>
static void q4e_gdn_gates_launch(int block_size, const dim3 grid, cudaStream_t stream,
        const nv_bfloat16 * wa, const nv_bfloat16 * wb, const float * y, const float * dt, const float * a,
        float * gate, float * beta, const int ncols2, const int64_t sra, const int64_t srb, const int scy2, const int H,
        const int64_t gsc, const int64_t bsc) {
    switch (block_size) {
#define Q4E_GATES_CASE(BS) case BS: q4e_gdn_gates_kernel<ncols_dst, BS><<<grid, BS, 0, stream>>>(wa, wb, y, dt, a, gate, beta, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        Q4E_GATES_CASE(32)
        Q4E_GATES_CASE(64)
        Q4E_GATES_CASE(96)
        Q4E_GATES_CASE(128)
        Q4E_GATES_CASE(160)
        Q4E_GATES_CASE(192)
        Q4E_GATES_CASE(224)
        Q4E_GATES_CASE(256)
#undef Q4E_GATES_CASE
        default: GGML_ABORT("unexpected block size");
    }
}

void ggml_cuda_op_q4e_gdn_gates(ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_a, const ggml_tensor * w_b, const ggml_tensor * y,
        const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta) {
    const int64_t ncols = w_a->ne[0];
    const int     H     = (int) w_a->ne[1];
    const int     T     = (int) y->ne[1];
    const int block_size = q4e_mmvf_block_size(ncols, 32);

    const nv_bfloat16 * wa = (const nv_bfloat16 *) w_a->data;
    const nv_bfloat16 * wb = (const nv_bfloat16 *) w_b->data;
    const int64_t sra = w_a->nb[1]/sizeof(nv_bfloat16);
    const int64_t srb = w_b->nb[1]/sizeof(nv_bfloat16);
    const int scy2 = (int) (y->nb[1]/sizeof(float))/2;

    // gate is [H, T] and beta [1, H, T], both contiguous
    const int64_t gsc = H;
    const int64_t bsc = H;

    const dim3 grid(2*H, 1, 1);
    cudaStream_t stream = ctx.stream();
    const float * yd  = (const float *) y->data;
    const float * dtd = (const float *) dt->data;
    const float * ad  = (const float *) a->data;
    float * gd = (float *) gate->data;
    float * bd = (float *) beta->data;
    const int ncols2 = (int) (ncols/2);

    switch (T) {
        case 1: q4e_gdn_gates_launch<1>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        case 2: q4e_gdn_gates_launch<2>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        case 3: q4e_gdn_gates_launch<3>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        case 4: q4e_gdn_gates_launch<4>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        case 5: q4e_gdn_gates_launch<5>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        case 6: q4e_gdn_gates_launch<6>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        case 7: q4e_gdn_gates_launch<7>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        case 8: q4e_gdn_gates_launch<8>(block_size, grid, stream, wa, wb, yd, dtd, ad, gd, bd, ncols2, sra, srb, scy2, H, gsc, bsc); break;
        default: GGML_ABORT("unexpected token count");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// rms_norm vector path (rms_norm_f32<block_size, 1, true, ...>) with a fused producer or consumer

// rms_norm_f32_dispatch takes the vector path with vpt = 1 for these sizes
static int q4e_rms_block_size(const int64_t ncols) {
    const int64_t n4 = ncols/4;
    if (n4 <= 32)   return 32;
    if (n4 <= 128)  return 128;
    if (n4 <= 256)  return 256;
    if (n4 <= 512)  return 512;
    if (n4 <= 1024) return 1024;
    return 0;
}

bool ggml_cuda_q4e_rms_vec_ok(const ggml_tensor * x, const ggml_tensor * mul, const ggml_tensor * dst) {
    const int64_t ncols = x->ne[0];
    if (ncols % 4 != 0 || ncols <= 0 || q4e_rms_block_size(ncols) == 0 || ((uintptr_t) dst->data & 15) != 0) {
        return false;
    }
    if (!ggml_cuda_is_aligned(x, 16) || x->nb[0] != sizeof(float)) {
        return false;
    }
    if (mul && (mul->ne[0] != ncols || !ggml_cuda_is_aligned(mul, 16) || mul->nb[0] != sizeof(float))) {
        return false;
    }
    return true;
}

template <int block_size>
static __global__ void q4e_hc_post_norm_kernel(
        const float * x, const float * residual, const float * post, float * hdst, const float * w, float * ndst,
        const int n_embd, const int hc,
        const int64_t sx1, const int64_t sr1, const int64_t sr2, const int64_t sp0, const int64_t sp1,
        const int64_t sd1, const int64_t sd2, const int64_t w_s1, const int64_t w_s2, const int w_ne1, const int w_ne2,
        const float gate_scale_in, const float gate_scale_out, const float eps) {
    const int c   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float s_sum[32];

    float p = post[c*sp0 + t*sp1];
    p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));

    const int ncols4 = n_embd / 4;
    const float * xr = x + t*sx1;
    const float * rr = residual + c*sr1 + t*sr2;
    float       * hr = hdst + c*sd1 + t*sd2;

    float4 xs = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float tmp = 0.0f;
    if (tid < ncols4) {
        const int e = 4*tid;
        float s0 = xr[e + 0] * p;
        s0 += rr[e + 0];
        float s1 = xr[e + 1] * p;
        s1 += rr[e + 1];
        float s2 = xr[e + 2] * p;
        s2 += rr[e + 2];
        float s3 = xr[e + 3] * p;
        s3 += rr[e + 3];
        hr[e + 0] = s0;
        hr[e + 1] = s1;
        hr[e + 2] = s2;
        hr[e + 3] = s3;
        xs = make_float4(s0, s1, s2, s3);
        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / n_embd + eps);

    if (tid < ncols4) {
        const float4 * w4 = (const float4 *) (w + (c % w_ne1)*w_s1 + (t % w_ne2)*w_s2);
        float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
        const float4 m = w4[tid];
        r.x *= m.x;
        r.y *= m.y;
        r.z *= m.z;
        r.w *= m.w;
        ((float4 *) (ndst + ((int64_t) t*hc + c)*n_embd))[tid] = r;
    }
}

void ggml_cuda_op_q4e_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out,
        const ggml_tensor * rms_norm, ggml_tensor * mul) {
    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * post     = gate_scale_in->src[0];
    const ggml_tensor * w        = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    const float gin  = ggml_get_op_params_f32(gate_scale_in,  0);
    const float gout = ggml_get_op_params_f32(gate_scale_out, 0);
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int n_embd = (int) hc_post->ne[0];
    const int hc     = (int) hc_post->ne[1];
    const int T      = (int) hc_post->ne[2];

    const dim3 grid(hc, T, 1);
    cudaStream_t stream = ctx.stream();

#define Q4E_HPN_CASE(BS) case BS: q4e_hc_post_norm_kernel<BS><<<grid, BS, 0, stream>>>( \
        (const float *) x->data, (const float *) residual->data, (const float *) post->data, (float *) hc_post->data, \
        (const float *) w->data, (float *) mul->data, n_embd, hc, \
        x->nb[1]/sizeof(float), residual->nb[1]/sizeof(float), residual->nb[2]/sizeof(float), \
        post->nb[0]/sizeof(float), post->nb[1]/sizeof(float), \
        hc_post->nb[1]/sizeof(float), hc_post->nb[2]/sizeof(float), \
        w->nb[1]/sizeof(float), w->nb[2]/sizeof(float), (int) w->ne[1], (int) w->ne[2], gin, gout, eps); break;
    switch (q4e_rms_block_size(n_embd)) {
        Q4E_HPN_CASE(32)
        Q4E_HPN_CASE(128)
        Q4E_HPN_CASE(256)
        Q4E_HPN_CASE(512)
        Q4E_HPN_CASE(1024)
        default: GGML_ABORT("unexpected row size");
    }
#undef Q4E_HPN_CASE
}

// rms_norm(x)*w, then sigmoid(z)*that as the fused unary+mul
template <int block_size>
static __global__ void q4e_norm_gate_kernel(
        const float * x, const float * w, const float * z, float * dst, const int ncols,
        const int64_t sx1, const int64_t sx2, const int64_t sx3,
        const int64_t w_s1, const int64_t w_s2, const int64_t w_s3, const int w_ne1, const int w_ne2, const int w_ne3,
        const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    __shared__ float s_sum[32];

    const int64_t   idx  = ((int64_t) (sample*nchannels + channel)*nrows + row)*ncols;
    const float4 * x4   = (const float4 *) (x + sample*sx3 + channel*sx2 + row*sx1);
    const float4 * mul4 = (const float4 *) (w + (sample % w_ne3)*w_s3 + (channel % w_ne2)*w_s2 + (row % w_ne1)*w_s1);
    const float4 * z4   = (const float4 *) (z + idx);
    float4       * dst4 = (float4 *) (dst + idx);

    const int ncols4 = ncols / 4;
    float4 xs = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float tmp = 0.0f;
    if (tid < ncols4) {
        xs = x4[tid];
        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

    if (tid < ncols4) {
        float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
        const float4 m = mul4[tid];
        r.x *= m.x;
        r.y *= m.y;
        r.z *= m.z;
        r.w *= m.w;
        const float4 g = z4[tid];
        float4 o;
        o.x = q4e_sigmoid(g.x) * r.x;
        o.y = q4e_sigmoid(g.y) * r.y;
        o.z = q4e_sigmoid(g.z) * r.z;
        o.w = q4e_sigmoid(g.w) * r.w;
        dst4[tid] = o;
    }
}

void ggml_cuda_op_q4e_norm_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
        const ggml_tensor * sigmoid, ggml_tensor * out) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * z = sigmoid->src[0];
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int ncols = (int) x->ne[0];
    const dim3 grid((unsigned) x->ne[1], (unsigned) x->ne[2], (unsigned) x->ne[3]);
    cudaStream_t stream = ctx.stream();

#define Q4E_NG_CASE(BS) case BS: q4e_norm_gate_kernel<BS><<<grid, BS, 0, stream>>>( \
        (const float *) x->data, (const float *) w->data, (const float *) z->data, (float *) out->data, ncols, \
        x->nb[1]/sizeof(float), x->nb[2]/sizeof(float), x->nb[3]/sizeof(float), \
        w->nb[1]/sizeof(float), w->nb[2]/sizeof(float), w->nb[3]/sizeof(float), (int) w->ne[1], (int) w->ne[2], (int) w->ne[3], eps); break;
    switch (q4e_rms_block_size(ncols)) {
        Q4E_NG_CASE(32)
        Q4E_NG_CASE(128)
        Q4E_NG_CASE(256)
        Q4E_NG_CASE(512)
        Q4E_NG_CASE(1024)
        default: GGML_ABORT("unexpected row size");
    }
#undef Q4E_NG_CASE
}

// two rms_norm -> scale pairs (the q and k l2 norms of a GDN layer) in one launch
struct q4e_rms_scale_job {
    const float * x;
    float       * dst;
    int64_t       sx1, sx2, sx3;
    int           nrows, nchannels, nblocks;
    float         eps, scale;
};

template <int block_size>
static __global__ void q4e_rms_scale2_kernel(const q4e_rms_scale_job j0, const q4e_rms_scale_job j1, const int ncols) {
    const bool second = (int) blockIdx.x >= j0.nblocks;
    const q4e_rms_scale_job & j = second ? j1 : j0;
    const int b   = second ? blockIdx.x - j0.nblocks : blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_sum[32];

    const int row     = b % j.nrows;
    const int channel = (b / j.nrows) % j.nchannels;
    const int sample  = b / (j.nrows*j.nchannels);

    const float4 * x4   = (const float4 *) (j.x + sample*j.sx3 + channel*j.sx2 + row*j.sx1);
    float4       * dst4 = (float4 *) (j.dst + (int64_t) b*ncols);

    const int ncols4 = ncols / 4;
    float4 xs = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float tmp = 0.0f;
    if (tid < ncols4) {
        xs = x4[tid];
        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + j.eps);

    if (tid < ncols4) {
        float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
        r.x *= j.scale;
        r.y *= j.scale;
        r.z *= j.scale;
        r.w *= j.scale;
        dst4[tid] = r;
    }
}

static q4e_rms_scale_job q4e_rms_scale_job_of(const ggml_tensor * rms, ggml_tensor * scale) {
    const ggml_tensor * x = rms->src[0];
    q4e_rms_scale_job j;
    j.x         = (const float *) x->data;
    j.dst       = (float *) scale->data;
    j.sx1       = x->nb[1]/sizeof(float);
    j.sx2       = x->nb[2]/sizeof(float);
    j.sx3       = x->nb[3]/sizeof(float);
    j.nrows     = (int) x->ne[1];
    j.nchannels = (int) x->ne[2];
    j.nblocks   = (int) (x->ne[1]*x->ne[2]*x->ne[3]);
    memcpy(&j.eps, rms->op_params, sizeof(float));
    j.scale     = ggml_get_op_params_f32(scale, 0);
    return j;
}

void ggml_cuda_op_q4e_rms_scale2(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rms0, ggml_tensor * scale0, const ggml_tensor * rms1, ggml_tensor * scale1) {
    const q4e_rms_scale_job j0 = q4e_rms_scale_job_of(rms0, scale0);
    const q4e_rms_scale_job j1 = q4e_rms_scale_job_of(rms1, scale1);
    const int ncols = (int) rms0->src[0]->ne[0];
    const dim3 grid(j0.nblocks + j1.nblocks, 1, 1);
    cudaStream_t stream = ctx.stream();

    switch (q4e_rms_block_size(ncols)) {
        case 32:   q4e_rms_scale2_kernel<32>  <<<grid, 32,   0, stream>>>(j0, j1, ncols); break;
        case 128:  q4e_rms_scale2_kernel<128> <<<grid, 128,  0, stream>>>(j0, j1, ncols); break;
        case 256:  q4e_rms_scale2_kernel<256> <<<grid, 256,  0, stream>>>(j0, j1, ncols); break;
        case 512:  q4e_rms_scale2_kernel<512> <<<grid, 512,  0, stream>>>(j0, j1, ncols); break;
        case 1024: q4e_rms_scale2_kernel<1024><<<grid, 1024, 0, stream>>>(j0, j1, ncols); break;
        default: GGML_ABORT("unexpected row size");
    }
}
