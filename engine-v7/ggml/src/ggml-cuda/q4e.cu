#include "q4e.cuh"

// q4e expert tiers, see ggml_q4e_route and ggml_q4e_wsum in ggml.h

template <bool weights>
static __global__ void q4e_route_kernel(
        const char * sel, const char * w, const float * table, void * dst,
        const int64_t sel_nb0, const int64_t sel_nb1, const int64_t w_nb1, const int64_t w_nb2,
        const int n_used, const int n_blocks, const int zero_slot, const int64_t n_virt, const int n_rows, const int n_tokens,
        const bool neg_miss, const bool rest, const float skip) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_rows*n_tokens) {
        return;
    }
    const int t = i / n_rows;
    const int r = i % n_rows;
    const int k = r / n_blocks;
    const int b = r % n_blocks;
    const int32_t e = *(const int32_t *) (sel + k*sel_nb0 + t*sel_nb1);
    const int64_t v = (int64_t) e*n_blocks + b;
    const float slot = v >= 0 && v < n_virt ? table[v] : -1.0f;
    if (rest) {
        // the blocks of the other tier (ggml_q4e_route_rest), in the order of the CPU reference
        float wsum = 0.0f;
        for (int kk = 0; kk < n_used; ++kk) {
            wsum += *(const float *) (w + kk*w_nb1 + t*w_nb2);
        }
        const float wk = *(const float *) (w + k*w_nb1 + t*w_nb2);
        const bool mine = v >= 0 && v < n_virt && slot < 0.0f && wk >= skip*wsum;
        if (weights) {
            ((float *) dst)[i] = mine ? wk : 0.0f;
        } else {
            ((int32_t *) dst)[i] = mine ? (int32_t) v : -1;
        }
        return;
    }
    if (weights) {
        const float wk = *(const float *) (w + k*w_nb1 + t*w_nb2);
        ((float *) dst)[i] = slot >= 0.0f ? wk : 0.0f;
    } else {
        ((int32_t *) dst)[i] = slot >= 0.0f ? (int32_t) slot : neg_miss ? -1 : zero_slot + r;
    }
}

void ggml_cuda_op_q4e_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel   = dst->src[0];
    const ggml_tensor * w     = dst->src[1];
    const ggml_tensor * table = dst->src[2];
    const int n_blocks  = ggml_get_op_params_i32(dst, 0);
    const int zero_slot = ggml_get_op_params_i32(dst, 1);
    const bool weights  = ggml_get_op_params_i32(dst, 2) != 0;
    const bool neg_miss = ggml_get_op_params_i32(dst, 3) != 0;
    const bool rest     = ggml_get_op_params_i32(dst, 4) != 0;
    const float skip    = ggml_get_op_params_f32(dst, 5);
    const int n_used    = (int) sel->ne[0];
    const int n_tokens  = (int) sel->ne[1];
    const int n_rows    = n_used*n_blocks;
    const int n         = n_rows*n_tokens;
    const int block     = 256;
    const dim3 grid((n + block - 1)/block);
    cudaStream_t stream = ctx.stream();
    if (weights) {
        q4e_route_kernel<true><<<grid, block, 0, stream>>>((const char *) sel->data, (const char *) w->data, (const float *) table->data, dst->data,
            sel->nb[0], sel->nb[1], w->nb[1], w->nb[2], n_used, n_blocks, zero_slot, ggml_nelements(table), n_rows, n_tokens, neg_miss, rest, skip);
    } else {
        q4e_route_kernel<false><<<grid, block, 0, stream>>>((const char *) sel->data, (const char *) w->data, (const float *) table->data, dst->data,
            sel->nb[0], sel->nb[1], w->nb[1], w->nb[2], n_used, n_blocks, zero_slot, ggml_nelements(table), n_rows, n_tokens, neg_miss, rest, skip);
    }
}

// one thread per output element, the rows of a token are summed in order
static __global__ void q4e_wsum_kernel(
        const char * y, const char * w, float * dst,
        const int64_t y_nb1, const int64_t y_nb2, const int64_t w_nb1, const int64_t w_nb2,
        const int n_embd, const int n_rows) {
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (e >= n_embd) {
        return;
    }
    float sum = 0.0f;
    for (int r = 0; r < n_rows; ++r) {
        const float wr = *(const float *) (w + r*w_nb1 + t*w_nb2);
        if (wr != 0.0f) {
            sum += wr*((const float *) (y + r*y_nb1 + t*y_nb2))[e];
        }
    }
    dst[(int64_t) t*n_embd + e] = sum;
}

void ggml_cuda_op_q4e_wsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * y = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const int n_embd   = (int) y->ne[0];
    const int n_rows   = (int) y->ne[1];
    const int n_tokens = (int) y->ne[2];
    const int block    = 256;
    const dim3 grid((n_embd + block - 1)/block, n_tokens);
    q4e_wsum_kernel<<<grid, block, 0, ctx.stream()>>>((const char *) y->data, (const char *) w->data, (float *) dst->data,
        y->nb[1], y->nb[2], w->nb[1], w->nb[2], n_embd, n_rows);
}
// silu(s*x + b)
static __global__ void q4e_scale_silu_kernel(const float * x, float * y, const float s, const float b, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = s*x[i] + b;
        y[i] = v/(1.0f + expf(-v));
    }
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale, ggml_tensor * silu) {
    const ggml_tensor * x = scale->src[0];
    const int n = (int) ggml_nelements(silu);
    const float s = ggml_get_op_params_f32(scale, 0);
    const float b = ggml_get_op_params_f32(scale, 1);
    q4e_scale_silu_kernel<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) x->data, (float *) silu->data, s, b, n);
}

// a[e, t] * sigmoid(g[t]): the shared expert gate is one value per token
static __global__ void q4e_mul_sigmoid_bcast_kernel(const float * a, const float * g, float * y, const int ne0, const int64_t g_stride, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        const float gv = g[(i / ne0)*g_stride];
        y[i] = a[i]/(1.0f + expf(-gv));
    }
}

void ggml_cuda_op_mul_sigmoid_bcast(ggml_backend_cuda_context & ctx, ggml_tensor * sigmoid, ggml_tensor * mul) {
    const ggml_tensor * g = sigmoid->src[0];
    const ggml_tensor * a = mul->src[0] == sigmoid ? mul->src[1] : mul->src[0];
    const int n = (int) ggml_nelements(mul);
    q4e_mul_sigmoid_bcast_kernel<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) a->data, (const float *) g->data, (float *) mul->data,
        (int) mul->ne[0], g->nb[1]/sizeof(float), n);
}
// qwen4exp hyper-connection mix (ggml_q4e_hc_mix) for up to Q4E_HC_MAX_T tokens with Q8_0 weights, in two kernels:
//   down: one warp per row of the down projection, the activations of a K tile in shared memory, silu(scale*dot)
//   up:   one warp per output element e, the hc rows c*n_embd + e of the up projection, sigmoid gate and the mean
#define Q4E_HC_MAX_T  4
#define Q4E_HC_TILE   2048
#define Q4E_HC_WARPS  8

template <int n_t>
static __global__ void q4e_hc_down_kernel(const float * xn, const block_q8_0 * w, const int64_t w_stride_blocks,
        float * lo, const int n_hc, const int n_lr, const float scale) {
    __shared__ float xs[n_t][Q4E_HC_TILE];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int r    = blockIdx.x*Q4E_HC_WARPS + warp;
    float acc[n_t] = { 0.0f };
    for (int k0 = 0; k0 < n_hc; k0 += Q4E_HC_TILE) {
        const int nk = min(Q4E_HC_TILE, n_hc - k0);
        __syncthreads();
        for (int i = warp*32 + lane; i < nk; i += 32*Q4E_HC_WARPS) {
#pragma unroll
            for (int t = 0; t < n_t; ++t) {
                xs[t][i] = xn[(int64_t) t*n_hc + k0 + i];
            }
        }
        __syncthreads();
        if (r < n_lr) {
            const block_q8_0 * row = w + r*w_stride_blocks + k0/QK8_0;
            for (int b = lane; b < nk/QK8_0; b += 32) {
                const float d = __half2float(row[b].d);
                const int8_t * q = row[b].qs;
                float s[n_t] = { 0.0f };
#pragma unroll
                for (int j = 0; j < QK8_0; ++j) {
                    const float qj = (float) q[j];
#pragma unroll
                    for (int t = 0; t < n_t; ++t) {
                        s[t] += qj*xs[t][b*QK8_0 + j];
                    }
                }
#pragma unroll
                for (int t = 0; t < n_t; ++t) {
                    acc[t] += d*s[t];
                }
            }
        }
    }
    if (r >= n_lr) {
        return;
    }
#pragma unroll
    for (int t = 0; t < n_t; ++t) {
        const float v = scale*warp_reduce_sum<32>(acc[t]);
        if (lane == 0) {
            lo[t*n_lr + r] = v/(1.0f + expf(-v));
        }
    }
}

template <int n_t>
static __global__ void q4e_hc_up_kernel(const float * xn, const float * lo, const block_q8_0 * w, const int64_t w_stride_blocks,
        float * dst, const int n_embd, const int hc, const int n_lr, const float scale) {
    extern __shared__ float los[]; // [n_t][n_lr]
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    for (int i = warp*32 + lane; i < n_t*n_lr; i += 32*Q4E_HC_WARPS) {
        los[i] = lo[i];
    }
    __syncthreads();
    const int e = blockIdx.x*Q4E_HC_WARPS + warp;
    if (e >= n_embd) {
        return;
    }
    const int n_hc = n_embd*hc;
    const int nb   = n_lr/QK8_0;
    float out[n_t] = { 0.0f };
    for (int c = 0; c < hc; ++c) {
        const int r = c*n_embd + e;
        const block_q8_0 * row = w + r*w_stride_blocks;
        float s[n_t] = { 0.0f };
        for (int b = 0; b < nb; ++b) {
            const float v = __half2float(row[b].d)*(float) row[b].qs[lane];
#pragma unroll
            for (int t = 0; t < n_t; ++t) {
                s[t] += v*los[t*n_lr + b*QK8_0 + lane];
            }
        }
#pragma unroll
        for (int t = 0; t < n_t; ++t) {
            const float g = warp_reduce_sum<32>(s[t]);
            out[t] += xn[(int64_t) t*n_hc + r]/(1.0f + expf(-g));
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < n_t; ++t) {
            dst[(int64_t) t*n_embd + e] = scale*out[t];
        }
    }
}

bool ggml_cuda_q4e_hc_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * xn = op->src[0];
    const ggml_tensor * down = op->src[1];
    const ggml_tensor * up = op->src[2];
    return xn->type == GGML_TYPE_F32 && ggml_is_contiguous(xn) && xn->ne[1] <= Q4E_HC_MAX_T &&
        down->type == GGML_TYPE_Q8_0 && up->type == GGML_TYPE_Q8_0 &&
        down->nb[0] == sizeof(block_q8_0) && up->nb[0] == sizeof(block_q8_0) &&
        xn->ne[0] % QK8_0 == 0 && up->ne[0] % QK8_0 == 0 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
}

template <int n_t>
static void q4e_hc_mix_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * xn = dst->src[0];
    const ggml_tensor * down = dst->src[1];
    const ggml_tensor * up = dst->src[2];
    const int   hc     = ggml_get_op_params_i32(dst, 0);
    const float scale  = ggml_get_op_params_f32(dst, 1);
    const int   n_hc   = (int) xn->ne[0];
    const int   n_embd = n_hc/hc;
    const int   n_lr   = (int) down->ne[1];
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> lo(ctx.pool(), (size_t) n_t*n_lr);
    const dim3 block(32, Q4E_HC_WARPS);
    q4e_hc_down_kernel<n_t><<<(n_lr + Q4E_HC_WARPS - 1)/Q4E_HC_WARPS, block, 0, stream>>>(
        (const float *) xn->data, (const block_q8_0 *) down->data, down->nb[1]/sizeof(block_q8_0), lo.get(), n_hc, n_lr, scale);
    q4e_hc_up_kernel<n_t><<<(n_embd + Q4E_HC_WARPS - 1)/Q4E_HC_WARPS, block, (size_t) n_t*n_lr*sizeof(float), stream>>>(
        (const float *) xn->data, lo.get(), (const block_q8_0 *) up->data, up->nb[1]/sizeof(block_q8_0), (float *) dst->data,
        n_embd, hc, n_lr, scale);
}

void ggml_cuda_op_q4e_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[1]) {
        case 1: q4e_hc_mix_launch<1>(ctx, dst); break;
        case 2: q4e_hc_mix_launch<2>(ctx, dst); break;
        case 3: q4e_hc_mix_launch<3>(ctx, dst); break;
        case 4: q4e_hc_mix_launch<4>(ctx, dst); break;
        default: GGML_ABORT("q4e_hc_mix: up to %d tokens", Q4E_HC_MAX_T);
    }
}