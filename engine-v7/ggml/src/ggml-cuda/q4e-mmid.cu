#include "q4e-mmid.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

// The q4e GPU tier per MoE layer and 1-4 tokens: n_ids = 10 experts x 2 blocks of 320 FFN rows per token, ids into a
// pool of blocks, -1 for a block that is not in VRAM. The generic mmvq kernels launch one CUDA block per (rows, id,
// token) and read a block that several tokens selected once per token; the down projection writes [n_embd, n_ids,
// n_tokens] rows that Q4E_WSUM reads back. Here:
//   gate/up: a CUDA block per (8 rows, id, token); the first token that selected a block computes it for every token
//            that selected it, the others exit. 16 lanes per row, the gate and up dots share the activation loads.
//   down:    a CUDA block per 4 rows of the output for all tokens; it lists the distinct blocks of all tokens in
//            shared memory and sums w * dot over them, so a row of a block is read once and y never exists.

#define Q4E_MMID_THREADS 128

template <ggml_type type> struct q4e_mmid_traits;
template <> struct q4e_mmid_traits<GGML_TYPE_IQ2_S>   { static constexpr int vdr = VDR_IQ2_S_Q8_1_MMVQ;   };
template <> struct q4e_mmid_traits<GGML_TYPE_IQ2_XXS> { static constexpr int vdr = VDR_IQ2_XXS_Q8_1_MMVQ; };
template <> struct q4e_mmid_traits<GGML_TYPE_IQ2_XS>  { static constexpr int vdr = VDR_IQ2_XS_Q8_1_MMVQ;  };
template <> struct q4e_mmid_traits<GGML_TYPE_IQ1_M>   { static constexpr int vdr = VDR_IQ1_M_Q8_1_MMVQ;   };
template <> struct q4e_mmid_traits<GGML_TYPE_Q2_0>    { static constexpr int vdr = VDR_Q2_0_Q8_1_MMVQ;    };

template <ggml_type type>
static __device__ __forceinline__ float q4e_mmid_vec_dot(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
        const int & kbx, const int & iqs) {
    if constexpr (type == GGML_TYPE_IQ2_S) {
        return vec_dot_iq2_s_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        return vec_dot_iq2_xs_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ1_M) {
        return vec_dot_iq1_m_q8_1(vbq, bq8_1, kbx, iqs);
    } else {
        static_assert(type == GGML_TYPE_Q2_0, "q4e_mmid: type");
        return vec_dot_q2_0_q8_1(vbq, bq8_1, kbx, iqs);
    }
}

bool ggml_cuda_q4e_mmid_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_Q4E_MMID"); return e == nullptr || atoi(e) != 0; }();
    return on;
}

bool ggml_cuda_q4e_mmid_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_Q2_0:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ float q4e_mmid_glu(const ggml_glu_op op, const float gate, const float up, const float limit) {
    switch (op) {
        case GGML_GLU_OP_SWIGLU:       return up*ggml_cuda_op_silu_single(gate);
        case GGML_GLU_OP_GEGLU:        return up*ggml_cuda_op_gelu_single(gate);
        case GGML_GLU_OP_SWIGLU_OAI:   return ggml_cuda_op_swiglu_oai_single(gate, up);
        case GGML_GLU_OP_SWIGLU_CLAMP: return ggml_cuda_op_swiglu_clamp_single(gate, up, limit);
        default:                       return up*gate;
    }
}

template <ggml_type type, int ntok, int gsize>
static __global__ void __launch_bounds__(Q4E_MMID_THREADS) q4e_mmid_gate_up_kernel(
        const char * __restrict__ vgate, const char * __restrict__ vup, const int64_t nb02, const int stride_row_x,
        const int blocks_per_row, const block_q8_1 * __restrict__ y, const int stride_col_y,
        const int32_t * __restrict__ ids, const int ids_stride, const int n_ids,
        float * __restrict__ dst, const int64_t stride_channel_dst, const int64_t stride_col_dst, const int nrows,
        const ggml_glu_op glu_op, const float glu_limit) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = q4e_mmid_traits<type>::vdr;
    constexpr int ipb = qi/vdr; // dot items per quant block
    constexpr int rows_per_block = Q4E_MMID_THREADS/gsize;

    const int t    = blockIdx.y / n_ids;
    const int j    = blockIdx.y - t*n_ids;
    const int lane = threadIdx.x % gsize;
    const int row  = blockIdx.x*rows_per_block + threadIdx.x/gsize;

    const int id = ids[t*ids_stride + j];
    if (id < 0) {
        // not in VRAM: zero rows
        if (lane == 0 && row < nrows) {
            dst[t*stride_col_dst + j*stride_channel_dst + row] = 0.0f;
        }
        return;
    }

    // the id of this block in each token, the first token that selected the block computes all of them
    int jt[ntok];
    if constexpr (ntok == 1) {
        jt[0] = j;
    } else {
        __shared__ int s_jt[ntok];
        if (threadIdx.x < ntok) {
            s_jt[threadIdx.x] = -1;
        }
        __syncthreads();
        for (int k = threadIdx.x; k < ntok*n_ids; k += blockDim.x) {
            const int tt = k / n_ids;
            const int jj = k - tt*n_ids;
            if (ids[tt*ids_stride + jj] == id) {
                s_jt[tt] = jj;
            }
        }
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < ntok; ++tt) {
            jt[tt] = s_jt[tt];
            if (tt < t && jt[tt] >= 0) {
                return;
            }
        }
    }

    float sum_g[ntok] = { 0.0f };
    float sum_u[ntok] = { 0.0f };
    if (row < nrows) {
        const char * xg = vgate + id*nb02;
        const char * xu = vup   + id*nb02;
        const int kbr = row*stride_row_x;
        const int n_items = blocks_per_row*ipb;
        for (int c = lane; c < n_items; c += gsize) {
            const int kbx = c / ipb;
            const int kqs = vdr*(c - kbx*ipb);
#pragma unroll
            for (int tt = 0; tt < ntok; ++tt) {
                if (jt[tt] >= 0) {
                    const block_q8_1 * yt = y + tt*stride_col_y + kbx*(qk/QK8_1);
                    sum_g[tt] += q4e_mmid_vec_dot<type>(xg, yt, kbr + kbx, kqs);
                    sum_u[tt] += q4e_mmid_vec_dot<type>(xu, yt, kbr + kbx, kqs);
                }
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < ntok; ++tt) {
#pragma unroll
        for (int off = gsize/2; off > 0; off >>= 1) {
            sum_g[tt] += __shfl_xor_sync(0xffffffff, sum_g[tt], off, 32);
            sum_u[tt] += __shfl_xor_sync(0xffffffff, sum_u[tt], off, 32);
        }
    }
    if (lane == 0 && row < nrows) {
#pragma unroll
        for (int tt = 0; tt < ntok; ++tt) {
            if (jt[tt] >= 0) {
                dst[tt*stride_col_dst + jt[tt]*stride_channel_dst + row] = q4e_mmid_glu(glu_op, sum_g[tt], sum_u[tt], glu_limit);
            }
        }
    }
}

template <ggml_type type, int ntok, int gsize>
static __global__ void __launch_bounds__(Q4E_MMID_THREADS) q4e_mmid_down_wsum_kernel(
        const char * __restrict__ vx, const int64_t nb02, const int stride_row_x, const int blocks_per_row,
        const block_q8_1 * __restrict__ y, const int stride_channel_y, const int stride_col_y,
        const int32_t * __restrict__ ids, const int n_ids,
        const char * __restrict__ w, const int64_t w_nb1, const int64_t w_nb2,
        float * __restrict__ dst, const int64_t stride_col_dst, const int nrows) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = q4e_mmid_traits<type>::vdr;
    constexpr int ipb = qi/vdr;
    constexpr int rows_per_block = Q4E_MMID_THREADS/gsize;

    // the distinct blocks of all tokens, in the order of their first selection (deterministic sums)
    __shared__ int   s_id[Q4E_MMID_MAX_PAIRS];
    __shared__ int   s_y[Q4E_MMID_MAX_PAIRS][ntok]; // offset of h[:, r, t] in y, -1: token t did not select the block
    __shared__ float s_w[Q4E_MMID_MAX_PAIRS][ntok];
    __shared__ int   s_first[Q4E_MMID_MAX_PAIRS];
    __shared__ int   s_ids[Q4E_MMID_MAX_PAIRS];

    const int n_pairs = ntok*n_ids;
    for (int k = threadIdx.x; k < n_pairs; k += blockDim.x) {
        s_ids[k] = ids[k];
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n_pairs; k += blockDim.x) {
        const int id = s_ids[k];
        bool first = id >= 0;
        for (int k2 = 0; k2 < k - k % n_ids && first; ++k2) {
            first = s_ids[k2] != id;
        }
        s_first[k] = first;
    }
    __syncthreads();
    // every thread counts, the thread of pair k fills its entry
    int n_uniq = 0;
    for (int k = 0; k < n_pairs; ++k) {
        if (s_first[k]) {
            if (k % blockDim.x == threadIdx.x) {
                const int id = s_ids[k];
                const int t0 = k / n_ids;
                s_id[n_uniq] = id;
#pragma unroll
                for (int tt = 0; tt < ntok; ++tt) {
                    int jj = -1;
                    if (tt == t0) {
                        jj = k - t0*n_ids;
                    } else if (tt > t0) {
                        for (int j2 = 0; j2 < n_ids; ++j2) {
                            if (s_ids[tt*n_ids + j2] == id) {
                                jj = j2;
                            }
                        }
                    }
                    s_y[n_uniq][tt] = jj >= 0 ? jj*stride_channel_y + tt*stride_col_y : -1;
                    s_w[n_uniq][tt] = jj >= 0 ? *(const float *) (w + jj*w_nb1 + tt*w_nb2) : 0.0f;
                }
            }
            n_uniq++;
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % gsize;
    const int row  = blockIdx.x*rows_per_block + threadIdx.x/gsize;
    const int ipr  = blocks_per_row*ipb; // dot items per row of a block

    float sum[ntok] = { 0.0f };
    if (row < nrows) {
        const int kbr = row*stride_row_x;
        const int n_items = n_uniq*ipr;
        for (int it = lane; it < n_items; it += gsize) {
            const int u   = it / ipr;
            const int c   = it - u*ipr;
            const int kbx = c / ipb;
            const int kqs = vdr*(c - kbx*ipb);
            const char * xb = vx + s_id[u]*nb02;
#pragma unroll
            for (int tt = 0; tt < ntok; ++tt) {
                const int yo = s_y[u][tt];
                if (yo >= 0) {
                    sum[tt] += s_w[u][tt]*q4e_mmid_vec_dot<type>(xb, y + yo + kbx*(qk/QK8_1), kbr + kbx, kqs);
                }
            }
        }
    }
#pragma unroll
    for (int tt = 0; tt < ntok; ++tt) {
#pragma unroll
        for (int off = gsize/2; off > 0; off >>= 1) {
            sum[tt] += __shfl_xor_sync(0xffffffff, sum[tt], off, 32);
        }
    }
    if (lane == 0 && row < nrows) {
#pragma unroll
        for (int tt = 0; tt < ntok; ++tt) {
            dst[tt*stride_col_dst + row] = sum[tt];
        }
    }
}

// lanes per row: env GGML_CUDA_Q4E_MMID_G (gate/up) and GGML_CUDA_Q4E_MMID_GD (down) for tuning
static int q4e_mmid_gsize(const char * name, int def) {
    const char * e = getenv(name);
    const int g = e ? atoi(e) : def;
    return g == 4 || g == 8 || g == 16 || g == 32 ? g : def;
}

template <ggml_type type, int ntok>
static void q4e_mmid_gate_up_launch(const int gsize, const dim3 & grid, cudaStream_t stream,
        const char * vgate, const char * vup, int64_t nb02, int stride_row_x, int blocks_per_row,
        const block_q8_1 * y, int stride_col_y, const int32_t * ids, int ids_stride, int n_ids,
        float * dst, int64_t s_ch, int64_t s_col, int nrows, ggml_glu_op op, float limit) {
#define Q4E_GU_LAUNCH(G) q4e_mmid_gate_up_kernel<type, ntok, G><<<dim3((nrows + Q4E_MMID_THREADS/G - 1)/(Q4E_MMID_THREADS/G), grid.y), Q4E_MMID_THREADS, 0, stream>>>( \
        vgate, vup, nb02, stride_row_x, blocks_per_row, y, stride_col_y, ids, ids_stride, n_ids, dst, s_ch, s_col, nrows, op, limit)
    switch (gsize) {
        case 8:  Q4E_GU_LAUNCH(8);  break;
        case 32: Q4E_GU_LAUNCH(32); break;
        default: Q4E_GU_LAUNCH(16); break;
    }
#undef Q4E_GU_LAUNCH
}

template <ggml_type type, int ntok>
static void q4e_mmid_down_launch(const int gsize, cudaStream_t stream,
        const char * vx, int64_t nb02, int stride_row_x, int blocks_per_row,
        const block_q8_1 * y, int stride_channel_y, int stride_col_y, const int32_t * ids, int n_ids,
        const char * w, int64_t w_nb1, int64_t w_nb2, float * dst, int64_t s_col, int nrows) {
#define Q4E_DW_LAUNCH(G) q4e_mmid_down_wsum_kernel<type, ntok, G><<<(nrows + Q4E_MMID_THREADS/G - 1)/(Q4E_MMID_THREADS/G), Q4E_MMID_THREADS, 0, stream>>>( \
        vx, nb02, stride_row_x, blocks_per_row, y, stride_channel_y, stride_col_y, ids, n_ids, w, w_nb1, w_nb2, dst, s_col, nrows)
    switch (gsize) {
        case 8:  Q4E_DW_LAUNCH(8);  break;
        case 16: Q4E_DW_LAUNCH(16); break;
        default: Q4E_DW_LAUNCH(32); break;
    }
#undef Q4E_DW_LAUNCH
}

#define Q4E_MMID_SWITCH(TYPE_EXPR, NTOK, FN, ...)                                                   \
    do {                                                                                          \
        switch (TYPE_EXPR) {                                                                      \
            case GGML_TYPE_IQ2_S:   Q4E_MMID_SWITCH_T(GGML_TYPE_IQ2_S,   NTOK, FN, __VA_ARGS__); break; \
            case GGML_TYPE_IQ2_XXS: Q4E_MMID_SWITCH_T(GGML_TYPE_IQ2_XXS, NTOK, FN, __VA_ARGS__); break; \
            case GGML_TYPE_IQ2_XS:  Q4E_MMID_SWITCH_T(GGML_TYPE_IQ2_XS,  NTOK, FN, __VA_ARGS__); break; \
            case GGML_TYPE_IQ1_M:   Q4E_MMID_SWITCH_T(GGML_TYPE_IQ1_M,   NTOK, FN, __VA_ARGS__); break; \
            case GGML_TYPE_Q2_0:    Q4E_MMID_SWITCH_T(GGML_TYPE_Q2_0,    NTOK, FN, __VA_ARGS__); break; \
            default: GGML_ABORT("q4e_mmid: unsupported type");                                      \
        }                                                                                         \
    } while (0)

#define Q4E_MMID_SWITCH_T(TYPE, NTOK, FN, ...)                    \
    do {                                                          \
        switch (NTOK) {                                           \
            case 1: FN<TYPE, 1>(__VA_ARGS__); break;              \
            case 2: FN<TYPE, 2>(__VA_ARGS__); break;              \
            case 3: FN<TYPE, 3>(__VA_ARGS__); break;              \
            case 4: FN<TYPE, 4>(__VA_ARGS__); break;              \
            default: GGML_ABORT("q4e_mmid: 1-4 tokens");           \
        }                                                         \
    } while (0)

void ggml_cuda_q4e_mmid_gate_up(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, ggml_glu_op glu_op, float glu_limit) {
    const ggml_type type = up_w->type;
    GGML_ASSERT(gate_w->type == type && ggml_are_same_stride(gate_w, up_w));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src1) && ggml_is_contiguous(dst));

    cudaStream_t stream = ctx.stream();
    const int64_t ne00     = up_w->ne[0];
    const int64_t nrows    = up_w->ne[1];
    const int     n_tokens = (int) src1->ne[2];
    const int     n_ids    = (int) ids->ne[0];
    const size_t  ts       = ggml_type_size(type);

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> y(ctx.pool(), n_tokens*ne10_padded*sizeof(block_q8_1)/QK8_1);
    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, y.get(), type, ne00, src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float),
        src1->nb[3]/sizeof(float), ne10_padded, 1, n_tokens, 1, stream);

    static const int gsize = q4e_mmid_gsize("GGML_CUDA_Q4E_MMID_G", 16);
    const dim3 grid(1, n_ids*n_tokens);
    Q4E_MMID_SWITCH(type, n_tokens, q4e_mmid_gate_up_launch, gsize, grid, stream,
        (const char *) gate_w->data, (const char *) up_w->data, (int64_t) up_w->nb[2], (int) (up_w->nb[1]/ts),
        (int) (ne00/ggml_blck_size(type)), (const block_q8_1 *) y.get(), (int) (ne10_padded/QK8_1),
        (const int32_t *) ids->data, (int) (ids->nb[1]/sizeof(int32_t)), n_ids,
        (float *) dst->data, (int64_t) (dst->nb[1]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)), (int) nrows, glu_op, glu_limit);
}

void ggml_cuda_q4e_mmid_down_wsum(ggml_backend_cuda_context & ctx, const ggml_tensor * down_w, const ggml_tensor * h,
        const ggml_tensor * ids, const ggml_tensor * w, ggml_tensor * dst, const std::function<void()> & before) {
    const ggml_type type = down_w->type;
    GGML_ASSERT(h->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && w->type == GGML_TYPE_F32);
    GGML_ASSERT(h->ne[3] == 1 && ggml_is_contiguous(dst));

    cudaStream_t stream = ctx.stream();
    const int64_t ne00     = down_w->ne[0];
    const int64_t nrows    = down_w->ne[1];
    const int     n_ids    = (int) h->ne[1];
    const int     n_tokens = (int) h->ne[2];
    const size_t  ts       = ggml_type_size(type);
    GGML_ASSERT(ids->ne[0] == n_ids && ids->ne[1] == n_tokens && n_ids*n_tokens <= Q4E_MMID_MAX_PAIRS);

    // h quantized and the ids packed: before() may reuse their memory (the graph considers them dead after down)
    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> y(ctx.pool(), n_tokens*n_ids*ne10_padded*sizeof(block_q8_1)/QK8_1);
    quantize_row_q8_1_cuda((const float *) h->data, nullptr, y.get(), type, ne00, h->nb[1]/sizeof(float), h->nb[2]/sizeof(float),
        h->nb[3]/sizeof(float), ne10_padded, n_ids, n_tokens, 1, stream);
    ggml_cuda_pool_alloc<int32_t> ids_c(ctx.pool(), n_ids*n_tokens);
    CUDA_CHECK(cudaMemcpy2DAsync(ids_c.get(), n_ids*sizeof(int32_t), ids->data, ids->nb[1], n_ids*sizeof(int32_t), n_tokens,
        cudaMemcpyDeviceToDevice, stream));

    before();

    static const int gsize = q4e_mmid_gsize("GGML_CUDA_Q4E_MMID_GD", 32);
    const int stride_channel_y = (int) (ne10_padded/QK8_1);
    Q4E_MMID_SWITCH(type, n_tokens, q4e_mmid_down_launch, gsize, stream,
        (const char *) down_w->data, (int64_t) down_w->nb[2], (int) (down_w->nb[1]/ts), (int) (ne00/ggml_blck_size(type)),
        (const block_q8_1 *) y.get(), stride_channel_y, stride_channel_y*n_ids, ids_c.get(), n_ids,
        (const char *) w->data, (int64_t) w->nb[1], (int64_t) w->nb[2], (float *) dst->data, (int64_t) (dst->nb[1]/sizeof(float)), (int) nrows);
}
