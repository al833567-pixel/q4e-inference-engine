#include "q4e.cuh"

// q4e expert tiers, see ggml_q4e_route and ggml_q4e_wsum in ggml.h

template <bool weights>
static __global__ void q4e_route_kernel(
        const char * sel, const char * w, const float * table, void * dst,
        const int64_t sel_nb0, const int64_t sel_nb1, const int64_t w_nb1, const int64_t w_nb2,
        const int n_used, const int n_blocks, const int zero_slot, const int64_t n_virt, const int n_rows, const int n_tokens,
        const bool neg_miss) {
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
    const int n_used    = (int) sel->ne[0];
    const int n_tokens  = (int) sel->ne[1];
    const int n_rows    = n_used*n_blocks;
    const int n         = n_rows*n_tokens;
    const int block     = 256;
    const dim3 grid((n + block - 1)/block);
    cudaStream_t stream = ctx.stream();
    if (weights) {
        q4e_route_kernel<true><<<grid, block, 0, stream>>>((const char *) sel->data, (const char *) w->data, (const float *) table->data, dst->data,
            sel->nb[0], sel->nb[1], w->nb[1], w->nb[2], n_used, n_blocks, zero_slot, ggml_nelements(table), n_rows, n_tokens, neg_miss);
    } else {
        q4e_route_kernel<false><<<grid, block, 0, stream>>>((const char *) sel->data, (const char *) w->data, (const float *) table->data, dst->data,
            sel->nb[0], sel->nb[1], w->nb[1], w->nb[2], n_used, n_blocks, zero_slot, ggml_nelements(table), n_rows, n_tokens, neg_miss);
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