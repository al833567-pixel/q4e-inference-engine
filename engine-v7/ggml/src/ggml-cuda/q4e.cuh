#include "common.cuh"

void ggml_cuda_op_q4e_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_q4e_wsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// fused element-wise pairs left by qwen4exp (CUDA only, the graph keeps the plain ops for the other backends)
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale, ggml_tensor * silu);
void ggml_cuda_op_mul_sigmoid_bcast(ggml_backend_cuda_context & ctx, ggml_tensor * sigmoid, ggml_tensor * mul);