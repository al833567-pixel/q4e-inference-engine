#pragma once

#include "common.cuh"

// q4e: CUDA-only fusions of the small ops of a qwen4exp decode step (the graph is unchanged, see ggml_cuda_try_fuse)

#define Q4E_FUSE_MAX_SLOTS 8

// concat(state, x^T) along dim 0, the rollback-slot copies of its tail and optionally ssm_conv + silu
struct q4e_conv_state_args {
    const ggml_tensor * concat;
    int                 n_slots;
    int                 slot_off[Q4E_FUSE_MAX_SLOTS]; // first column of each slot's tail in the concat rows
    float             * slot_dst[Q4E_FUSE_MAX_SLOTS]; // [state_cols, channels] destinations, one per slot
    int64_t             slot_s_stride;                 // floats between the sequences of a destination
    const ggml_tensor * conv_w;                        // ssm_conv weights, nullptr when the conv is not fused
    ggml_tensor       * conv_out;                      // the silu node
};
void ggml_cuda_op_q4e_conv_state(ggml_backend_cuda_context & ctx, const q4e_conv_state_args & args);

// gate = softplus(mm_a(y) + dt)*a and beta = sigmoid(mm_b(y)): two bf16 mat-vecs with the same rows as mul_mat_vec_f
void ggml_cuda_op_q4e_gdn_gates(ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_a, const ggml_tensor * w_b, const ggml_tensor * y,
        const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta);
bool ggml_cuda_q4e_gdn_gates_supported(const ggml_tensor * w_a, const ggml_tensor * y);

// dsv4_hc_post (with the fused sigmoid gate) followed by rms_norm*w of each stream
void ggml_cuda_op_q4e_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out,
        const ggml_tensor * rms_norm, ggml_tensor * mul);
bool ggml_cuda_q4e_rms_vec_ok(const ggml_tensor * x, const ggml_tensor * mul, const ggml_tensor * dst);

// rms_norm(x)*w*sigmoid(z)
void ggml_cuda_op_q4e_norm_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
        const ggml_tensor * sigmoid, ggml_tensor * out);

// two independent rms_norm -> scale pairs in one launch
void ggml_cuda_op_q4e_rms_scale2(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rms0, ggml_tensor * scale0, const ggml_tensor * rms1, ggml_tensor * scale1);
