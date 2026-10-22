#pragma once

#include "common.cuh"

#include <functional>

// q4e GPU expert tier for 1-4 tokens (MUL_MAT_ID over a pool of expert blocks, negative ids = blocks not in VRAM):
//   gate/up: MUL_MAT_ID gate + MUL_MAT_ID up + GLU in one kernel, a block that several tokens select is read once
//   down:    MUL_MAT_ID down + Q4E_WSUM in one kernel, the per-block rows y are never written

#define Q4E_MMID_MAX_TOKENS 4
#define Q4E_MMID_MAX_PAIRS  128 // ids per token x tokens

bool ggml_cuda_q4e_mmid_enabled();
bool ggml_cuda_q4e_mmid_type_supported(ggml_type type);

// dst = glu(gate_w x src1, up_w x src1) for the ids, src1 [ne10, 1, n_tokens], dst [ne01, n_ids, n_tokens]
void ggml_cuda_q4e_mmid_gate_up(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, ggml_glu_op glu_op, float glu_limit);

// dst[:, t] = sum_r w[r, t] * (down_w[ids[r, t]] x h[:, r, t]), h [ne00, n_ids, n_tokens], dst [ne01, n_tokens].
// before() runs after h and ids were copied out (it may overwrite them) and before w is read
void ggml_cuda_q4e_mmid_down_wsum(ggml_backend_cuda_context & ctx, const ggml_tensor * down_w, const ggml_tensor * h,
        const ggml_tensor * ids, const ggml_tensor * w, ggml_tensor * dst, const std::function<void()> & before);
