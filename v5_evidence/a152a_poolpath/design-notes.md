# A-15.2a: pool-path MoE FFN (static slot map) - design notes

## Scope
- This step: pool-path MoE FFN builder + static remap helper + CPU oracle test.
- NOT this step: A-15.2b (dynamic LRU + promotion), qwen4exp graph wiring, A-15.4+.
- v4.1 frozen: `build_moe_ffn` untouched, qwen4exp default graph path unchanged,
  pool path only reachable by explicit call (flag/wiring lands with A-15.2b).

## Interface
### `llm_graph_context::build_moe_ffn_pool(...)` (llama-graph.{h,cpp})
- `build_moe_ffn_pool(cur, gate_inp, up_pool, gate_pool, down_pool,
   n_expert, n_expert_used, type_op, norm_w, w_scale, gating_op, il)`
- pool tensors: gate/up `[n_ff, n_embd, R]`, down `[n_embd, n_ff, R]`; `R = up_pool->ne[2]`.
- gating/weights: identical to `build_moe_ffn` (SOFTMAX, top-k, norm_w, w_scale);
  asserts SILU + SOFTMAX (the qwen4exp caliber).
- FFN: up/gate/down `mul_mat_id` over the pool tensors with remapped ids;
  `swiglu_split(gate, up)` = silu(gate)*up; weighted per-expert sum (same view/add order).

### static remap
- slot map: `slot i = global expert i` for `i < R` (pool = first R experts).
- out of pool (`id >= R`): remapped to slot 0 AND weight zeroed (contribution 0).
- host helpers in llama-graph.h:
  - `q4e_pool_slot_of(id, R)` = `id < R ? id : 0`
  - `q4e_pool_count_out_of_pool(ids, n, R)` = count of ids >= R

### in-graph remap (arithmetic; this ggml has no comparison ops)
- `in_pool = clamp(scale_bias(id, -1, R), 0, 1)` via `ggml_scale_bias` + `ggml_clamp`
  (no scalar-add op and no constant tensors: the graph ctx is no_alloc).
- `pool_ids = cast(id * in_pool, I32)` -> in-pool id, else 0 (satisfies mul_mat_id I32/2D).
- `weights = weights * in_pool` (reshaped to [1,k,n] to match weights layout).
- All remap ops are exact in f32 for in-pool ids -> in-pool results are bit-identical
  to the full path (oracle tolerance 1e-4 is a safety margin, not the expected error).

## Oracle caliber (engine/tests/test-q4e-moe-pool.cpp)
- synthetic: 8 experts, R=5, top-k=3, n_embd=64, n_ff=128, 4 tokens, f32, seed 42,
  no GGUF, pure CPU ggml. Compute goes through `ggml_backend_sched_graph_compute`
  (a `llm_graph_result` graph ctx is no_alloc; plain `ggml_graph_compute_with_ctx`
  segfaults on it). Outputs read before the scheduler is freed (it owns the buffers).
- gate_inp crafted so per-token logits hit the intended top-k exactly
  (orthogonal token groups, dense random expert weights).
- T1 all in pool (top-3 = {0,1,2}): pool vs full path element-wise, hard gate
  `|p-f| <= 1e-4*(1+|f|)`; also vs double-precision host oracle.
- T2 out of pool (top-3 = {0,5,6}): pool output == host oracle with experts >= R zeroed;
  out-of-pool count = 8 (2/token x 4 tokens) via helper.
- T3 per-token mix (2 tokens {0,1,2}, 2 tokens {0,5,6}): per-token oracle, count = 4.
- cross-check: full path (frozen) also compared to host oracle with all experts,
  so a semantic drift in either builder is caught.

## Seam to A-15.2b
- replace the static remap block with a per-layer dynamic slot map
  (slot->global tensor from `q4e_expert_pool`, updated on promote/evict);
- out-of-pool zeroing becomes: miss -> `q4e_expert_h2d` promote -> slot update -> normal contribution;
- `q4e_pool_count_out_of_pool` kept as the promotion-demand counter for stats.

## Files
- engine/src/llama-graph.h: helpers + declaration
- engine/src/llama-graph.cpp: `build_moe_ffn_pool`
- engine/tests/test-q4e-moe-pool.cpp: new (oracle)
- engine/tests/CMakeLists.txt: target (private include src/, link llama)

## Result
- build 46, `test-q4e-moe-pool`: exit 0, all 1841 checks passed (see oracle.log).
- T1 all-in-pool: pool vs full max_abs_err = 0 (bit-identical), within 1e-4 gate.
- T2 out-of-pool count = 8, T3 mixed count = 4, helper count = 3 (all per spec).
- regressions: test-q4e-expert-cache / test-q4e-expert-warm / test-arg-parser exit 0.
