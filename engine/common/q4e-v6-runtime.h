// q4e v6 runtime promotion (C4-0): drives the global expert pool at decode time.
// Per compute round the host reads each MoE layer's router selections back from the
// graph, feeds them to the LRU policy (global LRU + per-layer floor), and stages the
// misses. commit() fills the staged slots with one tensor_set per part (H2D on device
// buffers, memcpy on host buffers) and rewrites every layer's slot table. commit()
// must run before the next graph build. v1 is synchronous; the async pipeline/staging
// path stays in q4e-h2d-pipeline.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "q4e-global-pool.h"

struct ggml_tensor;

struct q4e_v6_runtime_cfg {
    int n_layers = 0;  // MoE layer count wired to the pool
    int n_expert = 512;
    int n_used   = 10; // router top-k per token
    int n_slots  = 0;  // global pool slots (0 = runtime off)
    int floor    = 10; // per-layer minimum residents
};

// one expert slice is a contiguous ne0*ne1 block (expert dim last); the pool parts
// live on the compute device buffer, the sources stay on the host (mmap weights)
struct q4e_v6_layer_io {
    ggml_tensor * up_pool   = nullptr; // [n_embd, n_ff, n_slots]
    ggml_tensor * gate_pool = nullptr;
    ggml_tensor * down_pool = nullptr; // [n_ff, n_embd, n_slots]
    ggml_tensor * table     = nullptr; // [n_expert] f32 slot ids, n_slots = miss sentinel
    const void * up_src     = nullptr; // host source base, expert e at e * part_stride
    const void * gate_src   = nullptr;
    const void * down_src   = nullptr;
    int64_t up_stride       = 0;       // bytes between experts in each source
    int64_t gate_stride     = 0;
    int64_t down_stride     = 0;
};

class q4e_v6_runtime {
public:
    explicit q4e_v6_runtime(const q4e_v6_runtime_cfg & cfg);
    // throws std::invalid_argument: n_layers/n_expert/n_used <= 0, n_slots < 0,
    // floor < 0 or n_slots != 0 and n_slots < n_layers * floor

    void bind_layer(int il, const q4e_v6_layer_io & io);
    // throws std::invalid_argument: il out of range or bound twice

    // feed one layer's router selections (count elements, i32, expert ids);
    // touches residents and stages misses into the policy
    void observe(int il, const int32_t * sel, int64_t count);
    // throws std::invalid_argument: unbound layer

    // fill staged misses into their slots and rewrite every bound layer's table
    void commit();

    bool enabled() const { return cfg_.n_slots > 0; }
    q4e_global_pool_stats stats() const;
    int64_t insert_fails() const { return insert_fails_; }

private:
    void fill_slot(int slot, int il, int expert);
    void write_table(int il);
    void copy_slice(ggml_tensor * dst, const void * src, int64_t dst_offset, int64_t bytes);

    q4e_v6_runtime_cfg cfg_;
    q4e_global_pool pool_;

    struct layer_state {
        q4e_v6_layer_io io;
        bool bound = false;
    };
    std::vector<layer_state> layers_;

    // staged work of the current round: (layer, expert) pairs to make resident
    std::vector<std::pair<int, int>> pending_;

    int64_t insert_fails_ = 0;
};
