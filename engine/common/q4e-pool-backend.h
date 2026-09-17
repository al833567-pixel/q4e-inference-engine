// v6 physical pool backend (Phase 1): owns the L1 pool + L2 pinned-RAM buffers and
// implements the CPU fill/H2D backends injected into q4e_h2d_pipeline.
// One expert = one L2 slot: [up][gate][down] concatenated, each n_ff * n_embd floats.
// One L1 slot = one expert part: up_slot(s)/gate_slot(s)/down_slot(s) are contiguous
// [n_ff * n_embd] float blocks (slot-major), so a ggml view with the slot dim last
// (up/gate [n_embd, n_ff, n_slots], down [n_ff, n_embd, n_slots]) maps 1:1 - the
// layout build_moe_ffn_pool consumes. Pure CPU memcpy, no ggml dependency.
// The backend must outlive the pipeline it wires to.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "q4e-global-pool.h"
#include "q4e-h2d-pipeline.h"

struct q4e_pool_backend_config {
    int n_ff = 128;
    int n_embd = 64;
    int n_l1_slots = 8; // L1 slots; 0 = disabled
    int n_l2_slots = 8; // L2 slots the backend allocates; must cover the pipeline slot range
    bool external_l1 = false; // L1 lives outside the backend (model pool); attach_l1() required
    q4e_global_pool_config pool; // must have pool.n_slots == n_l1_slots
};

class q4e_pool_backend {
public:
    explicit q4e_pool_backend(const q4e_pool_backend_config & cfg);
    // throws std::invalid_argument: n_ff/n_embd <= 0, n_l1_slots < 0, n_l2_slots < 0,
    // pool.n_slots != n_l1_slots; the pool config is validated by q4e_global_pool

    int n_l1_slots() const { return cfg_.n_l1_slots; }
    int n_l2_slots() const { return cfg_.n_l2_slots; }
    int64_t slot_bytes() const { return (int64_t) cfg_.n_ff * cfg_.n_embd * (int64_t) sizeof(float); }
    int64_t expert_bytes() const { return 3 * slot_bytes(); }

    // one expert part per L1 slot, contiguous [n_ff * n_embd] float blocks (slot-major)
    float * up_slot(int s);
    float * gate_slot(int s);
    float * down_slot(int s);
    const float * up_slot(int s) const;
    const float * gate_slot(int s) const;
    const float * down_slot(int s) const;
    // throws std::out_of_range: s out of range

    // one full expert: [up][gate][down], 3 * n_ff * n_embd floats
    float * l2_slot(int s);
    const float * l2_slot(int s) const;
    // throws std::out_of_range: s out of range

    // per-layer source: the model's per-layer expert tensors; one expert must be a
    // contiguous n_ff * n_embd float block (expert dim last, like the per-layer tensors)
    void set_layer_source(int layer, const float * up, const float * gate, const float * down);
    // throws std::invalid_argument: layer out of range, null pointer

    // q4e_global_pool pass-through; insert/reset keep the slot tables in sync
    int insert(int layer, int expert); // evicted expert id, or -1
    bool touch(int layer, int expert);
    int slot_of(int layer, int expert) const; // -1 = not resident
    int occupancy_of(int layer) const;
    const char * state_of(int layer, int expert) const;
    void begin_round(int layer);
    void end_round(int layer);
    int round_m_of(int layer) const;
    q4e_global_pool_stats stats() const;
    void reset();
    const q4e_global_pool_config & pool_cfg() const { return pool_.cfg(); }

    // per-layer [n_experts] table: expert id -> L1 slot; out-of-pool -> 0
    const int32_t * slot_table(int layer) const; // nullptr = bad layer

    // inject this backend's fill/H2D callbacks into the pipeline
    void wire(q4e_h2d_pipeline & pipe);

    // attach the external L1 (model pool): slot-major [n_l1_slots * n_ff * n_embd] float
    // buffers, the layout build_moe_ffn_pool consumes; the caller keeps them alive
    void attach_l1(float * up, float * gate, float * down);
    // throws std::logic_error: external_l1 not set or already attached; std::invalid_argument: null

    // expert -> L1 slot, graph convention: out-of-pool -> n_l1_slots
    int graph_slot(int layer, int expert) const;

    // 3a runtime round: reserve L1 for the round's top-k, promote the missing through the
    // pipeline, drain, verify all top-k resident, refresh the tables
    void sync_round(q4e_h2d_pipeline & pipe, const std::vector<std::vector<int>> & topk);
    // throws std::invalid_argument: topk covers more layers than configured;
    // std::runtime_error: pool full at floor, h2d failure, or a top-k expert lost

private:
    int64_t fill(int layer, int expert, void * dst, int64_t bytes); // q4e_h2d_fill_fn body
    bool h2d(int64_t job_id, void * src, void * dst, int64_t bytes); // q4e_h2d_copy_fn body
    void refresh_tables();
    float * l1_base(int part); // 0 = up, 1 = gate, 2 = down; external buffer if attached
    const float * l1_base(int part) const;

    struct source {
        const float * up = nullptr;
        const float * gate = nullptr;
        const float * down = nullptr;
        bool set = false;
    };

    q4e_pool_backend_config cfg_;
    q4e_global_pool pool_;
    std::vector<float> up_;
    std::vector<float> gate_;
    std::vector<float> down_;
    std::vector<float> l2_;
    std::vector<source> sources_;
    std::vector<std::vector<int32_t>> tables_;
    std::map<int, std::pair<int, int>> l2_owner_; // L2 slot -> (layer, expert)
    float * l1_ext_[3] = { nullptr, nullptr, nullptr }; // external L1 (attach_l1)
};
