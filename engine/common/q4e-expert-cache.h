#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <set>
#include <unordered_map>
#include <vector>

// tiered MoE expert cache (q4e addition)
// spec: v2 contract (v3 cache.py, removed v4); frozen in test-q4e-expert-cache.cpp
// tiers: vram (L1 hot) / ram (L2) / ssd (L3, untracked, infinite)
// expert ids: 0..n_experts-1 routed; Q4E_EXPERT_SHARED_ID = per-layer shared expert

#define Q4E_EXPERT_SHARED_ID (-1)

struct q4e_expert_cache_config {
    int  n_layers = 48;
    int  n_experts_per_layer = 512;
    int  vram_capacity_per_layer = 20;
    int  ram_capacity_per_layer  = 142;
    bool has_shared = true;
};

// spec: touch return (v2 contract); frozen in test-q4e-expert-cache.cpp
struct q4e_expert_hits {
    int64_t vram = 0;
    int64_t ram  = 0;
    int64_t ssd  = 0;
};

struct q4e_expert_prefetch_stats {
    int64_t issued = 0;
    int64_t skipped_in_vram = 0;
    int64_t already_staged = 0;
    int64_t evicted_from_ram = 0;
    int64_t hits = 0;
};

struct q4e_expert_occupancy {
    double avg = 0.0;
    int    min = 0;
    int    max = 0;
};

struct q4e_expert_placement_event {
    int layer = 0;
    int expert = 0;
    int tier = 0; // 0=vram 1=ram 2=ssd
};

struct q4e_expert_cache_stats {
    int  layers = 0;
    int64_t touches = 0;
    q4e_expert_hits hits;
    double hit_rate_vram = 0.0;
    double hit_rate_ram  = 0.0;
    double hit_rate_ssd  = 0.0;
    q4e_expert_prefetch_stats prefetch;
    q4e_expert_occupancy occ_vram;
    q4e_expert_occupancy occ_ram;
    int64_t placement_events = 0;
    int64_t layers_touched = 0;        // A-9: distinct layers touched (M proxy denominator)
    int64_t hook_skips_device_ids = 0; // A-9: cb_eval skips of non-host-resident ids tensors
};

// placement hook; default is no-op (fake). real backends do SSD<->RAM<->VRAM moves
struct q4e_expert_cache_backend {
    virtual ~q4e_expert_cache_backend() = default;
    virtual void place(int layer, int expert, const char * tier) {}
    virtual void evict(int layer, int expert, const char * tier) {}
};

// deterministic state machine: same input sequence => same state (replayable)
struct q4e_expert_cache {
    explicit q4e_expert_cache(const q4e_expert_cache_config & cfg = q4e_expert_cache_config());
    q4e_expert_cache(const q4e_expert_cache_config & cfg, q4e_expert_cache_backend * backend);

    // one token at layer used these routed experts; throws std::invalid_argument on bad id/layer
    q4e_expert_hits touch(int layer, const int * expert_ids, int64_t n_ids);
    // speculative prefetch: previous token's selection; ram only, never vram
    void prefetch(int layer, const int * hint_ids, int64_t n_hints);

    q4e_expert_cache_stats stats() const;
    const char * tier_of(int layer, int expert) const; // "vram" | "ram" | "ssd"
    void reset();

    // A-9: cb_eval hot path never syncs D2H; this counts skipped non-host-resident ids tensors
    void note_hook_skip() { ++hook_skips_device_; }

    const q4e_expert_cache_config & cfg() const { return cfg_; }
    const std::vector<q4e_expert_placement_event> & placement_log() const { return placement_log_; }

private:
    void vram_insert(int layer, int expert);
    void ram_insert(int layer, int expert);
    void check_layer(int layer) const;

    q4e_expert_cache_config cfg_;
    q4e_expert_cache_backend * backend_;
    struct layer_state {
        std::list<int> vram_order; // front = LRU, back = MRU
        std::unordered_map<int, std::list<int>::iterator> vram_pos;
        std::list<int> ram_order;
        std::unordered_map<int, std::list<int>::iterator> ram_pos;
        std::set<int> prefetched;
    };
    std::vector<layer_state> layers_;
    int64_t touches_ = 0;
    q4e_expert_hits hits_;
    q4e_expert_prefetch_stats pref_;
    std::vector<q4e_expert_placement_event> placement_log_;
    std::vector<char> layer_touched_;
    int64_t hook_skips_device_ = 0;
};

struct ggml_tensor;

// layer id of a MoE expert tensor name ("blk.N.ffn_*_exps..."); -1 if not a MoE layer
int q4e_expert_layer_from_name(const char * name, int n_layers);

// llama_context_params.cb_eval hook: on MoE expert matmul (MUL_MAT_ID, gate/up path),
// reads the selected expert ids [n_expert_used, n_tokens] and touches the cache per token
// A-9: only host-resident ids are touched (no sync D2H); non-host ids are skipped and counted
// user_data must be a q4e_expert_cache * (or nullptr to no-op)
bool q4e_expert_cache_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
