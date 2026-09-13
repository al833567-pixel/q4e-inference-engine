// v6 global dynamic expert pool manager (A-16.1a)
// One L1 VRAM slot pool shared by all MoE layers; slot identity = (layer, expert).
// Global LRU eviction, except a layer never drops below its floor (floor=0 = pure global LRU).
// Pool full and every layer at the floor: insert throws std::runtime_error (caller runs the promotion path).
// Pure CPU bookkeeping, no ggml/CUDA dependency. The v5 q4e_expert_pool stays frozen and untouched.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

struct q4e_global_pool_config {
    int n_experts_per_layer = 512;
    int n_layers            = 48;
    int n_slots             = 5000; // global L1 slot count; 0 = disabled
    int floor               = 10;   // min residents per layer; clamped to n_experts_per_layer
};

struct q4e_global_pool_stats {
    int64_t hits       = 0;
    int64_t misses     = 0;
    int64_t evictions  = 0;
    int64_t floor_hits = 0; // eviction candidates skipped because their layer sat at the floor
    int64_t rounds     = 0;
    int64_t m_sum      = 0; // sum of per-round distinct expert count (M)
    int64_t m_max      = 0;
    std::vector<int64_t> hits_per_layer;
    std::vector<int64_t> misses_per_layer;
    std::vector<int64_t> evictions_per_layer;
};

class q4e_global_pool {
public:
    explicit q4e_global_pool(const q4e_global_pool_config & cfg = q4e_global_pool_config());
    // ctor throws std::invalid_argument: n_layers/n_experts_per_layer <= 0, n_slots < 0, floor < 0,
    // or n_slots (!= 0) < n_layers * floor; floor above n_experts_per_layer is clamped down

    // reserve an L1 slot for (layer, expert); returns the evicted expert id, or -1
    // throws std::runtime_error when the pool is full and every layer sits at the floor
    int insert(int layer, int expert);

    // record an access; true = resident (LRU refreshed), false = not resident
    bool touch(int layer, int expert);

    const char * state_of(int layer, int expert) const; // "l1" resident | "l3" not resident
    int occupancy_of(int layer) const;                   // resident count of a layer
    int slot_of(int layer, int expert) const;            // global slot index; -1 = not resident

    // round M = distinct experts of the layer's active round (touched or inserted, dedup);
    // round_m_of returns the live count while a round is open, the last committed value when idle
    void begin_round(int layer);
    void end_round(int layer);
    int round_m_of(int layer) const;

    q4e_global_pool_stats stats() const;
    void reset();

    const q4e_global_pool_config & cfg() const { return cfg_; }

private:
    void check_layer(int layer) const;
    void check_expert(int expert) const;
    void note_round(int layer, int expert);
    void refresh_lru(uint64_t key);
    std::list<uint64_t>::iterator find_victim(); // lru_.end() when every layer sits at the floor

    static uint64_t make_key(int layer, int expert) {
        return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
    }

    struct layer_state {
        std::set<int> round_ids;
        bool round_active = false;
        int last_m = 0;
    };

    q4e_global_pool_config cfg_;
    std::vector<layer_state> layers_;
    std::vector<int> layer_count_;
    std::vector<std::pair<int, int>> slot_owner_; // slot -> (layer, expert); {-1,-1} = free
    std::unordered_map<uint64_t, int> resident_;  // (layer, expert) -> slot
    std::list<uint64_t> lru_;                     // front = LRU, back = MRU
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_pos_;
    int64_t hits_ = 0;
    int64_t misses_ = 0;
    int64_t evictions_ = 0;
    int64_t floor_hits_ = 0;
    int64_t rounds_ = 0;
    int64_t m_sum_ = 0;
    int64_t m_max_ = 0;
    std::vector<int64_t> hits_layer_;
    std::vector<int64_t> misses_layer_;
    std::vector<int64_t> evictions_layer_;
};
