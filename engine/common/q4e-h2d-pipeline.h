// v6 H2D promotion pipeline (A-16.3): SSD -> L2 pinned fill + async L2 -> L1 job queue.
// Bookkeeping only: owns no buffer memory; backends are injected (production = SSD mmap
// read / cudaMemcpyAsync, tests = fakes). dst/src handles are L2 slot indices as void *.
// L2 is a global LRU pool; an expert with a live job is protected from eviction.
//
// Exception contract:
//   std::invalid_argument: bad config, layer/expert < 0
//   std::logic_error: backend not injected before it is needed
//   std::runtime_error: fill backend failure, L2 full while every resident expert is in flight
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// distinct from the v5 q4e_h2d_config in q4e-expert-warm.h (name collision in shared TUs)
struct q4e_h2d_pipeline_config {
    uint32_t l2_ram_mb    = 4000;    // L2 pinned expert pool MB (--q4e-moe-ram), >= 4
    uint64_t expert_bytes = 3216896; // one expert slot bytes, > 0
    uint32_t max_inflight = 4;       // in-flight H2D cap, >= 1
};

struct q4e_h2d_stats {
    uint64_t submits = 0;
    uint64_t busy = 0;      // submits rejected at the in-flight cap
    uint64_t fills = 0;     // SSD -> L2 fills
    uint64_t in_flight = 0; // live jobs now
    uint64_t completed = 0; // H2D jobs done
    uint64_t failed = 0;    // H2D jobs failed
    uint64_t evictions = 0; // L2 LRU evictions
};

// dst = L2 slot handle; returns bytes read, <= 0 = failure (pipeline throws runtime_error)
using q4e_h2d_fill_fn = std::function<int64_t(int layer, int expert, void * dst, int64_t bytes)>;
// src = L2 slot handle, dst resolved by the backend (A-16.2 staging); returns success
using q4e_h2d_copy_fn = std::function<bool(int64_t job_id, void * src, void * dst, int64_t bytes)>;

class q4e_h2d_pipeline {
public:
    explicit q4e_h2d_pipeline(const q4e_h2d_pipeline_config & cfg = q4e_h2d_pipeline_config());
    // throws std::invalid_argument: l2_ram_mb < 4, expert_bytes <= 0, max_inflight < 1, zero L2 slots

    void set_fill_backend(q4e_h2d_fill_fn fn);
    void set_h2d_backend(q4e_h2d_copy_fn fn);

    int64_t submit(int layer, int expert); // job id, or -1 when the in-flight cap is reached (busy)
    std::vector<int64_t> poll();           // done job ids; failures counted, not returned

    const char * l2_state(int layer, int expert) const; // "l2" | "l3"
    uint32_t l2_occupancy() const;                     // resident experts
    int64_t l2_slot_of(int layer, int expert) const;   // -1 = not resident
    uint32_t l2_slots() const;                         // total slot count

    q4e_h2d_stats stats() const;
    void reset();

private:
    struct l2_entry {
        int layer = 0;
        int expert = 0;
        uint32_t slot = 0;
    };
    struct job {
        int layer = 0;
        int expert = 0;
        uint32_t slot = 0;
        bool alive = false;
    };

    static int64_t make_key(int layer, int expert);
    void check_id(int layer, int expert) const;
    bool in_flight_of(int64_t key) const;

    q4e_h2d_pipeline_config cfg_;
    uint32_t n_slots_ = 0;
    std::vector<uint32_t> free_slots_;
    std::list<l2_entry> lru_; // front = most recently used
    std::unordered_map<int64_t, std::list<l2_entry>::iterator> lru_pos_;
    std::vector<job> jobs_;
    std::vector<size_t> free_jobs_;
    q4e_h2d_fill_fn fill_;
    q4e_h2d_copy_fn h2d_;
    q4e_h2d_stats stats_;
};
