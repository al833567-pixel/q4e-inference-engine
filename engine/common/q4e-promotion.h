// v6 promotion op (A-16.4b): drain the graph map's promotion queue into the H2D
// pipeline, then back-fill the completed jobs into the staging A/B buffers.
// One op = one layer. Bookkeeping only: references the map/pipeline/staging,
// owns none of them. A job whose place overflows the staging buffer is kept
// in L2 (staging_full event), never dropped.
//
// Exception contract:
//   std::invalid_argument: layer < 0, expert_bytes <= 0, expert_bytes > staging chunk
//   std::logic_error: staging protocol violation, map choices not set (via map)
//   std::runtime_error: fill backend failure (via pipeline submit)
#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "q4e-graph-map.h"
#include "q4e-h2d-pipeline.h"
#include "q4e-staging.h"

struct q4e_promotion_run {
    int n_submitted = 0;
    int n_busy_deferred = 0; // submit rejected at the in-flight cap, retried next run
    int n_failed = 0;        // H2D failures observed since the previous run (in reap)
};

struct q4e_promotion_stats {
    uint64_t runs = 0;
    uint64_t reaps = 0;
    uint64_t skipped = 0;      // experts already resident in staging
    uint64_t submitted = 0;
    uint64_t deferred = 0;
    uint64_t promoted = 0;
    uint64_t staging_full = 0; // place overflowed, expert kept in L2
    uint64_t failed = 0;       // H2D jobs failed
};

class q4e_promotion_op {
public:
    q4e_promotion_op(q4e_graph_map & map, q4e_h2d_pipeline & pipe, q4e_staging & st, int layer, int64_t expert_bytes);
    // throws std::invalid_argument: layer < 0, expert_bytes <= 0, expert_bytes > st.chunk_bytes()

    q4e_promotion_run run(); // submit the promotion queue, in first-seen order
    // throws std::logic_error: router choices not set (via map.promotion_queue)

    std::vector<int> reap(); // place completed jobs into staging; returns promoted ids in first-seen order
    // throws std::logic_error: h2d backend not set (via pipeline poll), map choices not set, staging violation

    void deferred_retry(std::vector<int> & out) const; // last run's busy deferred, minus now-resident
    bool resident(int expert) const;                   // placed in a filling/published staging buffer

    q4e_promotion_stats stats() const;
    void reset(); // clears op state only; map/pipe/staging are untouched

private:
    int next_fill_buf() const; // -1 when neither buffer is idle and none is filling

    q4e_graph_map & map_;
    q4e_h2d_pipeline & pipe_;
    q4e_staging & st_;
    int layer_ = 0;
    int64_t expert_bytes_ = 0;

    std::map<int, std::pair<int, uint32_t>> resident_; // expert -> {buf, slot}
    std::unordered_map<int64_t, int> job_expert_;      // live job id -> expert
    std::vector<int> deferred_;                        // last run's busy deferred
    uint64_t unreported_failed_ = 0;

    q4e_promotion_stats stats_;
};
