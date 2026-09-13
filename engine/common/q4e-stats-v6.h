// v6 /q4e/stats aggregation (A-16.6a): flat snapshot of the six component stats
// (pool / staging / pipe / map / promo / kv) plus a manual JSON assembly.
// Pure C++, no ggml/CUDA dependency, no external JSON library.
#pragma once

#include <cstdint>
#include <string>

class q4e_global_pool;
class q4e_staging;
class q4e_h2d_pipeline;
class q4e_graph_map;
class q4e_promotion_op;
class q4e_kv_paging;

// flat snapshot; every field is present and zero-initialized (null configuration = zero fields)
// pool group = scalar totals; the per-layer vectors of q4e_global_pool_stats are out of scope
struct q4e_stats_v6_t {
    int64_t pool_hits = 0, pool_misses = 0, pool_evictions = 0, pool_floor_hits = 0;
    int64_t pool_rounds = 0, pool_m_sum = 0, pool_m_max = 0;
    int64_t staging_fills = 0, staging_publishes = 0, staging_consumes = 0, staging_overflows = 0;
    int64_t pipe_submits = 0, pipe_busy = 0, pipe_fills = 0, pipe_in_flight = 0;
    int64_t pipe_completed = 0, pipe_failed = 0, pipe_evictions = 0;
    int64_t map_maps = 0, map_queries = 0;
    int64_t map_n_pairs = 0, map_n_staged_pairs = 0, map_n_missing_pairs = 0;
    int64_t map_n_unique_staged = 0, map_n_unique_missing = 0;
    int64_t promo_runs = 0, promo_reaps = 0, promo_skipped = 0, promo_submitted = 0;
    int64_t promo_deferred = 0, promo_promoted = 0, promo_staging_full = 0, promo_failed = 0;
    int64_t kv_promote_l2_to_l1 = 0, kv_promote_l3_to_l1 = 0;
    int64_t kv_demote_l1_to_l2 = 0, kv_demote_l1_to_l3 = 0, kv_demote_l2_to_l3 = 0;
    int64_t kv_l1_pages = 0, kv_l2_pages = 0, kv_l3_pages = 0;
    int64_t generated_us = 0; // injected timestamp (us); never read from the clock
};

// One snapshot across all provided components.
// Any null pointer leaves that component's fields at zero (partial configurations work).
// throws std::invalid_argument: all six pointers null
// q4e_graph_map::aggregate() throws std::logic_error when router choices are unset;
// in that case the map aggregate fields (map_n_*) stay zero
q4e_stats_v6_t q4e_stats_v6_collect(const q4e_global_pool * pool,
                                    const q4e_staging * staging,
                                    const q4e_h2d_pipeline * pipe,
                                    const q4e_graph_map * map,
                                    const q4e_promotion_op * promo,
                                    const q4e_kv_paging * kv,
                                    int64_t now_us);

// Manual JSON: fixed field order, snake_case keys, grouped by component
// (pool/staging/pipe/map/promo/kv/meta); all numeric fields present.
std::string q4e_stats_v6_json(const q4e_stats_v6_t & s);
