#include "q4e-stats-v6.h"

#include "q4e-global-pool.h"
#include "q4e-graph-map.h"
#include "q4e-h2d-pipeline.h"
#include "q4e-kv-paging.h"
#include "q4e-promotion.h"
#include "q4e-staging.h"

#include <stdexcept>

q4e_stats_v6_t q4e_stats_v6_collect(const q4e_global_pool * pool,
                                    const q4e_staging * staging,
                                    const q4e_h2d_pipeline * pipe,
                                    const q4e_graph_map * map,
                                    const q4e_promotion_op * promo,
                                    const q4e_kv_paging * kv,
                                    int64_t now_us) {
    if (pool == nullptr && staging == nullptr && pipe == nullptr && map == nullptr && promo == nullptr && kv == nullptr) {
        throw std::invalid_argument("q4e_stats_v6_collect: all component pointers are null");
    }
    q4e_stats_v6_t s;
    s.generated_us = now_us;
    if (pool != nullptr) {
        const q4e_global_pool_stats p = pool->stats();
        s.pool_hits = p.hits; s.pool_misses = p.misses; s.pool_evictions = p.evictions; s.pool_floor_hits = p.floor_hits;
        s.pool_rounds = p.rounds; s.pool_m_sum = p.m_sum; s.pool_m_max = p.m_max;
    }
    if (staging != nullptr) {
        const q4e_staging_stats t = staging->stats();
        s.staging_fills = t.fills; s.staging_publishes = t.publishes; s.staging_consumes = t.consumes; s.staging_overflows = t.overflows;
    }
    if (pipe != nullptr) {
        const q4e_h2d_stats p = pipe->stats();
        s.pipe_submits = p.submits; s.pipe_busy = p.busy; s.pipe_fills = p.fills; s.pipe_in_flight = p.in_flight;
        s.pipe_completed = p.completed; s.pipe_failed = p.failed; s.pipe_evictions = p.evictions;
    }
    if (map != nullptr) {
        const q4e_graph_map_stats m = map->stats();
        s.map_maps = m.maps; s.map_queries = m.queries;
        try {
            const q4e_graph_map_agg a = map->aggregate();
            s.map_n_pairs = a.n_pairs; s.map_n_staged_pairs = a.n_staged_pairs; s.map_n_missing_pairs = a.n_missing_pairs;
            s.map_n_unique_staged = a.n_unique_staged; s.map_n_unique_missing = a.n_unique_missing;
        } catch (const std::logic_error &) {
            // router choices not set: map_n_* stay zero
        }
    }
    if (promo != nullptr) {
        const q4e_promotion_stats p = promo->stats();
        s.promo_runs = p.runs; s.promo_reaps = p.reaps; s.promo_skipped = p.skipped; s.promo_submitted = p.submitted;
        s.promo_deferred = p.deferred; s.promo_promoted = p.promoted; s.promo_staging_full = p.staging_full; s.promo_failed = p.failed;
    }
    if (kv != nullptr) {
        const q4e_kv_paging_stats p = kv->stats();
        s.kv_promote_l2_to_l1 = p.promote_l2_to_l1; s.kv_promote_l3_to_l1 = p.promote_l3_to_l1;
        s.kv_demote_l1_to_l2 = p.demote_l1_to_l2; s.kv_demote_l1_to_l3 = p.demote_l1_to_l3; s.kv_demote_l2_to_l3 = p.demote_l2_to_l3;
        s.kv_l1_pages = kv->l1_pages_now(); s.kv_l2_pages = kv->l2_pages_now(); s.kv_l3_pages = kv->l3_pages_now();
    }
    return s;
}

namespace {
void put(std::string & o, bool & first, const char * key, int64_t v) {
    if (!first) o += ",";
    first = false;
    o += "\"";
    o += key;
    o += "\":";
    o += std::to_string(v);
}
} // namespace

std::string q4e_stats_v6_json(const q4e_stats_v6_t & s) {
    std::string o = "{";
    o += "\"pool\":{";
    bool first = true;
    put(o, first, "hits", s.pool_hits); put(o, first, "misses", s.pool_misses);
    put(o, first, "evictions", s.pool_evictions); put(o, first, "floor_hits", s.pool_floor_hits);
    put(o, first, "rounds", s.pool_rounds); put(o, first, "m_sum", s.pool_m_sum); put(o, first, "m_max", s.pool_m_max);
    o += "}";
    o += ",\"staging\":{";
    first = true;
    put(o, first, "fills", s.staging_fills); put(o, first, "publishes", s.staging_publishes);
    put(o, first, "consumes", s.staging_consumes); put(o, first, "overflows", s.staging_overflows);
    o += "}";
    o += ",\"pipe\":{";
    first = true;
    put(o, first, "submits", s.pipe_submits); put(o, first, "busy", s.pipe_busy);
    put(o, first, "fills", s.pipe_fills); put(o, first, "in_flight", s.pipe_in_flight);
    put(o, first, "completed", s.pipe_completed); put(o, first, "failed", s.pipe_failed); put(o, first, "evictions", s.pipe_evictions);
    o += "}";
    o += ",\"map\":{";
    first = true;
    put(o, first, "maps", s.map_maps); put(o, first, "queries", s.map_queries);
    put(o, first, "n_pairs", s.map_n_pairs); put(o, first, "n_staged_pairs", s.map_n_staged_pairs);
    put(o, first, "n_missing_pairs", s.map_n_missing_pairs);
    put(o, first, "n_unique_staged", s.map_n_unique_staged); put(o, first, "n_unique_missing", s.map_n_unique_missing);
    o += "}";
    o += ",\"promo\":{";
    first = true;
    put(o, first, "runs", s.promo_runs); put(o, first, "reaps", s.promo_reaps);
    put(o, first, "skipped", s.promo_skipped); put(o, first, "submitted", s.promo_submitted);
    put(o, first, "deferred", s.promo_deferred); put(o, first, "promoted", s.promo_promoted);
    put(o, first, "staging_full", s.promo_staging_full); put(o, first, "failed", s.promo_failed);
    o += "}";
    o += ",\"kv\":{";
    first = true;
    put(o, first, "promote_l2_to_l1", s.kv_promote_l2_to_l1); put(o, first, "promote_l3_to_l1", s.kv_promote_l3_to_l1);
    put(o, first, "demote_l1_to_l2", s.kv_demote_l1_to_l2); put(o, first, "demote_l1_to_l3", s.kv_demote_l1_to_l3);
    put(o, first, "demote_l2_to_l3", s.kv_demote_l2_to_l3);
    put(o, first, "l1_pages", s.kv_l1_pages); put(o, first, "l2_pages", s.kv_l2_pages); put(o, first, "l3_pages", s.kv_l3_pages);
    o += "}";
    o += ",\"meta\":{";
    first = true;
    put(o, first, "generated_us", s.generated_us);
    o += "}";
    return o + "}";
}
