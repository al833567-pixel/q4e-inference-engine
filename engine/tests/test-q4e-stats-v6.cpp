// Unit tests for the v6 /q4e/stats aggregation: q4e_stats_v6_collect plus the manual
// JSON assembly (common/q4e-stats-v6.{h,cpp}, A-16.6a).
#include "q4e-stats-v6.h"

#include "q4e-global-pool.h"
#include "q4e-graph-map.h"
#include "q4e-h2d-pipeline.h"
#include "q4e-kv-paging.h"
#include "q4e-promotion.h"
#include "q4e-staging.h"

#include <cstdio>
#include <stdexcept>
#include <string>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond) \
    do { \
        ++g_checks; \
        if (!(cond)) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

#define CHECK_EQ_I64(a, b) \
    do { \
        ++g_checks; \
        const int64_t va = (a); \
        const int64_t vb = (b); \
        if (va != vb) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s == %lld, expected %lld\n", __FILE__, __LINE__, #a, (long long) va, (long long) vb); \
        } \
    } while (0)

#define CHECK_CONTAINS(hay, needle) \
    do { \
        ++g_checks; \
        if (std::string(hay).find(needle) == std::string::npos) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: missing %s\n", __FILE__, __LINE__, #needle); \
        } \
    } while (0)

#define CHECK_THROWS_IA(expr) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            (void) (expr); \
        } catch (const std::invalid_argument &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

// fixture: pool 2 layers x 8 experts, 6 slots, floor 1
static q4e_global_pool_config pool_cfg() {
    q4e_global_pool_config c;
    c.n_experts_per_layer = 8;
    c.n_layers            = 2;
    c.n_slots             = 6;
    c.floor               = 1;
    return c;
}

// fixture: 2 MB per buffer, 1 MB chunks -> 2 chunks per buffer
static q4e_staging_config staging_cfg() {
    q4e_staging_config c;
    c.total_mb    = 4;
    c.chunk_bytes = 1048576;
    return c;
}

// fixture: 4 x 1 MB L2 slots, in-flight cap 1
static q4e_h2d_config pipe_cfg() {
    q4e_h2d_config c;
    c.l2_ram_mb    = 4;
    c.expert_bytes = 1048576;
    c.max_inflight = 1;
    return c;
}

// drive pool / kv / map a few steps so every stat group is non-trivial
static void setup(q4e_global_pool & pool, q4e_graph_map & map, q4e_kv_paging & kv) {
    // pool: 3 hits, 1 miss, 1 eviction, one round with m = 2
    pool.insert(0, 0);
    pool.insert(0, 1);
    pool.touch(0, 1); // hit
    pool.touch(0, 7); // miss
    pool.begin_round(0);
    pool.touch(0, 0);
    pool.touch(0, 1);
    pool.end_round(0);
    pool.insert(1, 0);
    pool.insert(1, 1);
    pool.insert(1, 2);
    pool.insert(1, 3); // 6 residents = n_slots, pool now full
    pool.insert(0, 2); // evicts the LRU (0,0)

    // kv (3 pages, l1 cap 2, l2 cap 1): p0,p1 -> L1; p2 -> L1 with p0 -> L2; p1 -> L2 with p0 -> L3
    kv.ensure_resident(0, 64);
    kv.ensure_resident(64, 96);
    kv.demote_range(32, 64);

    // map: one map, two queries, promotion queue [1, 2, 3]
    const int choices[4] = { 0, 1, 2, 3 };
    const int staged[1]  = { 0 };
    map.set_router_choices(choices, 4);
    map.set_staged(staged, 1);
    (void) map.token_view(0);
    (void) map.token_view(1);
}

// drive promo (in-flight cap 1: 1 submit per run, the rest deferred) then staging
static void drive(q4e_promotion_op & promo, q4e_staging & st) {
    promo.run(); // submit expert 1, defer experts 2 + 3
    promo.reap(); // expert 1 -> staging buf 0
    promo.run(); // expert 1 skipped (resident), submit expert 2, defer expert 3
    promo.reap(); // expert 2 -> staging buf 0
    st.publish(0);
    st.consume_begin(0);
    st.release(0);
    st.begin_fill(1);
    st.place(1, 1048576);
    st.place(1, 1048576);
    st.place(1, 1048576); // buffer full -> overflow event
}

// full configuration: every field of the snapshot must match the live component stats
static void test_collect_full() {
    q4e_global_pool  pool(pool_cfg());
    q4e_staging      st(staging_cfg());
    q4e_h2d_pipeline pipe(pipe_cfg());
    pipe.set_fill_backend([](int, int, void *, int64_t n) { return n; });
    pipe.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });
    q4e_graph_map    map(8, 2, 2);
    q4e_promotion_op promo(map, pipe, st, 0, 1048576);
    q4e_kv_paging    kv(96, 32, 2, 1, 1000);

    setup(pool, map, kv);
    drive(promo, st);

    // fixture sanity: the component stats themselves carry the expected counts
    const q4e_global_pool_stats ps  = pool.stats();
    const q4e_staging_stats     ts = st.stats();
    const q4e_h2d_stats         hs = pipe.stats();
    const q4e_graph_map_stats   ms = map.stats();
    const q4e_graph_map_agg     ma = map.aggregate();
    const q4e_promotion_stats   os = promo.stats();
    const q4e_kv_paging_stats   ks = kv.stats();
    CHECK_EQ_I64(ps.hits, 3); CHECK_EQ_I64(ps.misses, 1); CHECK_EQ_I64(ps.evictions, 1);
    CHECK_EQ_I64(ps.floor_hits, 0); CHECK_EQ_I64(ps.rounds, 1); CHECK_EQ_I64(ps.m_sum, 2); CHECK_EQ_I64(ps.m_max, 2);
    CHECK_EQ_I64(ts.fills, 2); CHECK_EQ_I64(ts.publishes, 1); CHECK_EQ_I64(ts.consumes, 1); CHECK_EQ_I64(ts.overflows, 1);
    CHECK_EQ_I64(hs.submits, 2); CHECK_EQ_I64(hs.busy, 3); CHECK_EQ_I64(hs.fills, 2);
    CHECK_EQ_I64(hs.in_flight, 0); CHECK_EQ_I64(hs.completed, 2); CHECK_EQ_I64(hs.failed, 0); CHECK_EQ_I64(hs.evictions, 0);
    CHECK_EQ_I64(ms.maps, 1); CHECK_EQ_I64(ms.queries, 2);
    CHECK_EQ_I64(ma.n_pairs, 4); CHECK_EQ_I64(ma.n_staged_pairs, 1); CHECK_EQ_I64(ma.n_missing_pairs, 3);
    CHECK_EQ_I64(ma.n_unique_staged, 1); CHECK_EQ_I64(ma.n_unique_missing, 3);
    CHECK_EQ_I64(os.runs, 2); CHECK_EQ_I64(os.reaps, 2); CHECK_EQ_I64(os.skipped, 1); CHECK_EQ_I64(os.submitted, 2);
    CHECK_EQ_I64(os.deferred, 3); CHECK_EQ_I64(os.promoted, 2); CHECK_EQ_I64(os.staging_full, 0); CHECK_EQ_I64(os.failed, 0);
    CHECK_EQ_I64(ks.promote_l2_to_l1, 0); CHECK_EQ_I64(ks.promote_l3_to_l1, 3);
    CHECK_EQ_I64(ks.demote_l1_to_l2, 2); CHECK_EQ_I64(ks.demote_l1_to_l3, 0); CHECK_EQ_I64(ks.demote_l2_to_l3, 1);
    CHECK_EQ_I64((int64_t) kv.l1_pages_now(), 1); CHECK_EQ_I64((int64_t) kv.l2_pages_now(), 1); CHECK_EQ_I64((int64_t) kv.l3_pages_now(), 1);

    const q4e_stats_v6_t s = q4e_stats_v6_collect(&pool, &st, &pipe, &map, &promo, &kv, 1234567);
    CHECK_EQ_I64(s.generated_us, 1234567);
    CHECK_EQ_I64(s.pool_hits, ps.hits); CHECK_EQ_I64(s.pool_misses, ps.misses); CHECK_EQ_I64(s.pool_evictions, ps.evictions);
    CHECK_EQ_I64(s.pool_floor_hits, ps.floor_hits); CHECK_EQ_I64(s.pool_rounds, ps.rounds);
    CHECK_EQ_I64(s.pool_m_sum, ps.m_sum); CHECK_EQ_I64(s.pool_m_max, ps.m_max);
    CHECK_EQ_I64(s.staging_fills, ts.fills); CHECK_EQ_I64(s.staging_publishes, ts.publishes);
    CHECK_EQ_I64(s.staging_consumes, ts.consumes); CHECK_EQ_I64(s.staging_overflows, ts.overflows);
    CHECK_EQ_I64(s.pipe_submits, hs.submits); CHECK_EQ_I64(s.pipe_busy, hs.busy); CHECK_EQ_I64(s.pipe_fills, hs.fills);
    CHECK_EQ_I64(s.pipe_in_flight, hs.in_flight); CHECK_EQ_I64(s.pipe_completed, hs.completed);
    CHECK_EQ_I64(s.pipe_failed, hs.failed); CHECK_EQ_I64(s.pipe_evictions, hs.evictions);
    CHECK_EQ_I64(s.map_maps, ms.maps); CHECK_EQ_I64(s.map_queries, ms.queries);
    CHECK_EQ_I64(s.map_n_pairs, ma.n_pairs); CHECK_EQ_I64(s.map_n_staged_pairs, ma.n_staged_pairs);
    CHECK_EQ_I64(s.map_n_missing_pairs, ma.n_missing_pairs);
    CHECK_EQ_I64(s.map_n_unique_staged, ma.n_unique_staged); CHECK_EQ_I64(s.map_n_unique_missing, ma.n_unique_missing);
    CHECK_EQ_I64(s.promo_runs, os.runs); CHECK_EQ_I64(s.promo_reaps, os.reaps);
    CHECK_EQ_I64(s.promo_skipped, os.skipped); CHECK_EQ_I64(s.promo_submitted, os.submitted);
    CHECK_EQ_I64(s.promo_deferred, os.deferred); CHECK_EQ_I64(s.promo_promoted, os.promoted);
    CHECK_EQ_I64(s.promo_staging_full, os.staging_full); CHECK_EQ_I64(s.promo_failed, os.failed);
    CHECK_EQ_I64(s.kv_promote_l2_to_l1, ks.promote_l2_to_l1); CHECK_EQ_I64(s.kv_promote_l3_to_l1, ks.promote_l3_to_l1);
    CHECK_EQ_I64(s.kv_demote_l1_to_l2, ks.demote_l1_to_l2); CHECK_EQ_I64(s.kv_demote_l1_to_l3, ks.demote_l1_to_l3);
    CHECK_EQ_I64(s.kv_demote_l2_to_l3, ks.demote_l2_to_l3);
    CHECK_EQ_I64(s.kv_l1_pages, (int64_t) kv.l1_pages_now());
    CHECK_EQ_I64(s.kv_l2_pages, (int64_t) kv.l2_pages_now());
    CHECK_EQ_I64(s.kv_l3_pages, (int64_t) kv.l3_pages_now());
}

// null configurations: only pool -> pool group set, everything else zero; all null throws
static void test_null_configs() {
    q4e_global_pool pool(pool_cfg());
    pool.insert(0, 3);
    pool.touch(0, 3); // one hit

    const q4e_stats_v6_t s = q4e_stats_v6_collect(&pool, nullptr, nullptr, nullptr, nullptr, nullptr, 42);
    const q4e_global_pool_stats ps = pool.stats();
    CHECK_EQ_I64(s.generated_us, 42);
    CHECK_EQ_I64(s.pool_hits, ps.hits);
    CHECK_EQ_I64(s.pool_rounds, ps.rounds);
    CHECK_EQ_I64(s.staging_fills, 0); CHECK_EQ_I64(s.staging_overflows, 0);
    CHECK_EQ_I64(s.pipe_submits, 0); CHECK_EQ_I64(s.pipe_busy, 0); CHECK_EQ_I64(s.pipe_evictions, 0);
    CHECK_EQ_I64(s.map_maps, 0); CHECK_EQ_I64(s.map_n_unique_missing, 0);
    CHECK_EQ_I64(s.promo_runs, 0); CHECK_EQ_I64(s.promo_promoted, 0);
    CHECK_EQ_I64(s.kv_promote_l3_to_l1, 0); CHECK_EQ_I64(s.kv_l1_pages, 0); CHECK_EQ_I64(s.kv_l3_pages, 0);

    CHECK_THROWS_IA(q4e_stats_v6_collect(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0));
}

// map without router choices: aggregate() throws logic_error, map_n_* must stay zero
static void test_map_no_choices() {
    q4e_graph_map map(8, 2, 2);
    const q4e_stats_v6_t s = q4e_stats_v6_collect(nullptr, nullptr, nullptr, &map, nullptr, nullptr, 7);
    CHECK_EQ_I64(s.generated_us, 7);
    CHECK_EQ_I64(s.map_maps, 0);
    CHECK_EQ_I64(s.map_queries, 0);
    CHECK_EQ_I64(s.map_n_pairs, 0);
    CHECK_EQ_I64(s.map_n_staged_pairs, 0);
    CHECK_EQ_I64(s.map_n_missing_pairs, 0);
    CHECK_EQ_I64(s.map_n_unique_staged, 0);
    CHECK_EQ_I64(s.map_n_unique_missing, 0);
}

// JSON: all seven group keys, correct values, injected timestamp passthrough
static void test_json_full() {
    q4e_global_pool  pool(pool_cfg());
    q4e_staging      st(staging_cfg());
    q4e_h2d_pipeline pipe(pipe_cfg());
    pipe.set_fill_backend([](int, int, void *, int64_t n) { return n; });
    pipe.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });
    q4e_graph_map    map(8, 2, 2);
    q4e_promotion_op promo(map, pipe, st, 0, 1048576);
    q4e_kv_paging    kv(96, 32, 2, 1, 1000);
    setup(pool, map, kv);
    drive(promo, st);

    const q4e_stats_v6_t s = q4e_stats_v6_collect(&pool, &st, &pipe, &map, &promo, &kv, 1700000000000ll);
    const std::string j = q4e_stats_v6_json(s);
    CHECK(j.rfind("{\"pool\":", 0) == 0);
    CHECK(j.rfind("}", j.size() - 1) == j.size() - 1);
    CHECK_CONTAINS(j, "\"staging\":");
    CHECK_CONTAINS(j, "\"pipe\":");
    CHECK_CONTAINS(j, "\"map\":");
    CHECK_CONTAINS(j, "\"promo\":");
    CHECK_CONTAINS(j, "\"kv\":");
    CHECK_CONTAINS(j, "\"meta\":");
    CHECK_CONTAINS(j, "\"hits\":3");
    CHECK_CONTAINS(j, "\"misses\":1");
    CHECK_CONTAINS(j, "\"m_max\":2");
    CHECK_CONTAINS(j, "\"publishes\":1");
    CHECK_CONTAINS(j, "\"overflows\":1");
    CHECK_CONTAINS(j, "\"busy\":3");
    CHECK_CONTAINS(j, "\"completed\":2");
    CHECK_CONTAINS(j, "\"in_flight\":0");
    CHECK_CONTAINS(j, "\"queries\":2");
    CHECK_CONTAINS(j, "\"n_unique_missing\":3");
    CHECK_CONTAINS(j, "\"deferred\":3");
    CHECK_CONTAINS(j, "\"staging_full\":0");
    CHECK_CONTAINS(j, "\"promote_l3_to_l1\":3");
    CHECK_CONTAINS(j, "\"demote_l2_to_l3\":1");
    CHECK_CONTAINS(j, "\"l1_pages\":1");
    CHECK_CONTAINS(j, "\"l2_pages\":1");
    CHECK_CONTAINS(j, "\"l3_pages\":1");
    CHECK_CONTAINS(j, "\"generated_us\":1700000000000");
}

// JSON: zero-valued snapshot must serialize to the exact fixed-order string
static void test_json_zero_snapshot() {
    const std::string expected =
        "{"
        "\"pool\":{\"hits\":0,\"misses\":0,\"evictions\":0,\"floor_hits\":0,\"rounds\":0,\"m_sum\":0,\"m_max\":0}"
        ",\"staging\":{\"fills\":0,\"publishes\":0,\"consumes\":0,\"overflows\":0}"
        ",\"pipe\":{\"submits\":0,\"busy\":0,\"fills\":0,\"in_flight\":0,\"completed\":0,\"failed\":0,\"evictions\":0}"
        ",\"map\":{\"maps\":0,\"queries\":0,\"n_pairs\":0,\"n_staged_pairs\":0,\"n_missing_pairs\":0,\"n_unique_staged\":0,\"n_unique_missing\":0}"
        ",\"promo\":{\"runs\":0,\"reaps\":0,\"skipped\":0,\"submitted\":0,\"deferred\":0,\"promoted\":0,\"staging_full\":0,\"failed\":0}"
        ",\"kv\":{\"promote_l2_to_l1\":0,\"promote_l3_to_l1\":0,\"demote_l1_to_l2\":0,\"demote_l1_to_l3\":0,\"demote_l2_to_l3\":0,\"l1_pages\":0,\"l2_pages\":0,\"l3_pages\":0}"
        ",\"meta\":{\"generated_us\":0}"
        "}";
    CHECK(q4e_stats_v6_json(q4e_stats_v6_t()) == expected);

    // null configuration (pool only) -> zero fields for the other groups
    q4e_global_pool pool(pool_cfg());
    const q4e_stats_v6_t s = q4e_stats_v6_collect(&pool, nullptr, nullptr, nullptr, nullptr, nullptr, 42);
    const std::string j = q4e_stats_v6_json(s);
    CHECK_CONTAINS(j, "\"fills\":0");
    CHECK_CONTAINS(j, "\"submits\":0");
    CHECK_CONTAINS(j, "\"runs\":0");
    CHECK_CONTAINS(j, "\"l1_pages\":0");
    CHECK_CONTAINS(j, "\"generated_us\":42");
}

int main() {
    test_collect_full();
    test_null_configs();
    test_map_no_choices();
    test_json_full();
    test_json_zero_snapshot();
    if (g_failures) {
        fprintf(stderr, "q4e-stats-v6: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-stats-v6: all %d checks passed\n", g_checks);
    return 0;
}
