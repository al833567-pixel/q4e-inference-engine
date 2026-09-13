// Unit tests for the v6 global dynamic expert pool (common/q4e-global-pool.{h,cpp}, A-16.1a).
// Global LRU + Layer-Aware floor + round M counting. v5 q4e_expert_pool stays frozen.
#include "q4e-global-pool.h"

#include <cstdio>
#include <stdexcept>

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

#define CHECK_EQ_STR(a, b) \
    do { \
        ++g_checks; \
        const char * va = (a); \
        const char * vb = (b); \
        if (va == nullptr || vb == nullptr || strcmp(va, vb) != 0) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s == \"%s\", expected \"%s\"\n", __FILE__, __LINE__, #a, va ? va : "(null)", vb); \
        } \
    } while (0)

#define CHECK_THROWS(expr) \
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

#define CHECK_THROWS_CTOR(cfg) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            q4e_global_pool p(cfg); \
            (void) p; \
        } catch (const std::invalid_argument &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #cfg); \
        } \
    } while (0)

#define CHECK_THROWS_RUNTIME(expr) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            (void) (expr); \
        } catch (const std::runtime_error &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw std::runtime_error\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

static q4e_global_pool_config make_cfg(int n_experts = 512, int n_layers = 48, int n_slots = 5000, int floor = 10) {
    q4e_global_pool_config c;
    c.n_experts_per_layer = n_experts;
    c.n_layers            = n_layers;
    c.n_slots             = n_slots;
    c.floor               = floor;
    return c;
}

// test_global_lru_eviction_order: 2 layers x 4 experts, 6 global slots, floor 2
static void test_global_lru_eviction_order() {
    q4e_global_pool p(make_cfg(4, 2, 6, 2));
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK_EQ_I64(p.insert(0, 1), -1);
    CHECK_EQ_I64(p.insert(0, 2), -1);
    CHECK_EQ_I64(p.insert(0, 3), -1);
    CHECK_EQ_I64(p.insert(1, 0), -1);
    CHECK_EQ_I64(p.insert(1, 1), -1);
    CHECK_EQ_I64(p.occupancy_of(0), 4);
    CHECK_EQ_I64(p.occupancy_of(1), 2);
    CHECK_EQ_STR(p.state_of(0, 0), "l1");
    CHECK(p.slot_of(0, 0) >= 0);
    // pool full: next insert evicts the global LRU head (0,0)
    CHECK_EQ_I64(p.insert(1, 2), 0);
    CHECK_EQ_I64(p.slot_of(0, 0), -1);
    CHECK_EQ_STR(p.state_of(0, 0), "l3");
    CHECK(p.slot_of(1, 2) >= 0);
    CHECK_EQ_I64(p.insert(1, 3), 1);
    CHECK_EQ_I64(p.occupancy_of(0), 2);
    CHECK_EQ_I64(p.occupancy_of(1), 4);
    // L0 now at floor: LRU scan skips L0 entries, evicts L1 LRU head (1,0)
    CHECK_EQ_I64(p.insert(0, 0), 0);
    CHECK_EQ_I64(p.slot_of(1, 0), -1);
    CHECK(p.slot_of(0, 0) >= 0);
    const q4e_global_pool_stats s = p.stats();
    CHECK_EQ_I64(s.evictions, 3);
    CHECK_EQ_I64(s.floor_hits, 2);
    CHECK_EQ_I64(s.evictions_per_layer[0], 2);
    CHECK_EQ_I64(s.evictions_per_layer[1], 1);
    CHECK_EQ_I64(p.occupancy_of(0) + p.occupancy_of(1), 6);
}

// test_floor_protection: layer at floor never loses its floor slots; victims come from other layers
static void test_floor_protection() {
    q4e_global_pool p(make_cfg(8, 2, 6, 2));
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK_EQ_I64(p.insert(0, 1), -1); // L0 exactly at floor
    CHECK_EQ_I64(p.insert(1, 0), -1);
    CHECK_EQ_I64(p.insert(1, 1), -1);
    CHECK_EQ_I64(p.insert(1, 2), -1);
    CHECK_EQ_I64(p.insert(1, 3), -1); // pool full
    // hammer L1: victims must come from L1, L0 floor slots stay
    CHECK_EQ_I64(p.insert(1, 4), 0);
    CHECK(p.slot_of(0, 0) >= 0);
    CHECK(p.slot_of(0, 1) >= 0);
    CHECK_EQ_I64(p.slot_of(1, 0), -1);
    CHECK_EQ_I64(p.occupancy_of(0), 2);
    CHECK_EQ_I64(p.occupancy_of(1), 4);
    CHECK_EQ_I64(p.insert(1, 5), 1);
    CHECK(p.slot_of(0, 0) >= 0);
    CHECK(p.slot_of(0, 1) >= 0);
    CHECK_EQ_I64(p.slot_of(1, 1), -1);
    CHECK_EQ_I64(p.occupancy_of(0), 2);
    CHECK_EQ_I64(p.occupancy_of(1), 4);
    const q4e_global_pool_stats s = p.stats();
    CHECK_EQ_I64(s.evictions, 2);
    CHECK_EQ_I64(s.floor_hits, 4);
    CHECK_EQ_I64(s.evictions_per_layer[0], 0);
    CHECK_EQ_I64(s.evictions_per_layer[1], 2);
}

// test_all_at_floor_throws: pool full with every layer at floor
static void test_all_at_floor_throws() {
    q4e_global_pool p(make_cfg(4, 2, 4, 2));
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK_EQ_I64(p.insert(0, 1), -1);
    CHECK_EQ_I64(p.insert(1, 0), -1);
    CHECK_EQ_I64(p.insert(1, 1), -1); // full, both layers at floor
    CHECK_THROWS_RUNTIME(p.insert(0, 2));
    CHECK_EQ_I64(p.occupancy_of(0), 2);
    CHECK_EQ_I64(p.occupancy_of(1), 2);
    CHECK_EQ_I64(p.slot_of(0, 2), -1);
    CHECK_EQ_I64(p.stats().evictions, 0);
}

// test_touch_hit_miss_lru_refresh: touch refreshes LRU; hit/miss counters
static void test_touch_hit_miss_lru_refresh() {
    q4e_global_pool p(make_cfg(4, 2, 4, 1));
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK_EQ_I64(p.insert(0, 1), -1);
    CHECK_EQ_I64(p.insert(0, 2), -1);
    CHECK_EQ_I64(p.insert(0, 3), -1); // full
    CHECK(p.touch(0, 0)); // hit, refreshes LRU
    CHECK_EQ_I64(p.insert(1, 0), 1); // LRU head is now (0,1), not (0,0)
    CHECK(p.slot_of(0, 0) >= 0);
    CHECK_EQ_I64(p.slot_of(0, 1), -1);
    CHECK(!p.touch(0, 1)); // miss
    CHECK(p.touch(1, 0));  // hit
    // duplicate insert: no eviction, same slot
    const int s2 = p.slot_of(0, 2);
    CHECK_EQ_I64(p.stats().evictions, 1);
    CHECK_EQ_I64(p.insert(0, 2), -1);
    CHECK_EQ_I64(p.slot_of(0, 2), s2);
    CHECK_EQ_I64(p.stats().evictions, 1);
    const q4e_global_pool_stats s = p.stats();
    CHECK_EQ_I64(s.hits, 2);
    CHECK_EQ_I64(s.misses, 1);
    CHECK_EQ_I64(s.hits_per_layer[0], 1);
    CHECK_EQ_I64(s.hits_per_layer[1], 1);
    CHECK_EQ_I64(s.misses_per_layer[0], 1);
}

// test_round_m_counting: per-round distinct expert count (dedup), committed on end_round
static void test_round_m_counting() {
    q4e_global_pool p(make_cfg(4, 2, 6, 0));
    p.begin_round(0);
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK(p.touch(0, 0)); // duplicate, counted once
    CHECK_EQ_I64(p.insert(0, 1), -1);
    CHECK(p.touch(0, 1));
    CHECK_EQ_I64(p.insert(0, 2), -1);
    CHECK_EQ_I64(p.round_m_of(0), 3);
    p.end_round(0);
    CHECK_EQ_I64(p.round_m_of(0), 3); // committed
    p.begin_round(1);
    CHECK_EQ_I64(p.insert(1, 0), -1);
    CHECK_EQ_I64(p.insert(1, 0), -1); // duplicate
    p.end_round(1);
    CHECK_EQ_I64(p.round_m_of(1), 1);
    q4e_global_pool_stats s = p.stats();
    CHECK_EQ_I64(s.rounds, 2);
    CHECK_EQ_I64(s.m_sum, 4);
    CHECK_EQ_I64(s.m_max, 3);
    // access outside a round does not extend any round
    CHECK(!p.touch(0, 3));
    CHECK_EQ_I64(p.round_m_of(0), 3);
    CHECK_EQ_I64(p.stats().rounds, 2);
    // end_round without begin_round is a no-op
    p.end_round(0);
    CHECK_EQ_I64(p.stats().rounds, 2);
}

// test_stats_consistency: hits+misses = touches; evictions = inserts beyond pool
static void test_stats_consistency() {
    q4e_global_pool p(make_cfg(4, 2, 3, 1));
    CHECK(!p.touch(0, 0)); // miss 1
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK_EQ_I64(p.insert(0, 1), -1);
    CHECK_EQ_I64(p.insert(0, 2), -1); // pool full (3)
    CHECK(p.touch(0, 1)); // hit 1
    CHECK_EQ_I64(p.insert(1, 0), 0); // evicts (0,0)
    CHECK(p.touch(0, 1)); // hit 2
    const q4e_global_pool_stats s = p.stats();
    CHECK_EQ_I64(s.hits, 2);
    CHECK_EQ_I64(s.misses, 1);
    CHECK_EQ_I64(s.hits + s.misses, 3); // = number of touches
    CHECK_EQ_I64(s.evictions, 1);       // = distinct inserts (4) - slots (3)
    CHECK_EQ_I64(s.hits_per_layer[0], 2);
    CHECK_EQ_I64(s.misses_per_layer[0], 1);
    CHECK_EQ_I64(s.evictions_per_layer[0], 1);
    CHECK_EQ_I64(p.occupancy_of(0), 2);
    CHECK_EQ_I64(p.occupancy_of(1), 1);
    CHECK_EQ_I64(p.occupancy_of(0) + p.occupancy_of(1), 3);
}

// test_disabled_pool: n_slots = 0 is a valid disabled pool, no throw
static void test_disabled_pool() {
    q4e_global_pool p(make_cfg(4, 2, 0, 2));
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK(!p.touch(0, 0));
    CHECK_EQ_I64(p.slot_of(0, 0), -1);
    CHECK_EQ_STR(p.state_of(0, 0), "l3");
    CHECK_EQ_I64(p.occupancy_of(0), 0);
    p.begin_round(0);
    CHECK(!p.touch(0, 1));
    CHECK_EQ_I64(p.round_m_of(0), 1);
    p.end_round(0);
    q4e_global_pool_stats s = p.stats();
    CHECK_EQ_I64(s.hits, 0);
    CHECK_EQ_I64(s.misses, 2);
    CHECK_EQ_I64(s.evictions, 0);
    CHECK_EQ_I64(s.rounds, 1);
    CHECK_EQ_I64(s.m_sum, 1);
}

// test_ctor_validation: bounds, floor clamp, default config
static void test_ctor_validation() {
    q4e_global_pool def; // default 512/48/5000/10 is valid (48*10 <= 5000)
    CHECK_EQ_I64(def.cfg().n_slots, 5000);
    // floor clamped to n_experts_per_layer
    q4e_global_pool ok1(make_cfg(4, 2, 8, 1000));
    CHECK_EQ_I64(ok1.cfg().floor, 4);
    // n_slots exactly at the n_layers * floor boundary is valid
    q4e_global_pool ok2(make_cfg(4, 2, 6, 3));
    CHECK_EQ_I64(ok2.cfg().floor, 3);
    // disabled with zero floor is valid
    q4e_global_pool ok3(make_cfg(4, 2, 0, 0));
    CHECK_EQ_I64(ok3.cfg().n_slots, 0);
    CHECK_THROWS_CTOR(make_cfg(4, 2, 5, 3));   // 5 < 2*3
    CHECK_THROWS_CTOR(make_cfg(0, 2, 10, 1));  // n_experts <= 0
    CHECK_THROWS_CTOR(make_cfg(4, 0, 10, 1));  // n_layers <= 0
    CHECK_THROWS_CTOR(make_cfg(4, 2, -1, 0));  // n_slots < 0
    CHECK_THROWS_CTOR(make_cfg(4, 2, 10, -1)); // floor < 0
}

// test_invalid_ids: out-of-range layer/expert rejected on every API
static void test_invalid_ids() {
    q4e_global_pool p(make_cfg(4, 2, 6, 1));
    CHECK_THROWS(p.insert(2, 0));
    CHECK_THROWS(p.insert(99, 0));
    CHECK_THROWS(p.insert(0, 4));
    CHECK_THROWS(p.insert(0, -1));
    CHECK_THROWS(p.touch(5, 0));
    CHECK_THROWS(p.touch(0, 99));
    CHECK_THROWS(p.state_of(99, 0));
    CHECK_THROWS(p.state_of(0, 999));
    CHECK_THROWS(p.slot_of(0, -1));
    CHECK_THROWS(p.slot_of(7, 0));
    CHECK_THROWS(p.occupancy_of(-1));
    CHECK_THROWS(p.occupancy_of(4));
    CHECK_THROWS(p.begin_round(7));
    CHECK_THROWS(p.begin_round(-2));
    CHECK_THROWS(p.end_round(9));
    CHECK_THROWS(p.end_round(-3));
    CHECK_THROWS(p.round_m_of(10));
    CHECK_EQ_I64(p.occupancy_of(0), 0);
    CHECK_EQ_I64(p.stats().hits, 0);
}

// test_reset: state and counters cleared, pool still usable
static void test_reset() {
    q4e_global_pool p(make_cfg(4, 2, 4, 1));
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK_EQ_I64(p.insert(0, 1), -1);
    CHECK_EQ_I64(p.insert(1, 0), -1);
    CHECK_EQ_I64(p.insert(1, 1), -1);
    CHECK(p.touch(0, 0));
    p.begin_round(0);
    CHECK_EQ_I64(p.insert(0, 2), 1); // evicts (0,1)
    p.end_round(0);
    p.reset();
    CHECK_EQ_I64(p.occupancy_of(0), 0);
    CHECK_EQ_I64(p.occupancy_of(1), 0);
    CHECK_EQ_I64(p.slot_of(0, 0), -1);
    CHECK_EQ_STR(p.state_of(0, 0), "l3");
    CHECK_EQ_I64(p.round_m_of(0), 0);
    const q4e_global_pool_stats s = p.stats();
    CHECK_EQ_I64(s.hits, 0);
    CHECK_EQ_I64(s.misses, 0);
    CHECK_EQ_I64(s.evictions, 0);
    CHECK_EQ_I64(s.floor_hits, 0);
    CHECK_EQ_I64(s.rounds, 0);
    CHECK_EQ_I64(s.m_sum, 0);
    CHECK_EQ_I64(s.m_max, 0);
    CHECK_EQ_I64((int) s.hits_per_layer.size(), 2);
    CHECK_EQ_I64(s.hits_per_layer[0], 0);
    // pool works after reset
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK(p.slot_of(0, 0) >= 0);
    CHECK(p.touch(0, 0));
    CHECK_EQ_I64(p.stats().hits, 1);
}

// test_slot_identity: unique slots in range, freed slot reused by next insert
static void test_slot_identity() {
    q4e_global_pool p(make_cfg(4, 2, 4, 0));
    CHECK_EQ_I64(p.insert(0, 0), -1);
    CHECK_EQ_I64(p.insert(0, 1), -1);
    CHECK_EQ_I64(p.insert(0, 2), -1);
    CHECK_EQ_I64(p.insert(0, 3), -1);
    const int s00 = p.slot_of(0, 0);
    const int s01 = p.slot_of(0, 1);
    const int s02 = p.slot_of(0, 2);
    const int s03 = p.slot_of(0, 3);
    CHECK(s00 >= 0 && s00 < 4);
    CHECK(s01 >= 0 && s01 < 4);
    CHECK(s02 >= 0 && s02 < 4);
    CHECK(s03 >= 0 && s03 < 4);
    CHECK(s00 != s01 && s01 != s02 && s02 != s03);
    CHECK(s00 != s02 && s00 != s03 && s01 != s03);
    // floor 0 = pure global LRU; the freed slot is reused by the next insert
    CHECK_EQ_I64(p.insert(1, 0), 0);
    CHECK_EQ_I64(p.slot_of(1, 0), s00);
    CHECK_EQ_I64(p.slot_of(0, 0), -1);
    CHECK_EQ_I64(p.occupancy_of(0), 3);
    CHECK_EQ_I64(p.occupancy_of(1), 1);
}

int main() {
    test_global_lru_eviction_order();
    test_floor_protection();
    test_all_at_floor_throws();
    test_touch_hit_miss_lru_refresh();
    test_round_m_counting();
    test_stats_consistency();
    test_disabled_pool();
    test_ctor_validation();
    test_invalid_ids();
    test_reset();
    test_slot_identity();
    if (g_failures) {
        fprintf(stderr, "q4e-global-pool: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-global-pool: all %d checks passed\n", g_checks);
    return 0;
}
