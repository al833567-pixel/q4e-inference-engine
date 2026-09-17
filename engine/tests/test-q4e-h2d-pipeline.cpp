// Unit tests for the v6 H2D promotion pipeline (common/q4e-h2d-pipeline.{h,cpp}, A-16.3).
// submit -> fill -> in-flight -> poll -> done, L2 LRU with in-flight protection, stats.
// Plus the v6 Phase 1 physical pool backend (common/q4e-pool-backend.{h,cpp}):
// L1/L2 buffer layout, slot tables, fill/H2D backends, promotion-op end-to-end.
#include "q4e-h2d-pipeline.h"
#include "q4e-pool-backend.h"
#include "q4e-promotion.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

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

#define CHECK_THROWS_LE(expr) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            (void) (expr); \
        } catch (const std::logic_error &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

#define CHECK_THROWS_RE(expr) \
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
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

#define CHECK_THROWS_CTOR(cfg) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            q4e_h2d_pipeline p(cfg); \
            (void) p; \
        } catch (const std::invalid_argument &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #cfg); \
        } \
    } while (0)

#define CHECK_THROWS_OOR(expr) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            (void) (expr); \
        } catch (const std::out_of_range &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

// test flow: submit fills L2, job goes in flight, poll completes it
static void test_submit_fill_poll() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;      // 4 slots of 1 MB
    cfg.expert_bytes = 1048576;
    cfg.max_inflight = 2;
    q4e_h2d_pipeline p(cfg);
    CHECK_EQ_I64(p.l2_slots(), 4);

    int64_t fills = 0, copies = 0;
    int fill_layer = -1, fill_expert = -1;
    uint32_t fill_slot = 99;
    p.set_fill_backend([&](int l, int e, void * dst, int64_t bytes) {
        ++fills;
        fill_layer = l; fill_expert = e;
        fill_slot = (uint32_t) (uintptr_t) dst;
        (void) bytes;
        return bytes;
    });
    p.set_h2d_backend([&](int64_t id, void * src, void * dst, int64_t bytes) {
        ++copies;
        (void) id; (void) src; (void) dst; (void) bytes;
        return true;
    });

    const int64_t j0 = p.submit(3, 42);
    CHECK_EQ_I64(j0, 0);
    CHECK_EQ_I64(fills, 1);
    CHECK_EQ_I64(fill_layer, 3);
    CHECK_EQ_I64(fill_expert, 42);
    CHECK_EQ_I64(fill_slot, (uint32_t) p.l2_slot_of(3, 42));
    CHECK(p.l2_state(3, 42) == "l2");
    CHECK_EQ_I64(p.stats().in_flight, 1);
    CHECK_EQ_I64(p.stats().submits, 1);

    const std::vector<int64_t> done = p.poll();
    CHECK_EQ_I64(done.size(), 1);
    CHECK_EQ_I64(done[0], 0);
    CHECK_EQ_I64(copies, 1);
    CHECK_EQ_I64(p.stats().completed, 1);
    CHECK_EQ_I64(p.stats().in_flight, 0);
    CHECK_EQ_I64(p.stats().fills, 1);
    CHECK_EQ_I64(p.stats().busy, 0);
    CHECK_EQ_I64(p.stats().failed, 0);
}

// test_l2_hit: a resident expert is re-submitted without another fill
static void test_l2_hit() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;
    cfg.expert_bytes = 1048576;
    q4e_h2d_pipeline p(cfg);
    int64_t fills = 0;
    p.set_fill_backend([&](int, int, void *, int64_t bytes) { ++fills; return bytes; });
    p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });

    CHECK_EQ_I64(p.l2_slot_of(0, 1), -1); // not resident yet
    CHECK_EQ_I64(p.submit(0, 1), 0);
    CHECK_EQ_I64(fills, 1);
    const int64_t slot = p.l2_slot_of(0, 1);
    CHECK(slot >= 0); // filled now
    p.poll();
    CHECK_EQ_I64(p.submit(0, 1), 0); // job slot reused
    CHECK_EQ_I64(fills, 1); // L2 hit, no second fill
    CHECK_EQ_I64(p.l2_slot_of(0, 1), slot); // same slot kept
    p.poll();
    CHECK_EQ_I64(p.stats().fills, 1);
    CHECK_EQ_I64(p.stats().submits, 2);
    CHECK_EQ_I64(p.stats().completed, 2);
}

// test_busy: the in-flight cap rejects new submits without blocking
static void test_busy() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;
    cfg.expert_bytes = 1048576;
    cfg.max_inflight = 2;
    q4e_h2d_pipeline p(cfg);
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });

    CHECK_EQ_I64(p.submit(0, 0), 0);
    CHECK_EQ_I64(p.submit(0, 1), 1);
    CHECK_EQ_I64(p.submit(0, 2), -1); // cap reached
    CHECK_EQ_I64(p.stats().busy, 1);
    CHECK_EQ_I64(p.stats().in_flight, 2);
    CHECK_EQ_I64(p.submit(0, 3), -1); // still busy
    CHECK_EQ_I64(p.stats().busy, 2);
    const std::vector<int64_t> done = p.poll();
    CHECK_EQ_I64(done.size(), 2);
    CHECK_EQ_I64(p.stats().in_flight, 0);
    CHECK(p.submit(0, 2) >= 0); // cap freed, submit works again
}

// test_poll_failure: a failed H2D copy is a FAILED event, not a done id
static void test_poll_failure() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;
    cfg.expert_bytes = 1048576;
    q4e_h2d_pipeline p(cfg);
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    std::vector<bool> script { true, false, true };
    int idx = 0;
    p.set_h2d_backend([&](int64_t, void *, void *, int64_t) {
        return script[(std::size_t) (idx < (int) script.size() ? idx++ : (int) script.size() - 1)];
    });

    CHECK_EQ_I64(p.submit(0, 0), 0);
    CHECK_EQ_I64(p.submit(0, 1), 1);
    const std::vector<int64_t> done = p.poll();
    CHECK_EQ_I64(done.size(), 1); // only the successful copy
    CHECK_EQ_I64(done[0], 0);
    CHECK_EQ_I64(p.stats().completed, 1);
    CHECK_EQ_I64(p.stats().failed, 1);
    CHECK_EQ_I64(p.stats().in_flight, 0);
    // both jobs are terminal now; a fresh submit reuses a slot
    CHECK(p.submit(0, 2) >= 0);
    CHECK_EQ_I64(p.poll().size(), 1);
    CHECK_EQ_I64(p.stats().completed, 2);
}

// test_fill_failure: a failed SSD read throws and leaks no slot
static void test_fill_failure() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;
    cfg.expert_bytes = 1048576;
    q4e_h2d_pipeline p(cfg);
    p.set_fill_backend([](int, int, void *, int64_t) { return -1; });
    CHECK_THROWS_RE(p.submit(0, 0));
    CHECK_EQ_I64(p.l2_occupancy(), 0);
    CHECK_EQ_I64(p.l2_slot_of(0, 0), -1);
    CHECK_EQ_I64(p.stats().fills, 0);
    CHECK_EQ_I64(p.stats().submits, 0);
    // the pool is intact; a working fill succeeds afterwards
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    CHECK_EQ_I64(p.submit(0, 0), 0);
    CHECK_EQ_I64(p.stats().fills, 1);
}

// test_lru_eviction: pool full evicts the oldest, MRU touch changes the victim
static void test_lru_eviction() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4; // exactly 4 slots
    cfg.expert_bytes = 1048576;
    cfg.max_inflight = 4;
    q4e_h2d_pipeline p(cfg);
    int64_t fills = 0;
    p.set_fill_backend([&](int, int, void *, int64_t bytes) { ++fills; return bytes; });
    p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });

    p.submit(0, 0); p.submit(0, 1); p.submit(0, 2); p.submit(0, 3);
    CHECK_EQ_I64(p.l2_occupancy(), 4);
    CHECK_EQ_I64(fills, 4);
    p.poll();
    // touch (0,0): it becomes MRU, so the next fill evicts (0,1) instead of (0,0)
    CHECK_EQ_I64(p.submit(0, 0), 3); // L2 hit, no fill, job slot reused
    CHECK_EQ_I64(fills, 4);
    p.poll();
    CHECK_EQ_I64(p.submit(0, 4), 3); // pool full: evict LRU
    CHECK(p.l2_state(0, 1) == "l3");
    CHECK(p.l2_state(0, 0) == "l2");
    CHECK(p.l2_state(0, 3) == "l2");
    CHECK(p.l2_state(0, 4) == "l2");
    CHECK_EQ_I64(p.l2_occupancy(), 4);
    CHECK_EQ_I64(p.stats().evictions, 1);
}

// test_eviction_protection: in-flight experts are never evicted
static void test_eviction_protection() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4; // exactly 2 slots of 2 MB
    cfg.expert_bytes = 2097152;
    cfg.max_inflight = 2;
    q4e_h2d_pipeline p(cfg);
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });

    // pool { (1,0), (1,1) } full with (1,1) in flight: the LRU victim is (1,0), the in-flight one survives
    CHECK_EQ_I64(p.submit(1, 0), 0);
    p.poll();
    CHECK_EQ_I64(p.submit(1, 1), 0); // job slot reused
    CHECK_EQ_I64(p.submit(1, 2), 1); // evicts (1,0), keeps in-flight (1,1)
    CHECK(p.l2_state(1, 0) == "l3");
    CHECK(p.l2_state(1, 1) == "l2");
    CHECK(p.l2_state(1, 2) == "l2");
    CHECK_EQ_I64(p.stats().evictions, 1);
    p.poll();

    // every resident expert in flight: no victim, submit must throw
    // max_inflight above the slot count so the in-flight cap does not fire first
    q4e_h2d_pipeline_config cfg2;
    cfg2.l2_ram_mb    = 4;
    cfg2.expert_bytes = 2097152;
    cfg2.max_inflight = 4;
    q4e_h2d_pipeline q(cfg2);
    q.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    q.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });
    CHECK_EQ_I64(q.submit(2, 0), 0);
    CHECK_EQ_I64(q.submit(2, 1), 1); // pool full, both in flight
    CHECK_THROWS_RE(q.submit(2, 2));
    CHECK_EQ_I64(q.l2_occupancy(), 2); // nothing evicted
    CHECK_EQ_I64(q.stats().evictions, 0);
}

// test_backend_not_set: state violations throw logic_error
static void test_backend_not_set() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;
    cfg.expert_bytes = 1048576;
    q4e_h2d_pipeline p(cfg);
    CHECK_THROWS_LE(p.submit(0, 0)); // L2 miss, no fill backend
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    CHECK_EQ_I64(p.submit(0, 0), 0);
    CHECK_THROWS_LE(p.poll()); // in-flight job, no h2d backend
    p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });
    CHECK_EQ_I64(p.poll().size(), 1);
}

// test_stats: every counter matches the executed protocol exactly
static void test_stats() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;
    cfg.expert_bytes = 1048576;
    cfg.max_inflight = 2;
    q4e_h2d_pipeline p(cfg);
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    std::vector<bool> script { true, false, true, true, true };
    int idx = 0;
    p.set_h2d_backend([&](int64_t, void *, void *, int64_t) {
        return script[(std::size_t) (idx < (int) script.size() ? idx++ : (int) script.size() - 1)];
    });

    CHECK_EQ_I64(p.submit(0, 0), 0);
    CHECK_EQ_I64(p.submit(0, 1), 1);
    CHECK_EQ_I64(p.submit(0, 2), -1); // busy
    p.poll(); // job 0 ok, job 1 failed
    CHECK_EQ_I64(p.submit(0, 0), 1); // L2 hit, job slot reused
    p.poll(); // job 0 ok
    CHECK_EQ_I64(p.submit(0, 2), 1);
    CHECK_EQ_I64(p.submit(0, 3), 0); // pool full
    p.poll(); // both ok
    CHECK_EQ_I64(p.submit(0, 4), 1); // evicts one
    const q4e_h2d_stats st = p.stats();
    CHECK_EQ_I64(st.submits, 6);
    CHECK_EQ_I64(st.busy, 1);
    CHECK_EQ_I64(st.fills, 5);
    CHECK_EQ_I64(st.in_flight, 1);
    CHECK_EQ_I64(st.completed, 4);
    CHECK_EQ_I64(st.failed, 1);
    CHECK_EQ_I64(st.evictions, 1);
}

// test_reset: pool, jobs and counters cleared, pipeline still usable
static void test_reset() {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb    = 4;
    cfg.expert_bytes = 1048576;
    q4e_h2d_pipeline p(cfg);
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });
    CHECK_EQ_I64(p.submit(0, 0), 0);
    CHECK_EQ_I64(p.submit(0, 1), 1);
    p.poll();
    p.reset();
    CHECK_EQ_I64(p.l2_occupancy(), 0);
    CHECK_EQ_I64(p.l2_slot_of(0, 0), -1);
    const q4e_h2d_stats st = p.stats();
    CHECK_EQ_I64(st.submits, 0);
    CHECK_EQ_I64(st.fills, 0);
    CHECK_EQ_I64(st.in_flight, 0);
    CHECK_EQ_I64(st.completed, 0);
    CHECK_EQ_I64(st.evictions, 0);
    CHECK_EQ_I64(p.submit(0, 0), 0); // fresh fill after reset
    CHECK_EQ_I64(p.stats().fills, 1);
}

// test_invalid_id: negative ids rejected on every API
static void test_invalid_id() {
    q4e_h2d_pipeline p;
    CHECK_THROWS_IA(p.submit(-1, 0));
    CHECK_THROWS_IA(p.submit(0, -1));
    CHECK_THROWS_IA(p.l2_state(-1, 0));
    CHECK_THROWS_IA(p.l2_state(0, -2));
    CHECK_THROWS_IA(p.l2_slot_of(-1, 0));
    CHECK_THROWS_IA(p.l2_slot_of(0, -1));
    CHECK_EQ_I64(p.l2_occupancy(), 0);
}

// test_ctor_validation: config bounds and the zero-slot pool
static void test_ctor_validation() {
    q4e_h2d_pipeline def; // default 4000 MB / 3216896 B
    CHECK_EQ_I64(def.l2_slots(), 1303); // 4194304000 / 3216896
    q4e_h2d_pipeline_config bad1;
    bad1.l2_ram_mb = 3;
    CHECK_THROWS_CTOR(bad1);
    q4e_h2d_pipeline_config bad2;
    bad2.expert_bytes = 0;
    CHECK_THROWS_CTOR(bad2);
    q4e_h2d_pipeline_config bad3;
    bad3.max_inflight = 0;
    CHECK_THROWS_CTOR(bad3);
    q4e_h2d_pipeline_config bad4;
    bad4.l2_ram_mb    = 4;
    bad4.expert_bytes = 8388608; // 4 MB pool < one 8 MB slot
    CHECK_THROWS_CTOR(bad4);
}

// ---- v6 Phase 1: physical pool backend (common/q4e-pool-backend.{h,cpp}) ----

// fake per-layer expert source with known values, one expert = n_ff * n_embd floats
struct pool_src {
    int n_ff = 0;
    int n_embd = 0;
    int n_exp = 0;
    std::vector<float> up;
    std::vector<float> gate;
    std::vector<float> down;

    void make(int nff, int nembd, int nexp) {
        n_ff = nff;
        n_embd = nembd;
        n_exp = nexp;
        const std::size_t elems = (std::size_t) nff * nembd;
        up.assign(elems * (std::size_t) nexp, 0.0f);
        gate.assign(elems * (std::size_t) nexp, 0.0f);
        down.assign(elems * (std::size_t) nexp, 0.0f);
        for (int e = 0; e < nexp; ++e) {
            for (std::size_t j = 0; j < elems; ++j) {
                up[(std::size_t) e * elems + j] = (float) (1000 * e + j);
                gate[(std::size_t) e * elems + j] = (float) (100000 + 1000 * e + j);
                down[(std::size_t) e * elems + j] = (float) (200000 + 1000 * e + j);
            }
        }
    }
};

// test_pool_alloc_layout: slot strides, buffer sizes, write/read-back of all slots
static void test_pool_alloc_layout() {
    q4e_pool_backend_config cfg;
    cfg.n_ff = 16;
    cfg.n_embd = 8;
    cfg.n_l1_slots = 8;
    cfg.n_l2_slots = 4;
    cfg.pool.n_experts_per_layer = 8;
    cfg.pool.n_layers = 2;
    cfg.pool.n_slots = 8;
    cfg.pool.floor = 0;
    q4e_pool_backend be(cfg);
    CHECK_EQ_I64(be.n_l1_slots(), 8);
    CHECK_EQ_I64(be.n_l2_slots(), 4);
    CHECK_EQ_I64(be.slot_bytes(), (int64_t) 16 * 8 * 4);
    CHECK_EQ_I64(be.expert_bytes(), 3 * (int64_t) 16 * 8 * 4);
    // slot-major: each slot is a contiguous [n_ff * n_embd] float block
    CHECK_EQ_I64((int64_t) (be.up_slot(1) - be.up_slot(0)), (int64_t) 16 * 8);
    CHECK_EQ_I64((int64_t) (be.gate_slot(7) - be.gate_slot(6)), (int64_t) 16 * 8);
    CHECK_EQ_I64((int64_t) (be.down_slot(3) - be.down_slot(2)), (int64_t) 16 * 8);
    CHECK_EQ_I64((int64_t) (be.l2_slot(1) - be.l2_slot(0)), 3 * (int64_t) 16 * 8);
    CHECK(be.up_slot(0) != be.gate_slot(0));
    CHECK(be.gate_slot(0) != be.down_slot(0));
    CHECK_THROWS_OOR(be.up_slot(8));
    CHECK_THROWS_OOR(be.l2_slot(4));

    // write known values into every slot, read them back
    for (int s = 0; s < be.n_l1_slots(); ++s) {
        for (int j = 0; j < 16 * 8; ++j) {
            be.up_slot(s)[j] = (float) (s * 1000 + j);
            be.gate_slot(s)[j] = (float) (100000 + s * 1000 + j);
            be.down_slot(s)[j] = (float) (200000 + s * 1000 + j);
        }
    }
    for (int s = 0; s < be.n_l2_slots(); ++s) {
        for (int j = 0; j < 3 * 16 * 8; ++j) {
            be.l2_slot(s)[j] = (float) (300000 + s * 1000 + j);
        }
    }
    for (int s = 0; s < be.n_l1_slots(); ++s) {
        for (int j = 0; j < 16 * 8; ++j) {
            CHECK(be.up_slot(s)[j] == (float) (s * 1000 + j));
            CHECK(be.gate_slot(s)[j] == (float) (100000 + s * 1000 + j));
            CHECK(be.down_slot(s)[j] == (float) (200000 + s * 1000 + j));
        }
    }
    for (int s = 0; s < be.n_l2_slots(); ++s) {
        for (int j = 0; j < 3 * 16 * 8; ++j) {
            CHECK(be.l2_slot(s)[j] == (float) (300000 + s * 1000 + j));
        }
    }
}

// test_pool_slot_table: table follows insert/evict exactly, oracle = q4e_global_pool
static void test_pool_slot_table() {
    q4e_global_pool_config pcfg;
    pcfg.n_experts_per_layer = 6;
    pcfg.n_layers = 2;
    pcfg.n_slots = 4;
    pcfg.floor = 0;
    q4e_global_pool oracle(pcfg);

    q4e_pool_backend_config cfg;
    cfg.n_ff = 4;
    cfg.n_embd = 4;
    cfg.n_l1_slots = 4;
    cfg.n_l2_slots = 2;
    cfg.pool = pcfg;
    q4e_pool_backend be(cfg);

    auto sync_check = [&]() {
        for (int l = 0; l < 2; ++l) {
            const int32_t * tb = be.slot_table(l);
            CHECK(tb != nullptr);
            for (int e = 0; e < 6; ++e) {
                const int s = oracle.slot_of(l, e);
                CHECK_EQ_I64((int64_t) tb[e], (int64_t) (s >= 0 ? s : 0));
            }
        }
    };

    CHECK(be.slot_table(2) == nullptr);
    sync_check(); // all out-of-pool -> 0

    const int seq[] = { 0, 0,  0, 1,  1, 2,  1, 3,  0, 4,  0, 5,  1, 5,  0, 3 };
    for (size_t i = 0; i < sizeof(seq) / sizeof(seq[0]); i += 2) {
        const int ev_oracle = oracle.insert(seq[i], seq[i + 1]);
        const int ev = be.insert(seq[i], seq[i + 1]);
        CHECK_EQ_I64(ev, ev_oracle);
        sync_check();
    }

    // final residents: (0,4)->slot 0, (0,5)->slot 1, (1,5)->slot 2, (0,3)->slot 3
    const int32_t * t0 = be.slot_table(0);
    const int32_t * t1 = be.slot_table(1);
    CHECK_EQ_I64((int64_t) t0[3], 3);
    CHECK_EQ_I64((int64_t) t0[4], 0); // resident in slot 0
    CHECK_EQ_I64((int64_t) t0[5], 1);
    CHECK_EQ_I64((int64_t) t1[5], 2);
    CHECK_EQ_I64((int64_t) t1[3], 0); // evicted -> 0 marker
}

// test_pool_fill: the fill backend copies the source expert slice into the L2 slot
static void test_pool_fill() {
    const int n_ff = 8, n_embd = 8, n_exp = 6;
    const int64_t ebytes = (int64_t) 3 * n_ff * n_embd * (int64_t) sizeof(float);
    pool_src src;
    src.make(n_ff, n_embd, n_exp);

    q4e_h2d_pipeline_config pcfg;
    pcfg.l2_ram_mb = 4;
    pcfg.expert_bytes = ebytes;
    pcfg.max_inflight = 1;
    q4e_h2d_pipeline pipe(pcfg);
    CHECK(pipe.l2_slots() > 0);

    q4e_pool_backend_config cfg;
    cfg.n_ff = n_ff;
    cfg.n_embd = n_embd;
    cfg.n_l1_slots = 4;
    cfg.n_l2_slots = (int) pipe.l2_slots() + 2; // must cover the pipeline slot range
    cfg.pool.n_experts_per_layer = n_exp;
    cfg.pool.n_layers = 2;
    cfg.pool.n_slots = 4;
    cfg.pool.floor = 0;
    q4e_pool_backend be(cfg);
    be.set_layer_source(1, src.up.data(), src.gate.data(), src.down.data());
    be.wire(pipe);

    // source guards
    CHECK_THROWS_IA(be.set_layer_source(2, src.up.data(), src.gate.data(), src.down.data()));
    CHECK_THROWS_IA(be.set_layer_source(0, nullptr, src.gate.data(), src.down.data()));

    const int e = 3;
    CHECK_EQ_I64(pipe.submit(1, e), 0);
    CHECK_EQ_I64(pipe.stats().fills, 1);
    const int64_t s = pipe.l2_slot_of(1, e);
    CHECK(s >= 0);
    const std::size_t elems = (std::size_t) n_ff * n_embd;
    const float * p = be.l2_slot((int) s);
    for (std::size_t k = 0; k < 3 * elems; ++k) {
        const int part = (int) (k / elems);
        const std::size_t i = k % elems;
        const float expect = (part == 0) ? src.up[(std::size_t) e * elems + i]
                          : (part == 1) ? src.gate[(std::size_t) e * elems + i]
                                       : src.down[(std::size_t) e * elems + i];
        CHECK(p[k] == expect);
    }

    // no source on layer 0: fill backend fails, the pipeline throws, no slot leaks
    pipe.poll(); // terminal the in-flight job first (the copy fails: no L1 reservation)
    CHECK_EQ_I64(pipe.stats().failed, 1);
    CHECK_THROWS_RE(pipe.submit(0, 2));
    CHECK_EQ_I64(pipe.l2_occupancy(), 1);
}

// test_pool_h2d: L2 -> L1 copy lands in the reserved pool slot; no reservation -> fail
static void test_pool_h2d() {
    const int n_ff = 8, n_embd = 8, n_exp = 6;
    const int64_t ebytes = (int64_t) 3 * n_ff * n_embd * (int64_t) sizeof(float);
    pool_src src;
    src.make(n_ff, n_embd, n_exp);

    q4e_h2d_pipeline_config pcfg;
    pcfg.l2_ram_mb = 4;
    pcfg.expert_bytes = ebytes;
    q4e_h2d_pipeline pipe(pcfg);

    q4e_pool_backend_config cfg;
    cfg.n_ff = n_ff;
    cfg.n_embd = n_embd;
    cfg.n_l1_slots = 4;
    cfg.n_l2_slots = (int) pipe.l2_slots() + 2;
    cfg.pool.n_experts_per_layer = n_exp;
    cfg.pool.n_layers = 2;
    cfg.pool.n_slots = 4;
    cfg.pool.floor = 0;
    q4e_pool_backend be(cfg);
    be.set_layer_source(1, src.up.data(), src.gate.data(), src.down.data());
    be.wire(pipe);

    const std::size_t elems = (std::size_t) n_ff * n_embd;
    auto check_pool = [&](int s, int e) {
        for (std::size_t k = 0; k < elems; ++k) {
            CHECK(be.up_slot(s)[k] == src.up[(std::size_t) e * elems + k]);
            CHECK(be.gate_slot(s)[k] == src.gate[(std::size_t) e * elems + k]);
            CHECK(be.down_slot(s)[k] == src.down[(std::size_t) e * elems + k]);
        }
    };

    // not reserved in L1: the copy fails, the expert stays out of the pool
    CHECK_EQ_I64(pipe.submit(1, 4), 0);
    CHECK_EQ_I64(pipe.poll().size(), 0);
    CHECK_EQ_I64(pipe.stats().failed, 1);
    CHECK_EQ_I64(be.slot_of(1, 4), -1);

    // reserve an L1 slot, then the same L2 slot copies in cleanly (L2 hit, no re-fill)
    CHECK_EQ_I64(be.insert(1, 4), -1);
    const int p = be.slot_of(1, 4);
    CHECK(p >= 0);
    CHECK_EQ_I64(pipe.submit(1, 4), 0);
    CHECK_EQ_I64(pipe.stats().fills, 1);
    CHECK_EQ_I64(pipe.poll().size(), 1);
    CHECK_EQ_I64(pipe.stats().failed, 1); // no new failures
    check_pool(p, 4);

    // a fresh expert end-to-end: fill + copy in one flow
    CHECK_EQ_I64(be.insert(1, 5), -1);
    const int p2 = be.slot_of(1, 5);
    CHECK(p2 >= 0);
    CHECK(p2 != p);
    CHECK_EQ_I64(pipe.submit(1, 5), 0); // job ids are reused LIFO
    CHECK_EQ_I64(pipe.stats().fills, 2);
    CHECK_EQ_I64(pipe.poll().size(), 1);
    check_pool(p2, 5);
}

// test_pool_end_to_end: promotion op -> pipeline -> backends; the expert lands in the
// pool with exact source contents and in the staging view
static void test_pool_end_to_end() {
    const int n_ff = 8, n_embd = 8, n_exp = 6;
    const int64_t ebytes = (int64_t) 3 * n_ff * n_embd * (int64_t) sizeof(float);
    pool_src src;
    src.make(n_ff, n_embd, n_exp);

    q4e_h2d_pipeline_config pcfg;
    pcfg.l2_ram_mb = 4;
    pcfg.expert_bytes = ebytes;
    pcfg.max_inflight = 2;
    q4e_h2d_pipeline pipe(pcfg);

    q4e_staging_config scfg;
    scfg.total_mb = 4;
    scfg.chunk_bytes = (uint64_t) ebytes;
    q4e_staging st(scfg);

    q4e_graph_map map(n_exp, 3, 2);
    const int choices[] = { 0, 1, 3, 2, 3, 5 };
    map.set_router_choices(choices, 6);
    const int staged[] = { 0, 1, 2, 5 };
    map.set_staged(staged, 4); // expert 3 is missing -> promotion queue

    q4e_pool_backend_config cfg;
    cfg.n_ff = n_ff;
    cfg.n_embd = n_embd;
    cfg.n_l1_slots = 4;
    cfg.n_l2_slots = (int) pipe.l2_slots() + 2;
    cfg.pool.n_experts_per_layer = n_exp;
    cfg.pool.n_layers = 2;
    cfg.pool.n_slots = 4;
    cfg.pool.floor = 0;
    q4e_pool_backend be(cfg);
    be.set_layer_source(0, src.up.data(), src.gate.data(), src.down.data());
    be.wire(pipe);

    q4e_promotion_op op(map, pipe, st, 0, ebytes);

    // the promotion path reserves the L1 slot before the copy is issued
    CHECK_EQ_I64(be.insert(0, 3), -1);
    const int p = be.slot_of(0, 3);
    CHECK(p >= 0);

    const q4e_promotion_run r = op.run();
    CHECK_EQ_I64(r.n_submitted, 1);
    CHECK_EQ_I64(r.n_busy_deferred, 0);
    CHECK_EQ_I64(r.n_failed, 0);
    CHECK_EQ_I64(pipe.stats().fills, 1);
    CHECK_EQ_I64(pipe.stats().in_flight, 1);

    const std::vector<int> promoted = op.reap();
    CHECK(promoted.size() == 1 && promoted[0] == 3);
    CHECK_EQ_I64(pipe.stats().completed, 1);
    CHECK_EQ_I64(pipe.stats().in_flight, 0);

    // the expert now sits in the pool slot with the exact source contents
    const std::size_t elems = (std::size_t) n_ff * n_embd;
    for (std::size_t k = 0; k < elems; ++k) {
        CHECK(be.up_slot(p)[k] == src.up[(std::size_t) 3 * elems + k]);
        CHECK(be.gate_slot(p)[k] == src.gate[(std::size_t) 3 * elems + k]);
        CHECK(be.down_slot(p)[k] == src.down[(std::size_t) 3 * elems + k]);
    }

    // the staging view holds the chunk; the slot table agrees with the pool
    CHECK(st.filling_buf() >= 0);
    CHECK_EQ_I64(st.placed_of(st.filling_buf()), 1);
    CHECK(op.resident(3));
    const q4e_promotion_stats os = op.stats();
    CHECK_EQ_I64(os.promoted, 1);
    CHECK_EQ_I64(os.submitted, 1);
    CHECK_EQ_I64(os.failed, 0);
    CHECK_EQ_I64((int64_t) be.slot_table(0)[3], (int64_t) p);
    CHECK_EQ_I64((int64_t) be.slot_table(0)[2], 0); // staged but not promoted: out-of-pool
    CHECK_EQ_I64(be.occupancy_of(0), 1);
}

int main() {
    test_submit_fill_poll();
    test_l2_hit();
    test_busy();
    test_poll_failure();
    test_fill_failure();
    test_lru_eviction();
    test_eviction_protection();
    test_backend_not_set();
    test_stats();
    test_reset();
    test_invalid_id();
    test_ctor_validation();
    test_pool_alloc_layout();
    test_pool_slot_table();
    test_pool_fill();
    test_pool_h2d();
    test_pool_end_to_end();
    if (g_failures) {
        fprintf(stderr, "q4e-h2d-pipeline: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-h2d-pipeline: all %d checks passed\n", g_checks);
    return 0;
}
