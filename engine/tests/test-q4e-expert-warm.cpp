// Unit tests for the live expert page-warm backend (common/q4e-expert-warm.{h,cpp}).
// Deterministic manual mode: no worker thread, no sleeps; drain_once()/has_pending() until quiescent.
#include "q4e-expert-warm.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <set>
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

static q4e_expert_cache_config module_cfg() {
    q4e_expert_cache_config c;
    c.n_layers                = 2;
    c.n_experts_per_layer     = 4;
    c.vram_capacity_per_layer = 2;
    c.ram_capacity_per_layer  = 8;
    c.has_shared              = true;
    return c;
}

// warm declared first: destroyed last, so it always outlives the bound cache
struct module {
    q4e_expert_warm  warm;
    q4e_expert_cache cache;
    module() : warm(""), cache(module_cfg()) {}
};

// 2 layers; idx 0 = shared (256 B), idx 1..4 = experts 0..3 (2 slices x 512 B, all in range of the 2048 B heap)
static std::vector<std::vector<std::vector<q4e_warm_slice>>> make_index(const uint8_t * base) {
    std::vector<std::vector<std::vector<q4e_warm_slice>>> idx(2);
    for (int L = 0; L < 2; L++) {
        idx[L].resize(5);
        idx[L][0].push_back(q4e_warm_slice{base, 0, 256});
        for (int e = 0; e < 4; e++) {
            idx[L][e + 1].push_back(q4e_warm_slice{base, (int64_t) e * 512, 512});
            idx[L][e + 1].push_back(q4e_warm_slice{base, (int64_t) e * 512, 512});
        }
    }
    return idx;
}

static void make_module(module & m, const uint8_t * heap) {
    m.warm.set_index_for_test(make_index(heap));
    m.warm.set_queue_caps_for_test(8, 8);
    m.warm.bind(m.cache);
}

static void drain_until_quiescent(q4e_expert_warm & w) {
    while (w.has_pending()) {
        w.drain_once();
    }
}

static void test_synthetic_a1_a4_a8(const uint8_t * heap) {
    module m;
    make_module(m, heap);
    const q4e_expert_cache_stats before = m.cache.stats();

    // A1: selection is only recorded; its warm requests are drained with the next selection
    const int32_t a1[] = {2, 3, Q4E_EXPERT_SHARED_ID};
    CHECK(m.warm.enqueue_host_ids(0, a1, 3));
    drain_until_quiescent(m.warm);
    CHECK(m.warm.io_issued() == 0);
    CHECK(m.warm.io_completed() == 0);
    CHECK(m.warm.overlap_hits() == 0);
    // touch() bumps touches_ per id (3, shared included): fresh experts -> ssd, shared -> vram
    CHECK_EQ_I64(m.cache.stats().touches - before.touches, 3);
    CHECK_EQ_I64(m.cache.stats().hits.ssd - before.hits.ssd, 2);
    CHECK_EQ_I64(m.cache.stats().hits.vram - before.hits.vram, 1);
    CHECK_EQ_I64(m.cache.stats().hits.ram, 0);

    // A2: A1's selection is now warmed: experts 2, 3 (2 x 1024 B) + shared (256 B)
    const int32_t a2[] = {1, 2};
    CHECK(m.warm.enqueue_host_ids(0, a2, 2));
    drain_until_quiescent(m.warm);
    CHECK(m.warm.io_issued() == 3);
    CHECK(m.warm.io_bytes() == 2304);
    CHECK(m.warm.io_completed() == 3);

    // A3: expert 2 re-selected: overlap hit, its re-request skipped; expert 1 (1024 B) warmed
    const int32_t a3[] = {2};
    CHECK(m.warm.enqueue_host_ids(0, a3, 1));
    drain_until_quiescent(m.warm);
    CHECK(m.warm.overlap_hits() == 1);
    CHECK(m.warm.io_issued() == 4);
    CHECK(m.warm.io_bytes() == 3328);
    CHECK(m.warm.io_completed() == 4);

    // A4: second identical hint is deduped; the drained request hits the warm_done skip
    m.warm.warm_hint(0, 1);
    m.warm.warm_hint(0, 1);
    drain_until_quiescent(m.warm);
    CHECK(m.warm.io_issued() == 4);

    // A8: cache contract for A1-A3 (A4 only hinted, no touch/prefetch)
    const q4e_expert_cache_stats st = m.cache.stats();
    CHECK_EQ_I64(st.touches, 6); // touch() counts per id: 3 + 2 + 1
    // prefetch() bumps exactly one of issued/skipped_in_vram/already_staged per hint:
    // 2 batches (A2: 3 hints, A3: 2 hints) = 5 hints total
    CHECK_EQ_I64(st.prefetch.skipped_in_vram + st.prefetch.already_staged + st.prefetch.issued, 5);
    CHECK_EQ_I64(st.prefetch.skipped_in_vram, 4);
    CHECK_EQ_I64(st.prefetch.already_staged, 1);
    CHECK_EQ_I64(st.prefetch.issued, 0);
}

static void test_synthetic_a5(const uint8_t * heap) {
    module m;
    make_module(m, heap);
    m.warm.set_queue_caps_for_test(1, 2);
    m.warm.warm_hint(0, 0);
    m.warm.warm_hint(0, 1);
    m.warm.warm_hint(0, 2);
    m.warm.warm_hint(0, 3);
    CHECK(m.warm.io_dropped() == 2);
    CHECK(m.warm.io_issued() == 0);
}

static void test_synthetic_a6(const uint8_t * heap) {
    module m;
    make_module(m, heap);
    m.warm.set_queue_caps_for_test(1, 8);
    const int32_t id1[] = {1};
    const int32_t id2[] = {2};
    const int32_t id3[] = {3};
    CHECK(m.warm.enqueue_host_ids(0, id1, 1));
    CHECK(!m.warm.enqueue_host_ids(1, id2, 1));
    CHECK(!m.warm.enqueue_host_ids(0, id3, 1));
    CHECK(m.warm.events_dropped() == 2);
}

static void replay_abc(module & m) {
    const int32_t a1[] = {2, 3, Q4E_EXPERT_SHARED_ID};
    const int32_t a2[] = {1, 2};
    const int32_t a3[] = {2};
    CHECK(m.warm.enqueue_host_ids(0, a1, 3));
    drain_until_quiescent(m.warm);
    CHECK(m.warm.enqueue_host_ids(0, a2, 2));
    drain_until_quiescent(m.warm);
    CHECK(m.warm.enqueue_host_ids(0, a3, 1));
    drain_until_quiescent(m.warm);
}

static void test_synthetic_a7(const uint8_t * heap) {
    module m4;
    make_module(m4, heap);
    module m5;
    make_module(m5, heap);
    replay_abc(m4);
    replay_abc(m5);
    // same input sequence => identical counters (replay determinism)
    CHECK(m4.warm.io_issued() == m5.warm.io_issued());
    CHECK(m4.warm.io_bytes() == m5.warm.io_bytes());
    CHECK(m4.warm.io_completed() == m5.warm.io_completed());
    CHECK(m4.warm.io_dropped() == m5.warm.io_dropped());
    CHECK(m4.warm.overlap_hits() == m5.warm.overlap_hits());
    CHECK(m4.warm.events_dropped() == m5.warm.events_dropped());
    CHECK(m4.warm.io_issued() == 4);
    CHECK(m4.warm.io_bytes() == 3328);
    CHECK(m4.warm.io_completed() == 4);
    CHECK(m4.warm.io_dropped() == 0);
    CHECK(m4.warm.overlap_hits() == 1);
    CHECK(m4.warm.events_dropped() == 0);
}

static void test_slot_exhaustion(const uint8_t * heap) {
    module m;
    make_module(m, heap);
    const int32_t id[] = {1};
    CHECK(m.warm.enqueue_host_ids(0, id, 1));
    CHECK_EQ_I64(m.warm.free_slots_for_test(), 31);
    m.warm.mark_all_slots_busy_for_test();
    CHECK_EQ_I64(m.warm.free_slots_for_test(), 0);
    const int64_t dropped = m.warm.events_dropped();
    CHECK(!m.warm.enqueue_host_ids(0, id, 1));
    CHECK_EQ_I64(m.warm.events_dropped(), dropped + 1);
    drain_until_quiescent(m.warm); // worker drains the queued event and releases its slot
    CHECK(m.warm.free_slots_for_test() > 0);
    CHECK(m.warm.enqueue_host_ids(0, id, 1));
    drain_until_quiescent(m.warm);
}

static void test_touches_seen(const uint8_t * heap) {
    module m;
    make_module(m, heap);
    CHECK_EQ_I64(m.warm.touches_seen(), 0);
    replay_abc(m); // A1: 3 ids, A2: 2 ids, A3: 1 id
    CHECK_EQ_I64(m.warm.touches_seen(), 6);
}

static void test_real_gguf() {
    const char * dir = std::getenv("Q4E_WARM_REAL_DIR");
    if (dir == nullptr) {
        printf("skipping real-gguf section (Q4E_WARM_REAL_DIR unset)\n");
        return;
    }
    q4e_expert_warm real(dir);
    bool ok = false;
    try {
        ok = real.init();
    } catch (const std::exception & e) {
        fprintf(stderr, "real init failed: %s\n", e.what());
    }
    CHECK(ok);
    if (!ok) {
        return; // dtor closes whatever mappings init left open
    }
    // layer 0 expert 0 must be warmable on a real MoE model
    real.warm_hint(0, 0);
    drain_until_quiescent(real);
    CHECK(real.io_issued() >= 1 && real.io_bytes() > 0);
    // a second distinct expert: try (47,511); fall back to (0,1) if out of range
    const int64_t issued_before = real.io_issued();
    const int64_t bytes_before  = real.io_bytes();
    real.warm_hint(47, 511);
    drain_until_quiescent(real);
    int layer  = 47;
    int expert = 511;
    int64_t delta = real.io_issued() - issued_before;
    if (delta == 0) {
        layer  = 0;
        expert = 1;
        real.warm_hint(0, 1);
        drain_until_quiescent(real);
        delta = real.io_issued() - issued_before;
    }
    CHECK(delta == 1);
    printf("real-gguf: warmed expert (%d,%d) delta=%d bytes=%lld\n",
           layer, expert, (int) delta, (long long) (real.io_bytes() - bytes_before));
}

// ngram (PLE) row prefetch tests (A-14.4); fake 64 KB buffer, row0=0, rows=64 (entries=4), row=90 B

// how many of the 16 rows are distinct for a token triple (mirrors q4e_ngram_table::row_ids_for_token)
static int64_t ngram_rows_distinct(int64_t entries, int64_t t, int64_t t1, int64_t t2) {
    auto entry_of = [entries](int head, const int64_t * ids, int n_ids) -> uint64_t {
        uint8_t raw[4 + 8 * 16];
        raw[0] = (uint8_t)(head & 0xFF);
        raw[1] = (uint8_t)((head >> 8) & 0xFF);
        raw[2] = (uint8_t)((head >> 16) & 0xFF);
        raw[3] = (uint8_t)((head >> 24) & 0xFF);
        for (int i = 0; i < n_ids; i++) {
            for (int b = 0; b < 8; b++) {
                raw[4 + i * 8 + b] = (uint8_t)((ids[i] >> (8 * b)) & 0xFF);
            }
        }
        return q4e_ngram_fnv1a64(raw, 4 + (size_t) n_ids * 8) % (uint64_t) entries;
    };
    std::set<int64_t> rows;
    for (int h = 0; h < 8; h++) {
        const int64_t ids[2] = {t1, t};
        rows.insert((int64_t) entry_of(h, ids, 2) * 16 + h);
    }
    for (int h = 8; h < 16; h++) {
        const int64_t ids[3] = {t2, t1, t};
        rows.insert((int64_t) entry_of(h, ids, 3) * 16 + (h - 8));
    }
    return (int64_t) rows.size();
}

static void test_ngram_prefetch() {
    std::vector<uint8_t> nheap(64 * 1024, (uint8_t) 0xAB);

    // design a token triple whose 16 rows are all distinct (required by the full-mode check below)
    int64_t t = 0, t1 = 0, t2 = 0;
    bool found = false;
    for (int i = 0; i < 64 && !found; i++) {
        const int64_t a = 1000 + i, b = 2000 + i, c = 3000 + i;
        if (ngram_rows_distinct(4, a, b, c) == 16) {
            t = a; t1 = b; t2 = c;
            found = true;
        }
    }
    CHECK(found);
    if (!found) {
        return;
    }

    // mode off (0): hint is a no-op, nothing counted, queue stays empty
    {
        module m;
        make_module(m, nheap.data());
        m.warm.set_ngram_for_test(nheap.data(), 0, 64, 90, true);
        m.warm.set_ngram_mode(0);
        m.warm.ngram_hint(t, t1, t2);
        CHECK(m.warm.ngram_hints() == 0);
        CHECK(!m.warm.has_pending());
        CHECK(m.warm.ngram_rows_warmed() == 0);
        CHECK(m.warm.ngram_bytes_warmed() == 0);
    }

    // shim (1): hints counted, no I/O, no pending requests
    {
        module m;
        make_module(m, nheap.data());
        m.warm.set_ngram_for_test(nheap.data(), 0, 64, 90, true);
        m.warm.set_ngram_mode(1);
        m.warm.ngram_hint(t, t1, t2);
        m.warm.ngram_hint(t + 1, t1 + 1, t2 + 1);
        m.warm.ngram_hint(t + 2, t1 + 2, t2 + 2);
        CHECK(m.warm.ngram_hints() == 3);
        CHECK(m.warm.ngram_rows_warmed() == 0);
        CHECK(m.warm.ngram_bytes_warmed() == 0);
        CHECK(!m.warm.has_pending());
    }

    // full (2): 16 distinct rows enqueued and warmed; a second identical hint is deduped
    {
        module m;
        make_module(m, nheap.data());
        m.warm.set_ngram_for_test(nheap.data(), 0, 64, 90, true);
        m.warm.set_ngram_mode(2);
        m.warm.ngram_hint(t, t1, t2);
        CHECK(m.warm.ngram_hints() == 1);
        drain_until_quiescent(m.warm);
        CHECK(!m.warm.has_pending());
        CHECK_EQ_I64(m.warm.ngram_rows_warmed(), 16);
        CHECK_EQ_I64(m.warm.ngram_bytes_warmed(), 16 * 90);
        m.warm.ngram_hint(t, t1, t2); // same rows: warmed once only
        drain_until_quiescent(m.warm);
        CHECK_EQ_I64(m.warm.ngram_rows_warmed(), 16);
        CHECK_EQ_I64(m.warm.ngram_bytes_warmed(), 16 * 90);
    }

    // mode switch: full -> shim (accounting only) -> full again (rows warm once)
    {
        module m;
        make_module(m, nheap.data());
        m.warm.set_ngram_for_test(nheap.data(), 0, 64, 90, true);
        m.warm.set_ngram_mode(2);
        CHECK(m.warm.ngram_mode() == 2);
        m.warm.set_ngram_mode(1);
        CHECK(m.warm.ngram_mode() == 1);
        m.warm.ngram_hint(t, t1, t2);
        CHECK(m.warm.ngram_hints() == 1);
        CHECK(!m.warm.has_pending());
        m.warm.set_ngram_mode(2);
        CHECK(m.warm.ngram_mode() == 2);
        m.warm.ngram_hint(t, t1, t2);
        drain_until_quiescent(m.warm);
        CHECK_EQ_I64(m.warm.ngram_rows_warmed(), 16);
        CHECK_EQ_I64(m.warm.ngram_bytes_warmed(), 16 * 90);
    }

    // invalid modes clamp to off (0)
    {
        module m;
        make_module(m, nheap.data());
        m.warm.set_ngram_for_test(nheap.data(), 0, 64, 90, true);
        m.warm.set_ngram_mode(7);
        CHECK(m.warm.ngram_mode() == 0);
        m.warm.set_ngram_mode(-3);
        CHECK(m.warm.ngram_mode() == 0);
        m.warm.ngram_hint(t, t1, t2);
        CHECK(m.warm.ngram_hints() == 0);
        CHECK(!m.warm.has_pending());
    }
}

// v5 H2D promotion pipeline tests (A-15.3); deterministic tick clock, fake backend counts copies
struct h2d_fake : public q4e_expert_h2d_backend {
    int64_t calls = 0;
    int64_t bytes = 0;
    std::vector<std::pair<int, int>> order;
    void copy_expert(int layer, int expert, int64_t n) override {
        ++calls;
        bytes += n;
        order.push_back({layer, expert});
    }
};

// exact counters: bytes, dedup (queued + resident), overlap window, FIFO order, ratio
static void test_h2d_standalone_exact() {
    q4e_h2d_config cfg;
    cfg.expert_bytes = 1000;
    cfg.queue_cap    = 8;
    cfg.copy_latency = 2;
    cfg.n_layers     = 2;
    cfg.n_experts_per_layer = 8;
    h2d_fake be;
    q4e_expert_h2d h2d(&be, cfg);

    CHECK(h2d.hint(0, 0));
    CHECK(!h2d.hint(0, 0));  // dedup: already queued
    h2d.access(0, 1);        // L1 miss: lazy issue
    CHECK(!h2d.hint(0, 1));  // dedup: already queued
    CHECK_EQ_I64(h2d.pending_count(), 2);

    h2d.drain_once();        // issue both, FIFO; ready at tick 2
    CHECK_EQ_I64(h2d.h2d_issued(), 2);
    CHECK_EQ_I64(h2d.h2d_bytes(), 2000);
    CHECK_EQ_I64(h2d.h2d_completed(), 0);

    h2d.access(0, 0);        // inflight -> overlap
    h2d.access(0, 1);        // inflight -> overlap
    h2d.tick();
    h2d.drain_once();        // still inflight (ready 2 > now 1)
    h2d.access(0, 0);        // inflight -> overlap
    h2d.tick();
    h2d.drain_once();        // both complete
    CHECK_EQ_I64(h2d.h2d_completed(), 2);
    h2d.access(0, 0);        // resident -> no overlap, no re-issue
    h2d.access(0, 1);
    CHECK(!h2d.hint(0, 0));  // dedup: already resident (done set)
    CHECK(h2d.hint(0, 2));   // left queued (pending)

    CHECK_EQ_I64(h2d.dedup_skips(), 3);
    CHECK_EQ_I64(h2d.overlap_hits(), 3);
    CHECK_EQ_I64(h2d.accesses(), 6);
    CHECK_EQ_I64(h2d.h2d_dropped(), 0);
    CHECK(h2d.overlap_ratio() == 3.0 / 6.0);
    CHECK_EQ_I64(be.calls, 2);
    CHECK_EQ_I64(be.bytes, 2000);
    CHECK(be.order.size() == 2 && be.order[0].second == 0 && be.order[1].second == 1);
    CHECK(h2d.has_pending());
    CHECK_EQ_I64(h2d.pending_count(), 1);
}

// bounded FIFO: full -> drop, counted; a dropped job may be retried once the queue frees
static void test_h2d_queue_cap_drop() {
    q4e_h2d_config cfg;
    cfg.expert_bytes = 500;
    cfg.queue_cap    = 2;
    cfg.copy_latency = 1;
    cfg.n_layers     = 1;
    cfg.n_experts_per_layer = 4;
    h2d_fake be;
    q4e_expert_h2d h2d(&be, cfg);

    CHECK(h2d.hint(0, 0));
    CHECK(h2d.hint(0, 1));
    CHECK(!h2d.hint(0, 2));  // queue full -> drop
    CHECK(!h2d.hint(0, 3));  // queue full -> drop
    CHECK_EQ_I64(h2d.h2d_dropped(), 2);
    CHECK_EQ_I64(h2d.h2d_issued(), 0);

    h2d.drain_once();
    CHECK_EQ_I64(h2d.h2d_issued(), 2);
    CHECK_EQ_I64(h2d.h2d_bytes(), 1000);
    h2d.tick();
    h2d.drain_once();
    CHECK_EQ_I64(h2d.h2d_completed(), 2);

    CHECK(h2d.hint(0, 2));   // retry after drop, queue free
    h2d.drain_once();
    CHECK_EQ_I64(h2d.h2d_issued(), 3);
    CHECK_EQ_I64(h2d.h2d_bytes(), 1500);
    CHECK_EQ_I64(h2d.dedup_skips(), 0);
    CHECK_EQ_I64(h2d.accesses(), 0);
    CHECK(h2d.overlap_ratio() == 0.0);
}

// pool binding: state_of() read before issue, promote() on job done;
// L3 -> L2 fill reserve first, then L2 -> L1; already-L1 hints dedup
static void test_h2d_pool_integration() {
    q4e_expert_pool_config pcfg;
    pcfg.n_layers            = 2;
    pcfg.n_experts_per_layer = 8;
    pcfg.l1_slots_per_layer  = 2;
    pcfg.l2_slots_per_layer  = 8;
    q4e_expert_pool pool(pcfg);

    q4e_h2d_config cfg;
    cfg.expert_bytes = 64;
    cfg.queue_cap    = 8;
    cfg.copy_latency = 1;
    h2d_fake be;
    q4e_expert_h2d h2d(&be, cfg);
    h2d.bind(&pool);

    CHECK(std::strcmp(pool.state_of(0, 0), "l3") == 0);
    CHECK(h2d.hint(0, 0));         // L3: issue first job
    CHECK(!h2d.hint(0, 0));        // dedup: in flight
    h2d.drain_once();
    CHECK_EQ_I64(h2d.h2d_issued(), 1);
    h2d.access(0, 0);              // inflight -> overlap
    CHECK_EQ_I64(h2d.overlap_hits(), 1);
    h2d.tick();
    h2d.drain_once();              // complete -> pool.promote: L3 -> L2
    CHECK_EQ_I64(h2d.h2d_completed(), 1);
    CHECK(std::strcmp(pool.state_of(0, 0), "l2") == 0);
    CHECK_EQ_I64(pool.stats().l2_fills, 1);
    CHECK_EQ_I64(pool.stats().l2_to_l1, 0);

    h2d.access(0, 0);              // L2: not L1 yet -> issue second job
    CHECK(!h2d.hint(0, 0));        // dedup: in flight
    h2d.drain_once();
    h2d.access(0, 0);              // inflight -> overlap
    h2d.tick();
    h2d.drain_once();              // complete -> pool.promote: L2 -> L1
    CHECK(std::strcmp(pool.state_of(0, 0), "l1") == 0);
    CHECK_EQ_I64(pool.stats().l2_to_l1, 1);
    CHECK_EQ_I64(pool.stats().l1_evictions, 0);
    h2d.access(0, 0);              // L1: resident, no overlap
    CHECK(!h2d.hint(0, 0));        // dedup: already L1
    h2d.access(0, 1);              // L3: lazy issue, left pending

    CHECK_EQ_I64(h2d.h2d_issued(), 2);
    CHECK_EQ_I64(h2d.h2d_bytes(), 128);
    CHECK_EQ_I64(h2d.h2d_completed(), 2);
    CHECK_EQ_I64(h2d.h2d_dropped(), 0);
    CHECK_EQ_I64(h2d.dedup_skips(), 3);
    CHECK_EQ_I64(h2d.overlap_hits(), 2);
    CHECK_EQ_I64(h2d.accesses(), 5);
    CHECK(h2d.overlap_ratio() == 2.0 / 5.0);
    CHECK(std::strcmp(pool.state_of(0, 1), "l3") == 0);
    CHECK(h2d.has_pending());
    CHECK_EQ_I64(be.calls, 2);
    CHECK_EQ_I64(be.bytes, 128);
}

// ngram early hints: FIFO issue order, hint/access dedup interaction both directions
static void test_h2d_ngram_order_and_dedup() {
    q4e_h2d_config cfg;
    cfg.expert_bytes = 100;
    cfg.queue_cap    = 8;
    cfg.copy_latency = 1;
    cfg.n_layers     = 1;
    cfg.n_experts_per_layer = 16;
    h2d_fake be;
    q4e_expert_h2d h2d(&be, cfg);

    // ngram early hints, before any access; FIFO issue order
    CHECK(h2d.hint(0, 5));
    CHECK(h2d.hint(0, 6));
    h2d.drain_once();
    CHECK_EQ_I64(h2d.h2d_issued(), 2);
    CHECK_EQ_I64(be.calls, 2);
    CHECK(be.order.size() == 2 && be.order[0].second == 5 && be.order[1].second == 6);

    h2d.access(0, 5);              // early hint in flight -> overlap
    CHECK_EQ_I64(h2d.overlap_hits(), 1);
    CHECK(!h2d.hint(0, 5));        // dedup: in flight
    CHECK(h2d.hint(0, 7));
    h2d.tick();
    h2d.drain_once();              // 5, 6 done; 7 issued
    CHECK_EQ_I64(h2d.h2d_completed(), 2);
    CHECK_EQ_I64(h2d.h2d_issued(), 3);
    h2d.access(0, 6);              // resident now
    h2d.access(0, 5);
    h2d.access(0, 7);              // in flight -> overlap
    CHECK_EQ_I64(h2d.overlap_hits(), 2);
    h2d.tick();
    h2d.drain_once();              // 7 done
    h2d.access(0, 7);              // resident now

    // access before hint: access lazy-issues, the hint then dedups
    h2d.access(0, 8);
    CHECK(!h2d.hint(0, 8));
    h2d.drain_once();

    CHECK_EQ_I64(h2d.h2d_issued(), 4);
    CHECK_EQ_I64(h2d.h2d_bytes(), 400);
    CHECK_EQ_I64(h2d.h2d_completed(), 3);
    CHECK_EQ_I64(h2d.h2d_dropped(), 0);
    CHECK_EQ_I64(h2d.dedup_skips(), 2);
    CHECK_EQ_I64(h2d.accesses(), 6);
    CHECK(h2d.overlap_ratio() == 2.0 / 6.0);
    CHECK_EQ_I64(be.calls, 4);
    CHECK_EQ_I64(be.bytes, 400);
}

// range guards + ratio edge + per-hint bytes override
static void test_h2d_guards_and_edges() {
    q4e_expert_pool_config pcfg;
    pcfg.n_layers            = 2;
    pcfg.n_experts_per_layer = 8;
    q4e_expert_pool pool(pcfg);
    h2d_fake be;
    q4e_expert_h2d h2d(&be);
    h2d.bind(&pool);
    CHECK(!h2d.hint(9, 0));       // out of range: rejected, nothing counted
    CHECK(!h2d.hint(0, 511));
    h2d.access(9, 0);
    h2d.access(0, 8);
    CHECK_EQ_I64(h2d.accesses(), 0);
    CHECK_EQ_I64(h2d.dedup_skips(), 0);
    CHECK_EQ_I64(h2d.h2d_dropped(), 0);
    CHECK(!h2d.has_pending());

    q4e_expert_h2d solo(nullptr); // fresh: ratio 0, drain is a no-op
    CHECK(solo.overlap_ratio() == 0.0);
    solo.drain_once();
    CHECK_EQ_I64(solo.h2d_issued(), 0);

    q4e_h2d_config cfg;
    cfg.expert_bytes = 10;
    h2d_fake be2;
    q4e_expert_h2d h2d2(&be2, cfg);
    CHECK(h2d2.hint(0, 0, 777));  // per-hint bytes override wins
    h2d2.drain_once();
    CHECK_EQ_I64(h2d2.h2d_bytes(), 777);
    CHECK_EQ_I64(be2.bytes, 777);
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;
    // heap the synthetic index slices point at; declared before any module so it outlives them
    std::vector<uint8_t> heap(2048);
    for (int i = 0; i < (int) heap.size(); i++) {
        heap[i] = (uint8_t) (i % 251);
    }
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "q4e_warm_test";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    {
        std::ofstream f(dir / "fake.gguf", std::ios::binary);
        f.write((const char *) heap.data(), (std::streamsize) heap.size());
    }

    test_synthetic_a1_a4_a8(heap.data());
    test_synthetic_a5(heap.data());
    test_synthetic_a6(heap.data());
    test_synthetic_a7(heap.data());
    test_slot_exhaustion(heap.data());
    test_touches_seen(heap.data());
    test_ngram_prefetch();
    test_real_gguf();

    test_h2d_standalone_exact();
    test_h2d_queue_cap_drop();
    test_h2d_pool_integration();
    test_h2d_ngram_order_and_dedup();
    test_h2d_guards_and_edges();

    if (g_failures) {
        fprintf(stderr, "q4e-expert-warm: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-expert-warm: all %d checks passed\n", g_checks);
    return 0;
}
