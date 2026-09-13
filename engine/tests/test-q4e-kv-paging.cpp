// Unit tests for the v6 100k KV cache paging manager (common/q4e-kv-paging.{h,cpp}, A-16.5a).
// L1/L2/L3 tier paging, LRU order, promote/demote event order, range boundaries,
// backend failure, stats/reset, and a fixed-seed sliding-window oracle against a reference model.
#include "q4e-kv-paging.h"

#include <algorithm>
#include <cstdio>
#include <random>
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

#define CHECK_THROWS_CTOR(...) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            q4e_kv_paging m(__VA_ARGS__); \
            (void) m; \
        } catch (const std::invalid_argument &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: ctor %s did not throw\n", __FILE__, __LINE__, #__VA_ARGS__); \
        } \
    } while (0)

static const int T1 = 0, T2 = 1, T3 = 2; // tier::L1/L2/L3 values

struct fake_backend {
    struct ev { int page, from, to; };
    std::vector<ev> events;
    bool fail_next = false;
};

static void attach(q4e_kv_paging & m, fake_backend & fb) {
    m.set_tier_backend([&fb](int page, int from, int to) {
        if (fb.fail_next) { fb.fail_next = false; return false; }
        fb.events.push_back({page, from, to});
        return true;
    });
}

static void check_tier(q4e_kv_paging & m, int page, int expect) {
    CHECK_EQ_I64((int) m.tier_of(page), expect);
}

static void check_sum(q4e_kv_paging & m, int n) {
    CHECK_EQ_I64(m.l1_pages_now() + m.l2_pages_now() + m.l3_pages_now(), n);
}

// test_basic_flow: initial all L3; promote window, demote it back, re-promote from L2
static void test_basic_flow() {
    q4e_kv_paging m(1000, 100, 4, 3, 16); // 10 pages
    fake_backend fb;
    attach(m, fb);
    for (int p = 0; p < 10; ++p) check_tier(m, p, T3);
    CHECK_EQ_I64(m.l1_pages_now(), 0);
    CHECK_EQ_I64(m.l2_pages_now(), 0);
    CHECK_EQ_I64(m.l3_pages_now(), 10);
    m.ensure_resident(0, 250); // pages 0..2
    CHECK_EQ_I64(fb.events.size(), 3);
    CHECK(fb.events[0].page == 0 && fb.events[0].from == T3 && fb.events[0].to == T1);
    CHECK(fb.events[1].page == 1 && fb.events[1].from == T3 && fb.events[1].to == T1);
    CHECK(fb.events[2].page == 2 && fb.events[2].from == T3 && fb.events[2].to == T1);
    check_tier(m, 0, T1); check_tier(m, 1, T1); check_tier(m, 2, T1);
    for (int p = 3; p < 10; ++p) check_tier(m, p, T3);
    CHECK_EQ_I64(m.l1_pages_now(), 3);
    CHECK_EQ_I64(m.l3_pages_now(), 7);
    m.demote_range(0, 250); // L1 LRU order 0,1,2
    CHECK_EQ_I64(fb.events.size(), 6);
    CHECK(fb.events[3].page == 0 && fb.events[3].from == T1 && fb.events[3].to == T2);
    CHECK(fb.events[4].page == 1 && fb.events[4].from == T1 && fb.events[4].to == T2);
    CHECK(fb.events[5].page == 2 && fb.events[5].from == T1 && fb.events[5].to == T2);
    for (int p = 0; p < 3; ++p) check_tier(m, p, T2);
    CHECK_EQ_I64(m.l1_pages_now(), 0);
    CHECK_EQ_I64(m.l2_pages_now(), 3);
    m.ensure_resident(100, 300); // pages 1,2 from L2, in L2 LRU order
    CHECK_EQ_I64(fb.events.size(), 8);
    CHECK(fb.events[6].page == 1 && fb.events[6].from == T2 && fb.events[6].to == T1);
    CHECK(fb.events[7].page == 2 && fb.events[7].from == T2 && fb.events[7].to == T1);
    check_tier(m, 0, T2);
    check_tier(m, 1, T1); check_tier(m, 2, T1);
    check_sum(m, 10);
}

// test_lru_eviction_chain: L1 full -> L2, L2 full -> L3, exact event order
static void test_lru_eviction_chain() {
    q4e_kv_paging m(1000, 100, 2, 2, 8);
    fake_backend fb;
    attach(m, fb);
    m.ensure_resident(0, 100);
    m.ensure_resident(100, 200);
    m.ensure_resident(200, 300);
    CHECK(fb.events[2].page == 0 && fb.events[2].from == T1 && fb.events[2].to == T2); // evict LRU p0
    CHECK(fb.events[3].page == 2 && fb.events[3].from == T3 && fb.events[3].to == T1);
    m.ensure_resident(300, 400);
    CHECK(fb.events[4].page == 1 && fb.events[4].from == T1 && fb.events[4].to == T2);
    CHECK(fb.events[5].page == 3 && fb.events[5].from == T3 && fb.events[5].to == T1);
    m.ensure_resident(400, 500); // L2 full: spill L2 LRU (p0) to L3 first
    CHECK(fb.events[6].page == 0 && fb.events[6].from == T2 && fb.events[6].to == T3);
    CHECK(fb.events[7].page == 2 && fb.events[7].from == T1 && fb.events[7].to == T2);
    CHECK(fb.events[8].page == 4 && fb.events[8].from == T3 && fb.events[8].to == T1);
    CHECK_EQ_I64(fb.events.size(), 9);
    check_tier(m, 0, T3); check_tier(m, 1, T2); check_tier(m, 2, T2);
    check_tier(m, 3, T1); check_tier(m, 4, T1);
    CHECK_EQ_I64(m.l1_pages_now(), 2);
    CHECK_EQ_I64(m.l2_pages_now(), 2);
    CHECK_EQ_I64(m.l3_pages_now(), 6);
    check_sum(m, 10);
}

// test_l2_off: with l2_pages = 0, L1 victims go straight to L3
static void test_l2_off() {
    q4e_kv_paging m(1000, 100, 2, 0, 8);
    fake_backend fb;
    attach(m, fb);
    m.ensure_resident(0, 100);
    m.ensure_resident(100, 200);
    m.ensure_resident(200, 300);
    CHECK(fb.events[2].page == 0 && fb.events[2].from == T1 && fb.events[2].to == T3);
    CHECK(fb.events[3].page == 2 && fb.events[3].from == T3 && fb.events[3].to == T1);
    m.ensure_resident(300, 400);
    CHECK(fb.events[4].page == 1 && fb.events[4].from == T1 && fb.events[4].to == T3);
    CHECK(fb.events[5].page == 3 && fb.events[5].from == T3 && fb.events[5].to == T1);
    check_tier(m, 0, T3); check_tier(m, 1, T3); check_tier(m, 2, T1); check_tier(m, 3, T1);
    CHECK_EQ_I64(m.l2_pages_now(), 0);
    check_sum(m, 10);
    // demote with L2 off also lands in L3
    m.demote_range(200, 400);
    CHECK(fb.events[6].page == 2 && fb.events[6].from == T1 && fb.events[6].to == T3);
    CHECK(fb.events[7].page == 3 && fb.events[7].from == T1 && fb.events[7].to == T3);
    check_tier(m, 2, T3); check_tier(m, 3, T3);
    check_sum(m, 10);
}

// test_range_boundaries: page/token mapping edges, bad ranges, partial last page
static void test_range_boundaries() {
    q4e_kv_paging m(1000, 100, 4, 2, 16);
    CHECK_EQ_I64(m.page_of(0), 0);
    CHECK_EQ_I64(m.page_of(99), 0);
    CHECK_EQ_I64(m.page_of(100), 1);
    CHECK_EQ_I64(m.page_of(999), 9);
    CHECK_THROWS_OOR(m.page_of(1000));
    CHECK_THROWS_OOR(m.page_of(-1));
    auto r0 = m.token_range_of(0);
    auto r9 = m.token_range_of(9);
    CHECK(r0.lo == 0 && r0.hi == 100);
    CHECK(r9.lo == 900 && r9.hi == 1000);
    CHECK_THROWS_OOR(m.token_range_of(10));
    CHECK_THROWS_OOR(m.token_range_of(-1));
    CHECK_THROWS_IA(m.ensure_resident(100, 50));  // lo > hi
    CHECK_THROWS_IA(m.ensure_resident(500, 500)); // empty
    CHECK_THROWS_IA(m.ensure_resident(-1, 100));
    CHECK_THROWS_IA(m.ensure_resident(0, 1001));
    CHECK_THROWS_IA(m.ensure_resident(0, 500));   // 5 pages > L1 capacity 4
    CHECK_THROWS_IA(m.demote_range(300, 200));
    // partial last page: ctx 1000, page 300 -> 4 pages, last one holds 100 tokens
    q4e_kv_paging p(1000, 300, 2, 1, 16);
    CHECK_EQ_I64(p.n_pages(), 4);
    auto rl = p.token_range_of(3);
    CHECK(rl.lo == 900 && rl.hi == 1000);
    CHECK_EQ_I64(p.page_of(999), 3);
    fake_backend fb;
    attach(p, fb);
    p.ensure_resident(950, 1000); // only page 3
    CHECK_EQ_I64(fb.events.size(), 1);
    CHECK(fb.events[0].page == 3 && fb.events[0].from == T3 && fb.events[0].to == T1);
    check_tier(p, 3, T1);
    check_tier(p, 0, T3);
    p.demote_range(900, 999);
    CHECK(fb.events[1].page == 3 && fb.events[1].from == T1 && fb.events[1].to == T2);
    check_tier(p, 3, T2);
    check_sum(p, 4);
}

// test_backend_failure: a failing backend throws runtime_error, state unchanged
static void test_backend_failure() {
    q4e_kv_paging m(1000, 100, 2, 1, 8);
    fake_backend fb;
    fb.fail_next = true;
    attach(m, fb);
    CHECK_THROWS_RE(m.ensure_resident(0, 100));
    CHECK_EQ_I64(fb.events.size(), 0);
    check_tier(m, 0, T3); // the move was not applied
    CHECK_EQ_I64(m.l1_pages_now(), 0);
    CHECK_EQ_I64(m.stats().promote_l3_to_l1, 0);
    m.ensure_resident(0, 100); // backend recovers
    CHECK_EQ_I64(fb.events.size(), 1);
    check_tier(m, 0, T1);
    // no backend set at all: moves are bookkeeping only
    q4e_kv_paging n(1000, 100, 2, 1, 8);
    n.ensure_resident(0, 200);
    check_tier(n, 0, T1);
    check_tier(n, 1, T1);
    check_sum(n, 10);
}

// test_stats_and_reset: counters match the backend event log exactly, reset clears
static void test_stats_and_reset() {
    q4e_kv_paging m(1000, 100, 2, 2, 8);
    fake_backend fb;
    attach(m, fb);
    m.ensure_resident(0, 100);
    m.ensure_resident(100, 200);
    m.ensure_resident(200, 300); // evict p0 -> L2, promote p2
    m.ensure_resident(300, 400); // evict p1 -> L2, promote p3
    m.ensure_resident(400, 500); // L2 full: spill p0 -> L3, p2 -> L2, promote p4
    m.demote_range(300, 400);    // L2 full: spill p1 -> L3, p3 -> L2
    auto count = [&](int from, int to) -> int {
        int c = 0;
        for (auto & e : fb.events) if (e.from == from && e.to == to) ++c;
        return c;
    };
    CHECK_EQ_I64(fb.events.size(), 11);
    const auto st = m.stats();
    CHECK_EQ_I64(st.promote_l2_to_l1, count(T2, T1));
    CHECK_EQ_I64(st.promote_l3_to_l1, count(T3, T1));
    CHECK_EQ_I64(st.demote_l1_to_l2, count(T1, T2));
    CHECK_EQ_I64(st.demote_l1_to_l3, count(T1, T3));
    CHECK_EQ_I64(st.demote_l2_to_l3, count(T2, T3));
    CHECK_EQ_I64(st.promote_l3_to_l1, 5);
    CHECK_EQ_I64(st.demote_l1_to_l2, 4);
    CHECK_EQ_I64(st.demote_l2_to_l3, 2);
    check_tier(m, 0, T3); check_tier(m, 1, T3); check_tier(m, 2, T2);
    check_tier(m, 3, T2); check_tier(m, 4, T1);
    CHECK_EQ_I64(m.l1_pages_now(), 1);
    CHECK_EQ_I64(m.l2_pages_now(), 2);
    check_sum(m, 10);
    m.reset();
    for (int p = 0; p < 10; ++p) check_tier(m, p, T3);
    CHECK_EQ_I64(m.l1_pages_now(), 0);
    CHECK_EQ_I64(m.l2_pages_now(), 0);
    CHECK_EQ_I64(m.l3_pages_now(), 10);
    const auto st2 = m.stats();
    CHECK(st2.promote_l2_to_l1 == 0 && st2.promote_l3_to_l1 == 0 && st2.demote_l1_to_l2 == 0);
    CHECK(st2.demote_l1_to_l3 == 0 && st2.demote_l2_to_l3 == 0);
    m.ensure_resident(0, 100); // still usable after reset
    check_tier(m, 0, T1);
    check_sum(m, 10);
}

// test_ctor_validation: bounds, clamping of L1/L2 capacity to n_pages
static void test_ctor_validation() {
    q4e_kv_paging ok(1000, 100, 4, 2, 16);
    CHECK_EQ_I64(ok.n_pages(), 10);
    CHECK_EQ_I64(ok.l1_capacity(), 4);
    CHECK_EQ_I64(ok.l2_capacity(), 2);
    CHECK_EQ_I64(ok.page_tokens(), 100);
    CHECK_EQ_I64(ok.ctx_tokens(), 1000);
    CHECK_EQ_I64(ok.kv_bytes_per_page(), 16);
    q4e_kv_paging clamped(1000, 100, 100, 999, 16);
    CHECK_EQ_I64(clamped.l1_capacity(), 10); // clamped to n_pages
    CHECK_EQ_I64(clamped.l2_capacity(), 10);
    fake_backend fb;
    attach(clamped, fb);
    clamped.ensure_resident(0, 1000); // the whole context now fits L1
    CHECK_EQ_I64(clamped.l1_pages_now(), 10);
    check_sum(clamped, 10);
    q4e_kv_paging off(1000, 100, 4, 0, 16);
    CHECK_EQ_I64(off.l2_capacity(), 0);
    CHECK_THROWS_CTOR(0, 100, 4, 2, 16);
    CHECK_THROWS_CTOR(1000, 0, 4, 2, 16);
    CHECK_THROWS_CTOR(1000, 100, 0, 2, 16);
    CHECK_THROWS_CTOR(1000, 100, 4, -1, 16);
    CHECK_THROWS_CTOR(1000, 100, 4, 2, 0);
}

// reference model: independent vector-based re-implementation of the same rules
struct ref_paging {
    int n = 0, l1_cap = 0, l2_cap = 0;
    std::vector<int> tier; // T1/T2/T3
    std::vector<int> l1, l2; // front = LRU
    uint64_t p21 = 0, p31 = 0, d12 = 0, d13 = 0, d23 = 0;
    ref_paging(int n_, int l1c, int l2c) : n(n_), l1_cap(l1c), l2_cap(l2c), tier(n_, T3) {}
    void touch(std::vector<int> & v, int p) {
        v.erase(std::remove(v.begin(), v.end(), p), v.end());
        v.push_back(p);
    }
    void make_room() {
        while ((int) l1.size() >= l1_cap) {
            const int victim = l1.front();
            l1.erase(l1.begin());
            if (l2_cap == 0) {
                tier[victim] = T3;
                ++d13;
                continue;
            }
            if ((int) l2.size() >= l2_cap) {
                const int spill = l2.front();
                l2.erase(l2.begin());
                tier[spill] = T3;
                ++d23;
            }
            tier[victim] = T2;
            l2.push_back(victim);
            ++d12;
        }
    }
    void ensure(int p_lo, int p_hi) {
        std::vector<char> need2(n, 0), need3(n, 0);
        for (int p = p_lo; p <= p_hi; ++p) {
            if (tier[p] == T1) { touch(l1, p); }
            else if (tier[p] == T2) { need2[p] = 1; }
            else { need3[p] = 1; }
        }
        const std::vector<int> l2_snap = l2;
        for (int p : l2_snap) {
            if (!need2[p]) continue;
            make_room();
            l2.erase(std::remove(l2.begin(), l2.end(), p), l2.end());
            tier[p] = T1;
            l1.push_back(p);
            ++p21;
        }
        for (int p = 0; p < n; ++p) {
            if (!need3[p]) continue;
            make_room();
            tier[p] = T1;
            l1.push_back(p);
            ++p31;
        }
    }
};

// test_oracle_sliding_window: 100k ctx, 32-token pages, 300 random slides;
// every step: window pages in L1, tier sum == n_pages, per-page tier matches the reference
static void test_oracle_sliding_window() {
    const int ctx = 100000, page = 32, l1c = 2000, l2c = 3000;
    const int n = (ctx + page - 1) / page;
    q4e_kv_paging m(ctx, page, l1c, l2c, 1234);
    ref_paging ref(n, l1c, l2c);
    CHECK_EQ_I64(m.n_pages(), n);
    const int win = 1024; // 32 pages, fits L1
    std::mt19937 rng(20260117u);
    std::uniform_int_distribution<int> dist(-4000, 4000);
    int base = 10000;
    for (int step = 0; step < 300; ++step) {
        base = std::max(0, std::min(base + dist(rng), ctx - win));
        const int lo = base, hi = base + win;
        m.ensure_resident(lo, hi);
        ref.ensure(lo / page, (hi - 1) / page);
        CHECK_EQ_I64(m.l1_pages_now() + m.l2_pages_now() + m.l3_pages_now(), n);
        for (int p = lo / page; p <= (hi - 1) / page; ++p) {
            CHECK(m.tier_of(p) == q4e_kv_paging::tier::L1);
        }
        for (int p = 0; p < n; ++p) {
            CHECK_EQ_I64((int) m.tier_of(p), ref.tier[p]);
        }
    }
    const auto st = m.stats();
    CHECK_EQ_I64(st.promote_l2_to_l1, (int64_t) ref.p21);
    CHECK_EQ_I64(st.promote_l3_to_l1, (int64_t) ref.p31);
    CHECK_EQ_I64(st.demote_l1_to_l2, (int64_t) ref.d12);
    CHECK_EQ_I64(st.demote_l1_to_l3, (int64_t) ref.d13);
    CHECK_EQ_I64(st.demote_l2_to_l3, (int64_t) ref.d23);
    CHECK_EQ_I64(m.l1_pages_now(), (int) ref.l1.size());
    CHECK_EQ_I64(m.l2_pages_now(), (int) ref.l2.size());
}

int main() {
    test_basic_flow();
    test_lru_eviction_chain();
    test_l2_off();
    test_range_boundaries();
    test_backend_failure();
    test_stats_and_reset();
    test_ctor_validation();
    test_oracle_sliding_window();
    if (g_failures) {
        fprintf(stderr, "q4e-kv-paging: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-kv-paging: all %d checks passed\n", g_checks);
    return 0;
}
