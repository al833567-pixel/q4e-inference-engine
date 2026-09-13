// Unit tests for the v6 KV bridge: common_params v6 fields -> q4e_kv_paging
// assembly plus the decode sliding-window helper (common/q4e-kv-bridge.{h,cpp}, A-16.5b-1).
#include "q4e-kv-bridge.h"

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

static q4e_kv_cfg base_cfg() {
    q4e_kv_cfg c;
    c.ctx_tokens       = 100000;
    c.page_tokens      = 32;
    c.l1_tokens        = 64000;
    c.l2_mb            = 6000;
    c.kv_bytes_per_token = 1000;
    return c;
}

// spec example: page_bytes = 32000, n_pages = 3125,
// l2 raw = floor(6000 * 1048576 / 32000) = 196608 -> clamped to 3125, l1 = ceil(64000 / 32) = 2000
static void test_build_counts() {
    const q4e_kv_paging p = q4e_kv_bridge::build(base_cfg());
    CHECK_EQ_I64(p.n_pages(), 3125);
    CHECK_EQ_I64(p.ctx_tokens(), 100000);
    CHECK_EQ_I64(p.page_tokens(), 32);
    CHECK_EQ_I64((int64_t) p.kv_bytes_per_page(), 32000);
    CHECK_EQ_I64(p.l1_capacity(), 2000);
    CHECK_EQ_I64(p.l2_capacity(), 3125);
    CHECK_EQ_I64(p.l1_pages_now() + p.l2_pages_now() + p.l3_pages_now(), (int64_t) p.n_pages());
}

// l2_mb = 0 -> L2 off
static void test_l2_off() {
    q4e_kv_cfg c = base_cfg();
    c.l2_mb = 0;
    const q4e_kv_paging p = q4e_kv_bridge::build(c);
    CHECK_EQ_I64(p.l2_capacity(), 0);
    CHECK_EQ_I64(p.l1_capacity(), 2000);
}

// l1_tokens <= 0 -> fully resident, paging degrades
static void test_l1_fully_resident() {
    q4e_kv_cfg c = base_cfg();
    c.l1_tokens = 0;
    const q4e_kv_paging p = q4e_kv_bridge::build(c);
    CHECK_EQ_I64(p.l1_capacity(), (int) p.n_pages());
    CHECK_EQ_I64(p.l2_capacity(), 3125);

    c.l1_tokens = -5;
    const q4e_kv_paging q = q4e_kv_bridge::build(c);
    CHECK_EQ_I64(q.l1_capacity(), (int) q.n_pages());
}

// sliding window: interior, leading-edge clamp, trailing edge, span > ctx, illegal args
static void test_window() {
    const int ctx = 100;
    q4e_kv_bridge::win w;

    w = q4e_kv_bridge::window(ctx, 50, 8);
    CHECK_EQ_I64(w.lo, 43);
    CHECK_EQ_I64(w.hi, 51);

    w = q4e_kv_bridge::window(ctx, 3, 8); // leading edge clamps to 0
    CHECK_EQ_I64(w.lo, 0);
    CHECK_EQ_I64(w.hi, 4);

    w = q4e_kv_bridge::window(ctx, 99, 8); // last position, hi at ctx
    CHECK_EQ_I64(w.lo, 92);
    CHECK_EQ_I64(w.hi, 100);

    w = q4e_kv_bridge::window(ctx, 99, 1000); // span larger than ctx
    CHECK_EQ_I64(w.lo, 0);
    CHECK_EQ_I64(w.hi, 100);

    CHECK_THROWS_IA(q4e_kv_bridge::window(0, 5, 8));
    CHECK_THROWS_IA(q4e_kv_bridge::window(-1, 5, 8));
    CHECK_THROWS_IA(q4e_kv_bridge::window(ctx, -1, 8));
    CHECK_THROWS_IA(q4e_kv_bridge::window(ctx, 100, 8));
    CHECK_THROWS_IA(q4e_kv_bridge::window(ctx, 50, 0));
    CHECK_THROWS_IA(q4e_kv_bridge::window(ctx, 50, -3));
}

// every illegal build field must throw
static void test_build_validation() {
    q4e_kv_cfg c = base_cfg();

    c.ctx_tokens = 0;
    CHECK_THROWS_IA(q4e_kv_bridge::build(c));
    c = base_cfg();

    c.ctx_tokens = -10;
    CHECK_THROWS_IA(q4e_kv_bridge::build(c));
    c = base_cfg();

    c.page_tokens = 0;
    CHECK_THROWS_IA(q4e_kv_bridge::build(c));
    c = base_cfg();

    c.kv_bytes_per_token = 0;
    CHECK_THROWS_IA(q4e_kv_bridge::build(c));
    c = base_cfg();

    c.kv_bytes_per_token = -1;
    CHECK_THROWS_IA(q4e_kv_bridge::build(c));
    c = base_cfg();

    c.l2_mb = -1;
    CHECK_THROWS_IA(q4e_kv_bridge::build(c));
}

// build -> window -> ensure_resident round trip on a small config
static void test_integration_smoke() {
    q4e_kv_cfg c;
    c.ctx_tokens       = 1000;
    c.page_tokens      = 32;
    c.l1_tokens        = 64;    // l1 = 2 pages
    c.l2_mb            = 1;     // 1 MB / 32000 B = 32 pages -> clamped to n_pages
    c.kv_bytes_per_token = 1000;

    q4e_kv_paging p = q4e_kv_bridge::build(c);
    CHECK_EQ_I64(p.n_pages(), 32);
    CHECK_EQ_I64(p.l1_capacity(), 2);
    CHECK_EQ_I64(p.l2_capacity(), 32);

    // decode step at the last token, span = one page
    const q4e_kv_bridge::win w = q4e_kv_bridge::window(c.ctx_tokens, c.ctx_tokens - 1, c.page_tokens);
    CHECK_EQ_I64(w.lo, 968);
    CHECK_EQ_I64(w.hi, 1000);

    p.ensure_resident(w.lo, w.hi); // 2 pages == l1 capacity, must succeed
    CHECK_EQ_I64(p.l1_pages_now(), 2);
    CHECK_EQ_I64(p.l1_pages_now() + p.l2_pages_now() + p.l3_pages_now(), (int64_t) p.n_pages());

    // a window wider than the L1 capacity must be rejected by the paging manager
    CHECK_THROWS_IA(p.ensure_resident(0, 128));
}

int main() {
    test_build_counts();
    test_l2_off();
    test_l1_fully_resident();
    test_window();
    test_build_validation();
    test_integration_smoke();
    if (g_failures) {
        fprintf(stderr, "q4e-kv-bridge: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-kv-bridge: all %d checks passed\n", g_checks);
    return 0;
}
