// Unit tests for the v6 /q4e/stats server wiring: q4e_server_v6_cfg / q4e_server_v6_state
// assembly, enabled() boundary, the http/body handler shim, and reset (common/q4e-server-v6.{h,cpp}, A-16.6b).
#include "q4e-server-v6.h"

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

static void test_defaults_and_stats_json() {
    const q4e_server_v6_cfg cfg; // defaults
    q4e_server_v6_state st(cfg);

    CHECK(st.enabled());
    CHECK_EQ_I64(st.cfg().n_experts, 512);
    CHECK_EQ_I64(st.cfg().kv_ctx, 100000);
    CHECK(st.cfg().kv_bytes_per_token > 0);
    CHECK(st.cfg().expert_bytes > 0);

    // fresh state: all seven group keys present, all stats zero, now_us injected
    const std::string js = st.stats_json(12345);
    CHECK_CONTAINS(js, "\"pool\":{");
    CHECK_CONTAINS(js, "\"staging\":{");
    CHECK_CONTAINS(js, "\"pipe\":{");
    CHECK_CONTAINS(js, "\"map\":{");
    CHECK_CONTAINS(js, "\"promo\":{");
    CHECK_CONTAINS(js, "\"kv\":{");
    CHECK_CONTAINS(js, "\"meta\":{");
    CHECK_CONTAINS(js, "\"generated_us\":12345");
    CHECK_CONTAINS(js, "\"hits\":0");
    CHECK_CONTAINS(js, "\"n_pairs\":0");
    CHECK_CONTAINS(js, "\"promoted\":0");
    CHECK_CONTAINS(js, "\"promote_l2_to_l1\":0");
    CHECK_CONTAINS(js, "\"l1_pages\":0");

    // body shim: full {"v6": ...} envelope, now_us injected from the caller
    const std::string body = q4e_server_v6_body(&st, 777);
    CHECK(body.rfind("{\"v6\": ", 0) == 0);
    CHECK(body.back() == '}');
    CHECK_CONTAINS(body, "\"generated_us\":777");
    CHECK_EQ_I64(q4e_server_v6_http(&st), 200);
}

static void test_enabled_boundary() {
    q4e_server_v6_cfg off;
    off.moe_vram = 0;
    off.kv_l2_mb = 0;
    q4e_server_v6_state st_off(off); // staging/pipe stay on, so this must still construct
    CHECK(!st_off.enabled());

    q4e_server_v6_cfg on1 = off;
    on1.kv_l2_mb = 8000;
    q4e_server_v6_state st_on1(on1);
    CHECK(st_on1.enabled());

    q4e_server_v6_cfg on2 = off;
    on2.moe_vram = 500; // 500 >= 48 * 10, so the pool ctor accepts it
    q4e_server_v6_state st_on2(on2);
    CHECK(st_on2.enabled());
}

static void test_null_state_shim() {
    CHECK_EQ_I64(q4e_server_v6_http(nullptr), 500);
    CHECK(q4e_server_v6_body(nullptr, 1).empty());
}

static void test_ctor_throws() {
    q4e_server_v6_cfg bad;

    bad = q4e_server_v6_cfg(); bad.n_experts = 0;            CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.n_layers = 0;             CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.moe_vram = 5;             CHECK_THROWS_IA(q4e_server_v6_state(bad)); // 5 < 48*10
    bad = q4e_server_v6_cfg(); bad.staging_mb = 0;           CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.moe_ram = 0;              CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.max_inflight = 0;         CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.expert_bytes = 0;         CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.kv_ctx = 0;               CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.kv_page_tokens = 0;       CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.kv_bytes_per_token = 0;   CHECK_THROWS_IA(q4e_server_v6_state(bad));
    bad = q4e_server_v6_cfg(); bad.kv_l2_mb = -1;            CHECK_THROWS_IA(q4e_server_v6_state(bad));
}

static void test_reset_zero_snapshot() {
    const q4e_server_v6_cfg cfg;
    q4e_server_v6_state st(cfg);

    // drive some activity so the pre-reset snapshot is non-zero
    st.pool().insert(0, 3);
    st.pool().insert(1, 7);
    CHECK(st.pool().touch(0, 3));
    const int choices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    st.map().set_router_choices(choices, 8);

    std::string js = st.stats_json(1);
    CHECK_CONTAINS(js, "\"hits\":1");
    CHECK_CONTAINS(js, "\"maps\":1");

    st.reset();

    js = st.stats_json(2);
    CHECK_CONTAINS(js, "\"hits\":0");
    CHECK_CONTAINS(js, "\"misses\":0");
    CHECK_CONTAINS(js, "\"maps\":0");
    CHECK_CONTAINS(js, "\"promoted\":0");
    CHECK_CONTAINS(js, "\"promote_l2_to_l1\":0");
    CHECK_CONTAINS(js, "\"demote_l1_to_l2\":0");
    CHECK_CONTAINS(js, "\"l1_pages\":0");
    CHECK_CONTAINS(js, "\"l3_pages\":3125"); // 100000 tokens / 32 per page, all back to L3
    CHECK_CONTAINS(js, "\"generated_us\":2");
}

int main() {
    test_defaults_and_stats_json();
    test_enabled_boundary();
    test_null_state_shim();
    test_ctor_throws();
    test_reset_zero_snapshot();

    fprintf(stderr, "test-q4e-server-v6: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
