// Unit tests for the v6 CPU e2e gate set (tools/q4e-e2e/q4e-e2e-v6.{h,cpp}, A-16.7b):
// all-pass run, kv100k run, text report, and injected H2D failure degradation.
#include "../tools/q4e-e2e/q4e-e2e-v6.h"

#include <cstddef>
#include <cstdio>
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

#define CHECK_CONTAINS(haystack, needle) \
    do { \
        ++g_checks; \
        if (std::string(haystack).find(needle) == std::string::npos) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: missing %s\n", __FILE__, __LINE__, needle); \
        } \
    } while (0)

static void test_run_all_pass() {
    const q4e_e2e_v6_result r = q4e_e2e_v6_run(false);
    CHECK(r.gates.size() == 5);
    for (size_t i = 0; i < r.gates.size(); ++i) {
        CHECK(r.gates[i].ok);
        CHECK(r.gates[i].id == (int) i + 1);
    }
    CHECK(r.all_ok);
}

static void test_run_kv100k() {
    const q4e_e2e_v6_result r = q4e_e2e_v6_run(true);
    CHECK(r.gates.size() == 5);
    CHECK(r.gates[3].ok); // gate 4 (kv) with the extra 100-step load
    CHECK(r.all_ok);
}

static void test_text() {
    const q4e_e2e_v6_result r = q4e_e2e_v6_run(false);
    const std::string t = q4e_e2e_v6_text(r);
    CHECK_CONTAINS(t, "pool_assembly");
    CHECK_CONTAINS(t, "coverage_path");
    CHECK_CONTAINS(t, "pipeline_stress");
    CHECK_CONTAINS(t, "kv_sliding_window");
    CHECK_CONTAINS(t, "stats_json");
    CHECK_CONTAINS(t, "all_ok=");
}

// injected H2D failure: gates 2/3 record failure, the rest still pass, no crash
static void test_h2d_fail_injection() {
    const q4e_e2e_v6_result r = q4e_e2e_v6_run_h2d_fail(false);
    CHECK(r.gates.size() == 5);
    CHECK(!r.gates[1].ok); // gate 2 coverage path fails
    CHECK(!r.gates[2].ok); // gate 3 pipeline stress fails
    CHECK(r.gates[0].ok);  // gate 1 pool unaffected
    CHECK(r.gates[3].ok);  // gate 4 kv unaffected
    CHECK(r.gates[4].ok);  // gate 5 stats unaffected
    CHECK(!r.all_ok);
}

int main() {
    test_run_all_pass();
    test_run_kv100k();
    test_text();
    test_h2d_fail_injection();
    if (g_failures != 0) {
        fprintf(stderr, "%d of %d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("test-q4e-e2e-v6: all %d checks passed\n", g_checks);
    return 0;
}
