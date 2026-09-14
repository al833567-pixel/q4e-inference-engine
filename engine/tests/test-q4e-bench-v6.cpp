// Unit tests for the q4e-bench v6 micro-benchmark (tools/q4e-bench/q4e-bench-v6.{h,cpp}, A-16.7a).
// Small-iters run (ok + ops>0 per section), 100k sliding-window kv run, text report, invalid iters.
#include "../tools/q4e-bench/q4e-bench-v6.h"

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

static void test_run_small() {
    q4e_bench_v6_opts opt;
    opt.iters = 50;
    opt.kv100k = false;
    const q4e_bench_v6_result r = q4e_bench_v6_run(opt);
    CHECK(r.ok);
    CHECK(r.sections.size() == 5);
    for (const q4e_bench_v6_section & s : r.sections) {
        CHECK(s.n_ops > 0);
        CHECK(s.ops_per_sec > 0.0);
        CHECK(s.elapsed_ms >= 0.0);
    }
}

static void test_run_kv100k() {
    q4e_bench_v6_opts opt;
    opt.iters = 50;
    opt.kv100k = true;
    const q4e_bench_v6_result r = q4e_bench_v6_run(opt);
    CHECK(r.ok);
    CHECK(r.sections.size() == 5);
    if (r.sections.size() == 5) {
        // last section = kv_ensure, 300 sliding steps
        CHECK(r.sections[4].name == "kv_ensure");
        CHECK(r.sections[4].n_ops == 300);
    }
}

static void test_text() {
    q4e_bench_v6_opts opt;
    opt.iters = 50;
    const q4e_bench_v6_result r = q4e_bench_v6_run(opt);
    const std::string t = q4e_bench_v6_text(r);
    CHECK(t.find("pool_insert") != std::string::npos);
    CHECK(t.find("staging_place") != std::string::npos);
    CHECK(t.find("pipe_submit_poll") != std::string::npos);
    CHECK(t.find("map_promo_queue") != std::string::npos);
    CHECK(t.find("kv_ensure") != std::string::npos);
    CHECK(t.find("ops/sec=") != std::string::npos);
    CHECK(t.find("ok=") != std::string::npos);
    CHECK(t.find("note=") != std::string::npos);
}

static void test_invalid_iters() {
    q4e_bench_v6_opts opt;
    opt.iters = 0;
    bool threw = false;
    try {
        (void) q4e_bench_v6_run(opt);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    test_run_small();
    test_run_kv100k();
    test_text();
    test_invalid_iters();
    if (g_failures) {
        fprintf(stderr, "test-q4e-bench-v6: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("test-q4e-bench-v6: all %d checks passed\n", g_checks);
    return 0;
}
