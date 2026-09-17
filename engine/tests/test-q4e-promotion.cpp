// Unit tests for the v6 promotion op (common/q4e-promotion.{h,cpp}, A-16.4b).
// promotion queue -> H2D submit -> staging back-fill; skip/deferred/staging_full,
// oracle coverage against a set reference model, stats, reset, invalid args.
#include "q4e-promotion.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_set>
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

static void set_fakes(q4e_h2d_pipeline & p) {
    p.set_fill_backend([](int, int, void *, int64_t bytes) {
        return bytes;
    });
    p.set_h2d_backend([](int64_t, void *, void *, int64_t) {
        return true;
    });
}

// test_basic_flow: missing experts are submitted, reaped into a published staging buffer
static void test_basic_flow() {
    const int slot = 3216896;
    q4e_graph_map map(8, 2, 1);
    const int choices[] = { 0, 1 };
    map.set_router_choices(choices, 2);

    q4e_h2d_pipeline_config pcfg;
    pcfg.expert_bytes = slot;
    q4e_h2d_pipeline pipe(pcfg);
    set_fakes(pipe);
    q4e_staging st;
    q4e_promotion_op op(map, pipe, st, 7, slot);

    const q4e_promotion_run r = op.run();
    CHECK_EQ_I64(r.n_submitted, 2);
    CHECK_EQ_I64(r.n_busy_deferred, 0);
    CHECK_EQ_I64(r.n_failed, 0);

    const std::vector<int> promoted = op.reap();
    CHECK(promoted.size() == 2 && promoted[0] == 0 && promoted[1] == 1);

    // experts sit in a filling staging buffer, slot-checkable
    CHECK_EQ_I64(st.filling_buf(), 0);
    CHECK(st.state_of(0) == q4e_staging::state::FILLING);
    CHECK_EQ_I64(st.placed_of(0), 2);
    CHECK(op.resident(0));
    CHECK(op.resident(1));
    CHECK(!op.resident(2));
    CHECK(std::string(pipe.l2_state(7, 0)) == "l2");

    const q4e_promotion_stats s = op.stats();
    CHECK_EQ_I64(s.runs, 1);
    CHECK_EQ_I64(s.reaps, 1);
    CHECK_EQ_I64(s.submitted, 2);
    CHECK_EQ_I64(s.promoted, 2);
    CHECK_EQ_I64(s.skipped, 0);
}

// test_skip_duplicate: a resident expert is not submitted again; releasing re-allows it
static void test_skip_duplicate() {
    const int slot = 3216896;
    q4e_graph_map map(8, 2, 1);
    const int choices[] = { 0, 1 };
    map.set_router_choices(choices, 2);

    q4e_h2d_pipeline_config pcfg;
    pcfg.expert_bytes = slot;
    q4e_h2d_pipeline pipe(pcfg);
    set_fakes(pipe);
    q4e_staging st;
    q4e_promotion_op op(map, pipe, st, 0, slot);

    op.run();
    op.reap();
    CHECK_EQ_I64(st.filling_buf(), 0);

    const uint64_t submits_before = pipe.stats().submits;
    const q4e_promotion_run r2 = op.run();
    CHECK_EQ_I64(r2.n_submitted, 0);
    CHECK_EQ_I64(r2.n_busy_deferred, 0);
    CHECK(pipe.stats().submits == submits_before);
    CHECK_EQ_I64(op.stats().skipped, 2);

    st.publish(0); // batch ready
    CHECK(op.resident(0)); // still resident while published
    const q4e_promotion_run r4 = op.run();
    CHECK_EQ_I64(r4.n_submitted, 0);
    st.release(0); // expert leaves staging
    CHECK(!op.resident(0));
    const q4e_promotion_run r3 = op.run();
    CHECK_EQ_I64(r3.n_submitted, 2);
}

// test_deferred: in-flight cap 1 -> 1 submitted, 1 deferred, then the deferred one follows
static void test_deferred() {
    const int slot = 3216896;
    q4e_graph_map map(8, 2, 1);
    const int choices[] = { 5, 6 };
    map.set_router_choices(choices, 2);

    q4e_h2d_pipeline_config pcfg;
    pcfg.expert_bytes = slot;
    pcfg.max_inflight = 1;
    q4e_h2d_pipeline pipe(pcfg);
    set_fakes(pipe);
    q4e_staging st;
    q4e_promotion_op op(map, pipe, st, 2, slot);

    const q4e_promotion_run r1 = op.run();
    CHECK_EQ_I64(r1.n_submitted, 1);
    CHECK_EQ_I64(r1.n_busy_deferred, 1);

    const std::vector<int> p1 = op.reap();
    CHECK(p1.size() == 1 && p1[0] == 5);

    std::vector<int> retry;
    op.deferred_retry(retry);
    CHECK(retry.size() == 1 && retry[0] == 6);

    const q4e_promotion_run r2 = op.run();
    CHECK_EQ_I64(r2.n_submitted, 1);
    CHECK_EQ_I64(r2.n_busy_deferred, 0);

    const std::vector<int> p2 = op.reap();
    CHECK(p2.size() == 1 && p2[0] == 6);

    op.deferred_retry(retry);
    CHECK(retry.empty());
    CHECK_EQ_I64(op.stats().promoted, 2);
    CHECK_EQ_I64(op.stats().deferred, 1);
    CHECK(op.resident(5) && op.resident(6));
}

// test_staging_full: 1-slot buffer, second expert overflows -> staging_full, kept in L2
static void test_staging_full() {
    const int slot = 3216896;
    q4e_graph_map map(8, 2, 1);
    const int choices[] = { 0, 1 };
    map.set_router_choices(choices, 2);

    q4e_h2d_pipeline_config pcfg;
    pcfg.expert_bytes = slot;
    q4e_h2d_pipeline pipe(pcfg);
    set_fakes(pipe);
    q4e_staging_config scfg;
    scfg.total_mb = 8; // 4 MB per buffer / 1 slot -> exactly one chunk fits
    q4e_staging st(scfg);
    q4e_promotion_op op(map, pipe, st, 4, slot);

    const q4e_promotion_run r = op.run();
    CHECK_EQ_I64(r.n_submitted, 2);

    const std::vector<int> promoted = op.reap();
    CHECK(promoted.size() == 1 && promoted[0] == 0);

    const q4e_promotion_stats s = op.stats();
    CHECK_EQ_I64(s.promoted, 1);
    CHECK_EQ_I64(s.staging_full, 1);
    CHECK_EQ_I64(st.stats().overflows, 1);
    CHECK(!op.resident(1));
    CHECK(std::string(pipe.l2_state(4, 1)) == "l2"); // not dropped
    CHECK_EQ_I64(st.filling_buf(), 0);
    CHECK_EQ_I64(st.placed_of(0), 1);
    CHECK(st.state_of(0) == q4e_staging::state::FILLING);
}

// test_failed_jobs: a failed H2D job is counted, surfaced as n_failed on the next run
static void test_failed_jobs() {
    const int slot = 3216896;
    q4e_graph_map map(8, 2, 1);
    const int choices[] = { 0, 1 };
    map.set_router_choices(choices, 2);

    q4e_h2d_pipeline_config pcfg;
    pcfg.expert_bytes = slot;
    q4e_h2d_pipeline pipe(pcfg);
    pipe.set_fill_backend([](int, int, void *, int64_t bytes) {
        return bytes;
    });
    int fail_budget = 1;
    pipe.set_h2d_backend([&fail_budget](int64_t, void *, void *, int64_t) {
        if (fail_budget > 0) {
            --fail_budget;
            return false;
        }
        return true;
    });
    q4e_staging st;
    q4e_promotion_op op(map, pipe, st, 0, slot);

    op.run();
    const std::vector<int> promoted = op.reap();
    CHECK(promoted.size() == 1 && promoted[0] == 1);
    CHECK_EQ_I64(op.stats().failed, 1);
    CHECK(!op.resident(0));

    CHECK_EQ_I64(op.run().n_failed, 1); // carry-over
    CHECK_EQ_I64(op.run().n_failed, 0); // reported once
}

// test_oracle: random router choices, reference set model, full coverage and first-seen order
static void test_oracle() {
    const int n_tokens  = 64;
    const int top_k     = 8;
    const int n_experts = 512;
    const int slot      = 3216896;

    std::mt19937 rng(0x5EEDu);
    std::uniform_int_distribution<int> dist(0, n_experts - 1);
    std::vector<int> choices((size_t) n_tokens * top_k);
    for (int & c : choices) {
        c = dist(rng);
    }
    // partial staged set: 200 distinct experts
    std::unordered_set<int> staged_set;
    while (staged_set.size() < 200) {
        staged_set.insert(dist(rng));
    }
    std::vector<int> staged(staged_set.begin(), staged_set.end());

    // reference model: expected = all choices - initial staged
    std::set<int> expected;
    for (const int c : choices) {
        if (!staged_set.count(c)) {
            expected.insert(c);
        }
    }
    CHECK(expected.size() >= 1);

    q4e_graph_map map(n_experts, top_k, n_tokens);
    map.set_router_choices(choices.data(), (int) choices.size());
    map.set_staged(staged.data(), (int) staged.size());

    std::vector<int> first_seen;
    map.promotion_queue(first_seen);
    CHECK((int) first_seen.size() == (int) expected.size());
    std::unordered_map<int, int> index;
    for (int i = 0; i < (int) first_seen.size(); ++i) {
        index[first_seen[i]] = i;
    }

    q4e_h2d_pipeline_config pcfg;
    pcfg.expert_bytes = slot;
    pcfg.max_inflight = 4;
    q4e_h2d_pipeline pipe(pcfg);
    set_fakes(pipe);
    q4e_staging_config scfg;
    scfg.total_mb = 10240; // 5120 MB per buffer, holds all 512 experts
    q4e_staging st(scfg);
    q4e_promotion_op op(map, pipe, st, 9, slot);

    std::vector<int> promoted_all;
    int runs = 0;
    while (true) {
        ++runs;
        if (runs > 2000) { // safety net: no live-lock
            CHECK(false);
            break;
        }
        op.run();
        const std::vector<int> p = op.reap();
        promoted_all.insert(promoted_all.end(), p.begin(), p.end());
        std::vector<int> retry;
        op.deferred_retry(retry);
        if (retry.empty()) {
            break;
        }
    }
    CHECK_EQ_I64(pipe.stats().in_flight, 0);
    CHECK_EQ_I64(pipe.stats().failed, 0);

    // coverage: every expected expert is resident in staging
    for (const int e : expected) {
        CHECK(op.resident(e));
    }
    // promoted set == expected set, each expert exactly once
    CHECK(promoted_all.size() == expected.size());
    std::set<int> promoted_set(promoted_all.begin(), promoted_all.end());
    CHECK(promoted_set == expected);

    // order: promoted sequence is a subsequence of the first-seen order
    int last = -1;
    for (const int e : promoted_all) {
        const int i = index[e];
        CHECK(i > last);
        last = i;
    }

    // stats: replay the run loop as a reference model
    const int cap = 4;
    uint64_t ref_submitted = 0, ref_deferred = 0, ref_skipped = 0;
    int residents = 0, ref_runs = 0;
    while (true) {
        ++ref_runs;
        const int remaining = (int) expected.size() - residents;
        const int submit = std::min(cap, remaining);
        ref_submitted += (uint64_t) submit;
        ref_deferred  += (uint64_t) (remaining - submit);
        ref_skipped   += (uint64_t) residents;
        residents += submit;
        if (remaining - submit == 0) {
            break;
        }
    }
    const q4e_promotion_stats s = op.stats();
    CHECK_EQ_I64(s.runs, runs);
    CHECK_EQ_I64(s.runs, ref_runs);
    CHECK_EQ_I64(s.submitted, ref_submitted);
    CHECK_EQ_I64(s.deferred, ref_deferred);
    CHECK_EQ_I64(s.skipped, ref_skipped);
    CHECK_EQ_I64(s.promoted, expected.size());
    CHECK_EQ_I64(s.staging_full, 0);
    CHECK_EQ_I64(s.failed, 0);
}

// test_run_without_choices: map without router choices throws logic_error
static void test_run_without_choices() {
    q4e_graph_map map(8, 2, 1); // no set_router_choices
    q4e_h2d_pipeline pipe;
    q4e_staging st;
    q4e_promotion_op op(map, pipe, st, 0, 1000);
    CHECK_THROWS_LE(op.run());
}

// test_ctor_and_reset: invalid args throw; reset clears op state but keeps the trio usable
static void test_ctor_and_reset() {
    q4e_graph_map map(8, 2, 1);
    const int choices[] = { 0, 1 };
    map.set_router_choices(choices, 2);
    q4e_h2d_pipeline_config pcfg;
    pcfg.expert_bytes = 3216896;
    q4e_h2d_pipeline pipe(pcfg);
    set_fakes(pipe);
    q4e_staging st;

    CHECK_THROWS_IA(q4e_promotion_op(map, pipe, st, -1, 1000));
    CHECK_THROWS_IA(q4e_promotion_op(map, pipe, st, 0, 0));
    CHECK_THROWS_IA(q4e_promotion_op(map, pipe, st, 0, (int64_t) st.chunk_bytes() + 1));
    q4e_promotion_op ok(map, pipe, st, 0, (int64_t) st.chunk_bytes()); // boundary is valid
    (void) ok;

    q4e_promotion_op op(map, pipe, st, 1, 3216896);
    op.run();
    op.reap();
    CHECK(op.resident(0));

    op.reset();
    const q4e_promotion_stats s = op.stats();
    CHECK(s.runs == 0 && s.reaps == 0 && s.skipped == 0 && s.submitted == 0);
    CHECK(s.deferred == 0 && s.promoted == 0 && s.staging_full == 0 && s.failed == 0);
    CHECK(!op.resident(0));
    std::vector<int> retry;
    op.deferred_retry(retry);
    CHECK(retry.empty());

    const q4e_promotion_run r = op.run(); // usable again: re-submits
    CHECK_EQ_I64(r.n_submitted, 2);
    (void) pipe;
}

int main() {
    test_basic_flow();
    test_skip_duplicate();
    test_deferred();
    test_staging_full();
    test_failed_jobs();
    test_oracle();
    test_run_without_choices();
    test_ctor_and_reset();
    if (g_failures) {
        fprintf(stderr, "q4e-promotion: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-promotion: all %d checks passed\n", g_checks);
    return 0;
}
