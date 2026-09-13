// Unit tests for the v6 graph dynamic slot map (common/q4e-graph-map.{h,cpp}, A-16.4a).
// staged/missing split of router top_k, promotion queue dedup + first-seen order,
// aggregate consistency, staged set dedup, exception contract, reset.
#include "q4e-graph-map.h"

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

#define CHECK_THROWS_CTOR_GM(a, b, c) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            q4e_graph_map m(a, b, c); \
            (void) m; \
        } catch (const std::invalid_argument &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: q4e_graph_map(%s,%s,%s) did not throw\n", __FILE__, __LINE__, #a, #b, #c); \
        } \
    } while (0)

// test_mixed_split: exact staged/missing order per token, shared expert 2, promotion order
static void test_mixed_split() {
    q4e_graph_map m(8, 3, 2);
    const int choices[6] = { 0, 1, 2,  2, 5, 6 };
    const int staged[3]  = { 1, 2, 6 };
    m.set_router_choices(choices, 6);
    m.set_staged(staged, 3);

    const q4e_token_view t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 2);
    CHECK_EQ_I64(t0.n_missing, 1);
    CHECK(t0.staged[0] == 1 && t0.staged[1] == 2); // choice order
    CHECK(t0.missing[0] == 0);

    const q4e_token_view t1 = m.token_view(1);
    CHECK_EQ_I64(t1.n_staged, 2);
    CHECK_EQ_I64(t1.n_missing, 1);
    CHECK(t1.staged[0] == 2 && t1.staged[1] == 6);
    CHECK(t1.missing[0] == 5);

    std::vector<int> promo;
    m.promotion_queue(promo);
    CHECK(promo.size() == 2 && promo[0] == 0 && promo[1] == 5);

    const q4e_graph_map_agg a = m.aggregate();
    CHECK_EQ_I64(a.n_pairs, 6);
    CHECK_EQ_I64(a.n_staged_pairs, 4);
    CHECK_EQ_I64(a.n_missing_pairs, 2);
    CHECK_EQ_I64(a.n_unique_staged, 3);
    CHECK_EQ_I64(a.n_unique_missing, 2);
}

// test_all_missing: empty staging view, dedup + first-seen promotion order across tokens
static void test_all_missing() {
    q4e_graph_map m(16, 2, 3);
    const int choices[6] = { 3, 7,  7, 3,  7, 9 }; // expert 7 shared by all 3 tokens
    m.set_router_choices(choices, 6);
    m.set_staged(nullptr, 0);
    const q4e_token_view t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 0);
    CHECK_EQ_I64(t0.n_missing, 2);
    std::vector<int> promo;
    m.promotion_queue(promo);
    CHECK(promo.size() == 3 && promo[0] == 3 && promo[1] == 7 && promo[2] == 9);
    const q4e_graph_map_agg a = m.aggregate();
    CHECK_EQ_I64(a.n_pairs, 6);
    CHECK_EQ_I64(a.n_staged_pairs, 0);
    CHECK_EQ_I64(a.n_missing_pairs, 6);
    CHECK_EQ_I64(a.n_unique_staged, 0);
    CHECK_EQ_I64(a.n_unique_missing, 3);
}

// test_all_staged: everything in the staging view, empty promotion queue
static void test_all_staged() {
    q4e_graph_map m(16, 2, 2);
    const int choices[4] = { 3, 7,  7, 3 };
    const int staged[2]  = { 7, 3 };
    m.set_router_choices(choices, 4);
    m.set_staged(staged, 2);
    const q4e_token_view t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 2);
    CHECK_EQ_I64(t0.n_missing, 0);
    CHECK(t0.staged[0] == 3 && t0.staged[1] == 7);
    std::vector<int> promo;
    m.promotion_queue(promo);
    CHECK(promo.empty());
    const q4e_graph_map_agg a = m.aggregate();
    CHECK_EQ_I64(a.n_pairs, 4);
    CHECK_EQ_I64(a.n_staged_pairs, 4);
    CHECK_EQ_I64(a.n_missing_pairs, 0);
    CHECK_EQ_I64(a.n_unique_staged, 2);
    CHECK_EQ_I64(a.n_unique_missing, 0);
}

// test_staged_dedup: duplicated staged input counts once; staged set may be set before choices
static void test_staged_dedup() {
    q4e_graph_map m(4, 2, 1);
    const int staged[5] = { 1, 1, 2, 2, 1 };
    m.set_staged(staged, 5);
    const int choices[2] = { 1, 2 };
    m.set_router_choices(choices, 2);
    const q4e_token_view t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 2);
    CHECK_EQ_I64(t0.n_missing, 0);
    const q4e_graph_map_agg a = m.aggregate();
    CHECK_EQ_I64(a.n_pairs, 2);
    CHECK_EQ_I64(a.n_staged_pairs, 2);
    CHECK_EQ_I64(a.n_unique_staged, 2);
    CHECK_EQ_I64(a.n_unique_missing, 0);
}

// test_stats: maps and queries counters
static void test_stats() {
    q4e_graph_map m(8, 3, 2);
    const int choices[6] = { 0, 1, 2,  2, 5, 6 };
    const int staged[3]  = { 1, 2, 6 };
    CHECK_EQ_I64(m.stats().maps, 0);
    CHECK_EQ_I64(m.stats().queries, 0);
    m.set_router_choices(choices, 6);
    m.set_router_choices(choices, 6); // re-map
    m.set_staged(staged, 3);
    (void) m.token_view(0);
    (void) m.token_view(1);
    (void) m.token_view(0);
    const q4e_graph_map_stats s = m.stats();
    CHECK_EQ_I64(s.maps, 2);
    CHECK_EQ_I64(s.queries, 3);
}

// test_exceptions: ctor, empty-choices state, bad ids, bad n, token index bounds
static void test_exceptions() {
    CHECK_THROWS_CTOR_GM(0, 2, 2);
    CHECK_THROWS_CTOR_GM(512, 0, 2);
    CHECK_THROWS_CTOR_GM(512, 2, 0);
    CHECK_THROWS_CTOR_GM(-1, 2, 2);
    q4e_graph_map m(8, 3, 2);
    // choices not set
    CHECK_THROWS_LE(m.token_view(0));
    std::vector<int> promo;
    CHECK_THROWS_LE(m.promotion_queue(promo));
    CHECK_THROWS_LE(m.aggregate());
    // invalid router choices
    CHECK_THROWS_IA(m.set_router_choices(nullptr, 6));
    const int bad_n[4] = { 0, 1, 2, 3 };
    CHECK_THROWS_IA(m.set_router_choices(bad_n, 4)); // n != n_tokens * top_k
    const int bad_range[6] = { 0, 1, 2, 2, 5, 8 }; // 8 >= n_experts
    CHECK_THROWS_IA(m.set_router_choices(bad_range, 6));
    const int bad_neg[6] = { 0, 1, 2, 2, 5, -1 };
    CHECK_THROWS_IA(m.set_router_choices(bad_neg, 6));
    // invalid staged set
    const int bad_staged[1] = { 9 };
    CHECK_THROWS_IA(m.set_staged(bad_staged, 1));
    CHECK_THROWS_IA(m.set_staged(bad_staged, -1));
    CHECK_THROWS_IA(m.set_staged(nullptr, 3));
    // token index bounds (valid choices in place)
    const int choices[6] = { 0, 1, 2,  2, 5, 6 };
    m.set_router_choices(choices, 6);
    CHECK_THROWS_OOR(m.token_view(-1));
    CHECK_THROWS_OOR(m.token_view(2));
    (void) m.token_view(0); // valid
}

// test_reset: state and stats cleared, object usable again
static void test_reset() {
    q4e_graph_map m(8, 3, 2);
    const int choices[6] = { 0, 1, 2,  2, 5, 6 };
    const int staged[3]  = { 1, 2, 6 };
    m.set_router_choices(choices, 6);
    m.set_staged(staged, 3);
    (void) m.token_view(0);
    m.reset();
    const q4e_graph_map_stats s = m.stats();
    CHECK_EQ_I64(s.maps, 0);
    CHECK_EQ_I64(s.queries, 0);
    CHECK_THROWS_LE(m.token_view(0)); // choices gone
    std::vector<int> promo;
    CHECK_THROWS_LE(m.promotion_queue(promo));
    CHECK_THROWS_LE(m.aggregate());
    m.set_router_choices(choices, 6);
    m.set_staged(staged, 3);
    const q4e_token_view t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 2);
    m.promotion_queue(promo);
    CHECK(promo.size() == 2 && promo[0] == 0 && promo[1] == 5);
}

// test_boundaries: n_experts = 2, top_k = 1, n_tokens = 1, ids at both range ends
static void test_boundaries() {
    q4e_graph_map m(2, 1, 1);
    const int choices[1] = { 1 };
    m.set_router_choices(choices, 1);
    m.set_staged(nullptr, 0);
    q4e_token_view t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 0);
    CHECK_EQ_I64(t0.n_missing, 1);
    std::vector<int> promo;
    m.promotion_queue(promo);
    CHECK(promo.size() == 1 && promo[0] == 1);
    // duplicated staged input, id 0 selected by no token
    const int staged[3] = { 1, 0, 1 };
    m.set_staged(staged, 3);
    t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 1);
    CHECK_EQ_I64(t0.n_missing, 0);
    m.promotion_queue(promo);
    CHECK(promo.empty());
    const q4e_graph_map_agg a = m.aggregate();
    CHECK_EQ_I64(a.n_pairs, 1);
    CHECK_EQ_I64(a.n_staged_pairs, 1);
    CHECK_EQ_I64(a.n_missing_pairs, 0);
    CHECK_EQ_I64(a.n_unique_staged, 2);
    CHECK_EQ_I64(a.n_unique_missing, 0);
    // shrink the staged set: expert 1 missing again
    const int staged2[1] = { 0 };
    m.set_staged(staged2, 1);
    t0 = m.token_view(0);
    CHECK_EQ_I64(t0.n_staged, 0);
    CHECK_EQ_I64(t0.n_missing, 1);
    m.promotion_queue(promo);
    CHECK(promo.size() == 1 && promo[0] == 1);
}

int main() {
    test_mixed_split();
    test_all_missing();
    test_all_staged();
    test_staged_dedup();
    test_stats();
    test_exceptions();
    test_reset();
    test_boundaries();
    if (g_failures) {
        fprintf(stderr, "q4e-graph-map: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-graph-map: all %d checks passed\n", g_checks);
    return 0;
}
