// Unit tests for the v6 VRAM gather CPU-side contract (common/q4e-gather.h, A-16.2b).
// build_args offsets/bytes from the staging consume view, validate violations, stub launch.
#include "q4e-gather.h"

#include <cstddef>
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

#define CHECK_EQ_U64(a, b) \
    do { \
        ++g_checks; \
        const uint64_t va = (a); \
        const uint64_t vb = (b); \
        if (va != vb) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s == %llu, expected %llu\n", __FILE__, __LINE__, #a, (unsigned long long) va, (unsigned long long) vb); \
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

static q4e_gather_args_t make_valid(q4e_staging & st, uint64_t slot) {
    q4e_gather_args_t v;
    v.staging_base = &v;
    v.buf          = 0;
    v.n_experts    = 1;
    v.chunk_bytes  = slot;
    v.per_buf_bytes = st.per_buf_bytes();
    v.offsets      = { 0 };
    v.bytes        = { 16 };
    v.expert_ids   = { 0 };
    return v;
}

int main() {
    const uint64_t slot = 64 * 1024; // small slot for tests
    q4e_staging_config cfg;
    cfg.total_mb    = 4;
    cfg.chunk_bytes = slot;
    q4e_staging st(cfg);

    const int nplace = 3;
    const uint64_t placed_bytes[3] = { 1000, 2048, 4096 };
    st.begin_fill(0);
    for (int i = 0; i < nplace; ++i) {
        const int64_t off = st.place(0, placed_bytes[i]);
        if (off < 0) {
            ++g_failures;
            fprintf(stderr, "FAIL: place overflow in test setup\n");
            return 1;
        }
    }
    st.publish(0);

    // build_args: offsets follow the staging place order, bytes from the consume view
    const int ids[3] = { 17, 4, 100 };
    q4e_gather_args_t a;
    q4e_gather_build_args(st, 0, ids, 3, a);
    a.staging_base = &a;
    CHECK(a.buf == 0);
    CHECK(a.n_experts == 3);
    CHECK(a.chunk_bytes == slot);
    CHECK_EQ_U64(a.per_buf_bytes, st.per_buf_bytes());
    CHECK(a.offsets.size() == 3);
    CHECK_EQ_U64(a.offsets[0], 0);
    CHECK_EQ_U64(a.offsets[1], slot);
    CHECK_EQ_U64(a.offsets[2], 2 * slot);
    CHECK_EQ_U64(a.bytes[0], placed_bytes[0]);
    CHECK_EQ_U64(a.bytes[1], placed_bytes[1]);
    CHECK_EQ_U64(a.bytes[2], placed_bytes[2]);
    CHECK(a.expert_ids[0] == 17);
    CHECK(a.expert_ids[1] == 4);
    CHECK(a.expert_ids[2] == 100);
    q4e_gather_validate(a); // valid args pass
    CHECK(q4e_gather_launch_stub(a, nullptr) == false);
    CHECK(q4e_gather_launch_cuda(a, nullptr) == false);

    // build_args violations
    CHECK_THROWS_LE(q4e_gather_build_args(st, 1, ids, 1, a)); // buf 1 idle, not published
    CHECK_THROWS_IA(q4e_gather_build_args(st, 0, ids, 0, a)); // n == 0
    CHECK_THROWS_IA(q4e_gather_build_args(st, 0, ids, -1, a)); // n < 0
    CHECK_THROWS_IA(q4e_gather_build_args(st, 0, ids, 4, a)); // n > placed
    CHECK_THROWS_IA(q4e_gather_build_args(st, 0, nullptr, 1, a)); // null expert_ids
    CHECK_THROWS_IA(q4e_gather_build_args(st, 2, ids, 1, a)); // buf id above
    CHECK_THROWS_IA(q4e_gather_build_args(st, -1, ids, 1, a)); // buf id below

    // validate: valid base case passes
    q4e_gather_validate(make_valid(st, slot));

    // validate violations
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.staging_base = nullptr;
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.buf = 2;
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.buf = -1;
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.n_experts = 0;
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.chunk_bytes = 0;
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.per_buf_bytes = 0;
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.offsets.push_back(0); // size mismatch
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.offsets[0] = st.per_buf_bytes(); // offset out of capacity
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.offsets[0] = slot + 8; // not chunk aligned
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.bytes[0] = 0; // bytes out of slot
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }
    {
        q4e_gather_args_t v = make_valid(st, slot);
        v.bytes[0] = slot + 1; // bytes out of slot
        CHECK_THROWS_IA(q4e_gather_validate(v));
    }

    if (g_failures > 0) {
        fprintf(stderr, "test-q4e-gather: %d/%d checks failed\n", g_failures, g_checks);
        return 1;
    }
    printf("test-q4e-gather: all %d checks passed\n", g_checks);
    return 0;
}
