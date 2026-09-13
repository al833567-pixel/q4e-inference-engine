// Unit tests for the v6 staging buffer A/B double-buffer manager (common/q4e-staging.{h,cpp}, A-16.2a).
// fill -> publish -> consume -> release protocol, overflow events, stats consistency.
#include "q4e-staging.h"

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

#define CHECK_THROWS_CTOR(cfg) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            q4e_staging p(cfg); \
            (void) p; \
        } catch (const std::invalid_argument &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #cfg); \
        } \
    } while (0)

// test_ab_alternation: fill A, publish A, fill B, consume A, then the cycle repeats
static void test_ab_alternation() {
    q4e_staging s; // default 128 MB, slot 3216896 B
    const uint64_t slot = s.chunk_bytes();
    CHECK_EQ_I64(s.filling_buf(), -1);
    s.begin_fill(0);
    CHECK_EQ_I64(s.filling_buf(), 0);
    CHECK_EQ_I64(s.place(0, 1000000), 0);
    CHECK_EQ_I64(s.place(0, 2000000), (int64_t) slot);
    CHECK_EQ_I64(s.publish(0), 2);
    CHECK_EQ_I64(s.filling_buf(), -1);
    // fill B while A is published and consumed
    s.begin_fill(1);
    CHECK_EQ_I64(s.filling_buf(), 1);
    CHECK_EQ_I64(s.place(1, 1234567), 0);
    CHECK_EQ_I64(s.consume_begin(0), 2);
    q4e_staging_entry e0 = s.consume_at(0, 0);
    CHECK_EQ_I64(e0.offset, 0);
    CHECK_EQ_I64(e0.bytes, 1000000);
    q4e_staging_entry e1 = s.consume_at(0, 1);
    CHECK_EQ_I64(e1.offset, (int64_t) slot);
    CHECK_EQ_I64(e1.bytes, 2000000);
    s.release(0);
    CHECK(s.state_of(0) == q4e_staging::state::IDLE);
    s.publish(1);
    CHECK_EQ_I64(s.consume_begin(1), 1);
    CHECK_EQ_I64(s.consume_at(1, 0).bytes, 1234567);
    s.release(1);
    // the cycle repeats on A
    s.begin_fill(0);
    CHECK_EQ_I64(s.place(0, 999), 0);
    CHECK_EQ_I64(s.publish(0), 1);
    CHECK_EQ_I64(s.consume_begin(0), 1);
    CHECK_EQ_I64(s.consume_at(0, 0).bytes, 999);
    s.release(0);
    const q4e_staging_stats st = s.stats();
    CHECK_EQ_I64(st.fills, 3);
    CHECK_EQ_I64(st.publishes, 3);
    CHECK_EQ_I64(st.consumes, 3);
    CHECK_EQ_I64(st.overflows, 0);
}

// test_consume_before_publish: a filling or idle buffer is never consumable
static void test_consume_before_publish() {
    q4e_staging s;
    CHECK_THROWS_LE(s.consume_begin(0)); // idle
    CHECK_THROWS_LE(s.consume_at(0, 0)); // idle
    s.begin_fill(0);
    CHECK_EQ_I64(s.place(0, 1000), 0);
    CHECK_THROWS_LE(s.consume_begin(0)); // filling
    CHECK_THROWS_LE(s.release(0)); // not published yet
    s.publish(0);
    CHECK_THROWS_LE(s.publish(0)); // double publish
    s.release(0);
    CHECK_THROWS_LE(s.release(0)); // double release
}

// test_capacity_overflow: default config holds exactly 20 slots per buffer
static void test_capacity_overflow() {
    q4e_staging s;
    CHECK_EQ_I64(s.per_buf_bytes(), 128u * 1024u * 1024u / 2u);
    CHECK_EQ_I64(s.max_chunks_per_buf(), 20); // 67108864 / 3216896
    s.begin_fill(0);
    for (int i = 0; i < 20; ++i) {
        CHECK_EQ_I64(s.place(0, 3216896), (int64_t) i * 3216896);
    }
    CHECK_EQ_I64(s.place(0, 3216896), -1); // 21st chunk: buffer full
    CHECK_EQ_I64(s.place(0, 1), -1);
    CHECK_EQ_I64(s.stats().overflows, 2);
    CHECK_EQ_I64(s.placed_of(0), 20);
    CHECK_EQ_I64(s.publish(0), 20); // the 20 placed chunks are still consumable
    CHECK_EQ_I64(s.consume_at(0, 19).offset, (int64_t) 19 * 3216896);
}

// test_capacity_small: small config, 1 MB slots, 4 per buffer
static void test_capacity_small() {
    q4e_staging_config cfg;
    cfg.total_mb    = 8;
    cfg.chunk_bytes = 1048576;
    q4e_staging s(cfg);
    CHECK_EQ_I64(s.max_chunks_per_buf(), 4); // 4 MB / 1 MB
    s.begin_fill(1);
    CHECK_EQ_I64(s.place(1, 500000), 0);
    CHECK_EQ_I64(s.place(1, 500000), 1048576);
    CHECK_EQ_I64(s.place(1, 500000), 2 * 1048576);
    CHECK_EQ_I64(s.place(1, 500000), 3 * 1048576);
    CHECK_EQ_I64(s.place(1, 500000), -1);
    CHECK_EQ_I64(s.stats().overflows, 1);
}

// test_fill_rules: one filling buffer at a time; published buffer must be released first
static void test_fill_rules() {
    q4e_staging s;
    s.begin_fill(0);
    CHECK_THROWS_LE(s.begin_fill(0)); // same buffer already filling
    CHECK_THROWS_LE(s.begin_fill(1)); // another buffer is filling
    s.publish(0);
    CHECK_THROWS_LE(s.begin_fill(0)); // published, not released
    s.release(0);
    s.begin_fill(0);
    CHECK_EQ_I64(s.filling_buf(), 0);
    CHECK_EQ_I64(s.place(0, 100), 0);
    CHECK_EQ_I64(s.publish(0), 1);
    s.release(0);
    s.begin_fill(1);
    CHECK_EQ_I64(s.filling_buf(), 1);
    CHECK_EQ_I64(s.publish(1), 0); // publishing zero placements is legal
    s.release(1);
}

// test_place_validation: bytes bounds, slot alignment, index range
static void test_place_validation() {
    q4e_staging s;
    CHECK_THROWS_LE(s.place(0, 100)); // not filling
    s.begin_fill(0);
    CHECK_THROWS_IA(s.place(0, 0)); // zero bytes
    CHECK_THROWS_IA(s.place(0, s.chunk_bytes() + 1)); // larger than one slot
    CHECK_EQ_I64(s.place(0, s.chunk_bytes()), 0); // exactly one slot fits
    s.publish(0);
    CHECK_THROWS_OOR(s.consume_at(0, 1)); // only one placement
    CHECK_THROWS_LE(s.place(0, 100)); // published, no longer filling
}

// test_stats_consistency: counters match the executed protocol exactly
static void test_stats_consistency() {
    q4e_staging_config cfg;
    cfg.total_mb    = 8;
    cfg.chunk_bytes = 1048576; // 4 slots per buffer
    q4e_staging s(cfg);
    s.begin_fill(0);
    s.place(0, 1000);
    s.place(0, 2000);
    s.place(0, 3000);
    s.place(0, 4000);
    s.place(0, 5000); // overflow 1
    s.publish(0);
    s.consume_begin(0);
    s.release(0);
    s.begin_fill(1);
    s.place(1, 1000);
    s.place(1, 2000);
    s.place(1, 3000);
    s.place(1, 4000);
    s.place(1, 5000); // overflow 2
    s.place(1, 6000); // overflow 3
    s.publish(1);
    s.consume_begin(1);
    s.consume_begin(1); // re-consume the same batch
    s.release(1);
    const q4e_staging_stats st = s.stats();
    CHECK_EQ_I64(st.fills, 2);
    CHECK_EQ_I64(st.publishes, 2);
    CHECK_EQ_I64(st.consumes, 3);
    CHECK_EQ_I64(st.overflows, 3);
}

// test_reset: protocol state and counters cleared, manager still usable
static void test_reset() {
    q4e_staging s;
    s.begin_fill(0);
    CHECK_EQ_I64(s.place(0, 1000), 0);
    s.publish(0);
    CHECK_EQ_I64(s.consume_begin(0), 1);
    s.reset();
    CHECK_EQ_I64(s.filling_buf(), -1);
    CHECK(s.state_of(0) == q4e_staging::state::IDLE);
    CHECK(s.state_of(1) == q4e_staging::state::IDLE);
    CHECK_EQ_I64(s.placed_of(0), 0);
    const q4e_staging_stats st = s.stats();
    CHECK_EQ_I64(st.fills, 0);
    CHECK_EQ_I64(st.publishes, 0);
    CHECK_EQ_I64(st.consumes, 0);
    CHECK_EQ_I64(st.overflows, 0);
    CHECK_THROWS_LE(s.consume_begin(0)); // content is gone
    s.begin_fill(1);
    CHECK_EQ_I64(s.place(1, 2000), 0);
    CHECK_EQ_I64(s.publish(1), 1);
    CHECK_EQ_I64(s.stats().fills, 1);
}

// test_invalid_buf: out-of-range buf id rejected on every API
static void test_invalid_buf() {
    q4e_staging s;
    CHECK_THROWS_IA(s.begin_fill(2));
    CHECK_THROWS_IA(s.begin_fill(-1));
    CHECK_THROWS_IA(s.place(2, 100));
    CHECK_THROWS_IA(s.place(-1, 100));
    CHECK_THROWS_IA(s.publish(7));
    CHECK_THROWS_IA(s.publish(-3));
    CHECK_THROWS_IA(s.consume_begin(2));
    CHECK_THROWS_IA(s.consume_at(-1, 0));
    CHECK_THROWS_IA(s.release(3));
    CHECK_THROWS_IA(s.state_of(9));
    CHECK_THROWS_IA(s.placed_of(-2));
    CHECK_EQ_I64(s.filling_buf(), -1);
    s.begin_fill(0);
    CHECK_EQ_I64(s.place(0, 100), 0); // valid calls still work
}

// test_ctor_validation: bounds and the degenerate zero-slot buffer
static void test_ctor_validation() {
    q4e_staging def; // default 128 MB / 3216896 B slot is valid
    CHECK_EQ_I64(def.per_buf_bytes(), 64u * 1024u * 1024u);
    CHECK_EQ_I64(def.max_chunks_per_buf(), 20);
    q4e_staging_config c4;
    c4.total_mb = 4; // per buffer 2 MB < slot: zero slots, place always overflows
    q4e_staging s4(c4);
    CHECK_EQ_I64(s4.max_chunks_per_buf(), 0);
    s4.begin_fill(0);
    CHECK_EQ_I64(s4.place(0, 1000), -1);
    CHECK_EQ_I64(s4.stats().overflows, 1);
    q4e_staging_config bad1;
    bad1.total_mb = 3;
    CHECK_THROWS_CTOR(bad1);
    q4e_staging_config bad2;
    bad2.chunk_bytes = 0;
    CHECK_THROWS_CTOR(bad2);
}

int main() {
    test_ab_alternation();
    test_consume_before_publish();
    test_capacity_overflow();
    test_capacity_small();
    test_fill_rules();
    test_place_validation();
    test_stats_consistency();
    test_reset();
    test_invalid_buf();
    test_ctor_validation();
    if (g_failures) {
        fprintf(stderr, "q4e-staging: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-staging: all %d checks passed\n", g_checks);
    return 0;
}
