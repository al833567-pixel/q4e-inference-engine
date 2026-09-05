// Unit tests for the PLE ngram table (common/q4e-ngram.{h,cpp}).
// Vectors ported 1:1 from 推理工具/tests/test_ngram.py (spec oracle, green in Python).
// Cross-language oracles (FNV, row ids, RNG sequences) generated from ngram.py / random.Random.
#include "q4e-ngram.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
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

#define CHECK_EQ_F(a, b) \
    do { \
        ++g_checks; \
        const float va = (a); \
        const float vb = (b); \
        if (va != vb) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s == %g, expected %g\n", __FILE__, __LINE__, #a, (double) va, (double) vb); \
        } \
    } while (0)

#define CHECK_THROWS(expr) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            (void) (expr); \
        } catch (const std::exception &) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

static void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t)v); out.push_back((uint8_t)(v >> 8));
    out.push_back((uint8_t)(v >> 16)); out.push_back((uint8_t)(v >> 24));
}

static void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        out.push_back((uint8_t)(v >> (8 * i)));
    }
}

// byte-identical to tests/test_ngram.py write_fake_gguf (n_rows=128, row_len=160, d=1.0, quants=0)
static void write_fake_gguf(const std::string & path) {
    std::vector<uint8_t> out;
    out.insert(out.end(), (const uint8_t *)"GGUF", (const uint8_t *)"GGUF" + 4);
    put_u32(out, 3);
    put_u64(out, 1); // tensor_count
    put_u64(out, 2); // kv_count
    const char * keys[2] = { "general.architecture", "general.name" };
    const char * vals[2] = { "qwen3_8_next", "fake-45bit" };
    for (int i = 0; i < 2; i++) {
        put_u64(out, (uint64_t)strlen(keys[i]));
        out.insert(out.end(), keys[i], keys[i] + strlen(keys[i]));
        put_u32(out, 8); // string
        put_u64(out, (uint64_t)strlen(vals[i]));
        out.insert(out.end(), vals[i], vals[i] + strlen(vals[i]));
    }
    const char * name = "per_layer_token_embd.weight";
    put_u64(out, (uint64_t)strlen(name));
    out.insert(out.end(), name, name + strlen(name));
    put_u32(out, 2);
    put_u64(out, 160);
    put_u64(out, 128);
    put_u32(out, 20); // gtype (IQ4_NL, as in the real file)
    put_u64(out, 0); // offset is relative to the data section start
    // GGUF spec: data section starts at the alignment-padded position after metadata
    while (out.size() % 32 != 0) {
        out.push_back(0);
    }
    uint8_t block[18];
    block[0] = 0x00; block[1] = 0x3C; // d = 1.0 f16 LE
    for (int i = 2; i < 18; i++) {
        block[i] = 0;
    }
    for (int r = 0; r < 128; r++) {
        for (int b = 0; b < 5; b++) {
            out.insert(out.end(), block, block + 18);
        }
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
}

// test_decode_reference
static void test_decode_reference() {
    uint8_t block[18];
    block[0] = 0x00; block[1] = 0x40; // d = 2.0 f16 LE
    for (int i = 2; i < 18; i++) {
        block[i] = 0;
    }
    block[2] = 0x00;
    block[3] = 0xFF;
    block[4] = 0x53;
    float v[32];
    q4e_ngram_decode_block_45(block, 18, v);
    CHECK_EQ_F(v[0], (0 - 8) * 2.0f);
    CHECK_EQ_F(v[1], (0 - 8) * 2.0f);
    CHECK_EQ_F(v[2], (15 - 8) * 2.0f);
    CHECK_EQ_F(v[3], (15 - 8) * 2.0f);
    CHECK_EQ_F(v[4], (3 - 8) * 2.0f);
    CHECK_EQ_F(v[5], (5 - 8) * 2.0f);
    for (int i = 6; i < 32; i++) {
        CHECK_EQ_F(v[i], (-8.0f) * 2.0f);
    }
}

// test_decode_d1_range
static void test_decode_d1_range() {
    uint8_t block[18];
    block[0] = 0x00; block[1] = 0x3C; // d = 1.0
    for (int i = 2; i < 18; i++) {
        block[i] = 0;
    }
    float v[32];
    q4e_ngram_decode_block_45(block, 18, v);
    for (int i = 0; i < 32; i++) {
        CHECK_EQ_F(v[i], -8.0f);
    }
    for (int i = 2; i < 18; i++) {
        block[i] = 0xFF;
    }
    q4e_ngram_decode_block_45(block, 18, v);
    for (int i = 0; i < 32; i++) {
        CHECK_EQ_F(v[i], 7.0f);
    }
}

// test_decode_short_raises
static void test_decode_short_raises() {
    uint8_t block[17] = {0};
    float v[32];
    CHECK_THROWS(q4e_ngram_decode_block_45(block, 17, v));
}

// test_fnv_vectors (python oracle)
static void test_fnv_vectors() {
    CHECK(q4e_ngram_fnv1a64(nullptr, 0) == 0xCBF29CE484222325ull);
    const uint8_t b4[4] = {1, 2, 3, 4};
    CHECK(q4e_ngram_fnv1a64(b4, 4) == 0xBE7A5E775165785Dull);
    uint8_t b32[32];
    for (int i = 0; i < 32; i++) {
        b32[i] = (uint8_t)i;
    }
    CHECK(q4e_ngram_fnv1a64(b32, 32) == 0xE6CB594C1A148AC5ull);
    uint8_t bff[16];
    for (int i = 0; i < 16; i++) {
        bff[i] = 0xFF;
    }
    CHECK(q4e_ngram_fnv1a64(bff, 16) == 0xD6607508F5A1E855ull);
}

// test_parse_synthetic
static void test_parse_synthetic(const std::string & path) {
    q4e_ngram_table t;
    t.parse(path);
    CHECK_EQ_I64(t.row_len(), 160);
    CHECK_EQ_I64(t.n_rows(), 128);
    CHECK_EQ_I64(t.n_entries(), 8);
    CHECK_EQ_I64(t.row_bytes(), 90);
    CHECK_EQ_I64(t.size(), 128 * 90);
    CHECK_EQ_I64(t.gtype(), 20);
    t.close();
}

// test_row_ids_oracle (python oracle, n_entries=8)
static void test_row_ids_oracle(const std::string & path) {
    q4e_ngram_table t;
    t.parse(path);
    const auto r1 = t.row_ids_for_token(30, 20, 10);
    const auto r2 = t.row_ids_for_token(30, 20, 10);
    CHECK(r1 == r2);
    CHECK_EQ_I64((int64_t)r1.size(), 16);
    const int64_t expect[16] = {112, 97, 82, 67, 52, 37, 22, 7, 80, 65, 114, 99, 20, 5, 54, 39};
    for (int i = 0; i < 16; i++) {
        CHECK_EQ_I64(r1[(size_t)i], expect[i]);
    }
    for (int64_t x : r1) {
        CHECK(x >= 0 && x < 128);
    }
    const auto r3 = t.row_ids_for_token(99, 20, 10);
    CHECK(r3 != r1);
    const int64_t expect1[16] = {112, 97, 82, 67, 52, 37, 22, 7, 112, 97, 82, 67, 52, 37, 22, 7};
    const auto r4 = t.row_ids_for_token(42, 0, 0);
    for (int i = 0; i < 16; i++) {
        CHECK_EQ_I64(r4[(size_t)i], expect1[i]);
    }
    const int64_t expect9[16] = {112, 97, 18, 3, 52, 37, 86, 71, 64, 81, 34, 51, 4, 21, 102, 119};
    const auto r5 = t.row_ids_for_token(999, 8, 7);
    for (int i = 0; i < 16; i++) {
        CHECK_EQ_I64(r5[(size_t)i], expect9[i]);
    }
    t.close();
}

// test_read_row_values
static void test_read_row_values(const std::string & path) {
    q4e_ngram_table t;
    t.parse(path);
    const auto row = t.read_row(0);
    CHECK_EQ_I64((int64_t)row.size(), 160);
    for (float x : row) {
        CHECK_EQ_F(x, -8.0f);
    }
    CHECK_THROWS(t.read_row(128));
    CHECK_THROWS(t.read_row(-1));
    t.close();
}

// test_lookup
static void test_lookup(const std::string & path) {
    q4e_ngram_table t;
    t.parse(path);
    const std::vector<int64_t> ids = {10, 20, 30};
    const auto vals = t.lookup(ids);
    CHECK_EQ_I64((int64_t)vals.size(), 16 * 160);
    for (float x : vals) {
        CHECK_EQ_F(x, -8.0f);
    }
    const auto vals1 = t.lookup({42}); // single token -> 0 padding
    CHECK_EQ_I64((int64_t)vals1.size(), 16 * 160);
    CHECK_THROWS(t.lookup({}));
    t.close();
}

// test_read_sequence_oracle (python random.Random parity, n_rows=128)
static void test_read_sequence_oracle(const std::string & path) {
    q4e_ngram_table t;
    t.parse(path);
    const int64_t s30[30] = {124, 7, 99, 110, 0, 114, 68, 58, 26, 81, 7, 5, 6, 2, 97, 55, 108, 7, 56, 112, 126, 59, 88, 59, 56, 117, 74, 5, 106, 25};
    const auto seq1 = t.read_sequence(30, 1);
    CHECK_EQ_I64((int64_t)seq1.size(), 30);
    for (int i = 0; i < 30; i++) {
        CHECK_EQ_I64(seq1[(size_t)i], s30[i]);
    }
    CHECK(t.read_sequence(30, 1) == seq1); // deterministic
    const int64_t s10[10] = {35, 72, 35, 24, 64, 37, 79, 25, 18, 84};
    const auto seq2 = t.read_sequence(10, 0);
    for (int i = 0; i < 10; i++) {
        CHECK_EQ_I64(seq2[(size_t)i], s10[i]);
    }
    const int64_t s5[5] = {7, 23, 55, 59, 6};
    const auto seq3 = t.read_sequence(5, 42);
    for (int i = 0; i < 5; i++) {
        CHECK_EQ_I64(seq3[(size_t)i], s5[i]);
    }
    CHECK_THROWS(t.read_sequence(0, 1));
    t.close();
}

// test_bench_synthetic
static void test_bench_synthetic(const std::string & path) {
    q4e_ngram_table t;
    t.parse(path);
    const auto r = t.bench_random_reads(30, 1);
    CHECK_EQ_I64(r.n, 30);
    CHECK_EQ_I64(r.bytes_per_read, 90);
    CHECK(r.p99_ms >= r.p50_ms);
    CHECK(r.p50_ms >= r.min_ms);
    CHECK(r.mean_ms >= 0.0); // mmap page-cache reads can round to 0.0000 at 4-decimal ms
    CHECK(!r.mode.empty());
    CHECK_THROWS(t.bench_random_reads(0, 1));
    t.close();
}

// test_parse_missing_tensor
static void test_parse_missing_tensor(const std::string & path) {
    std::vector<uint8_t> out;
    out.insert(out.end(), (const uint8_t *)"GGUF", (const uint8_t *)"GGUF" + 4);
    put_u32(out, 3);
    put_u64(out, 0); // tensor_count
    put_u64(out, 0); // kv_count
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        exit(1);
    }
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    CHECK_THROWS(q4e_ngram_table().parse(path));
}

// test_offload_prefetch
static void test_offload_prefetch(const std::string & path) {
    q4e_ngram_table t;
    t.parse(path);
    t.offload("D:\\ng_offload");
    CHECK(t.offload_path() == "D:\\ng_offload");
    t.prefetch({1, 2, 3}); // sync reads, no throw
    t.close();
}

// test_real_gguf (skip if dir missing)
static void test_real_gguf(const std::string & dir) {
    q4e_ngram_table t;
    t.parse(dir);
    CHECK_EQ_I64(t.row_len(), 160);
    CHECK_EQ_I64(t.n_rows(), 320001536);
    CHECK_EQ_I64(t.size(), 28800138240);
    CHECK_EQ_I64(t.n_entries(), 20000096);
    CHECK_EQ_I64(t.row_bytes(), 90);
    const float ratio = (double)t.size() * 8.0 / ((double)t.row_len() * (double)t.n_rows());
    CHECK(std::fabs(ratio - 4.5) < 1e-6);
    const auto row = t.read_row(0);
    CHECK_EQ_I64((int64_t)row.size(), 160);
    for (float x : row) {
        CHECK(std::isfinite((double)x));
    }
    const auto r = t.bench_random_reads(32, 0);
    CHECK_EQ_I64(r.n, 32);
    CHECK(r.mean_ms >= 0.0);
    t.close();
}

int main(int argc, char ** argv) {
    // --bench <dir> [n] [seed]: live bench runner (evidence for the A-4 gate)
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0 && argc >= 3) {
        const std::string dir = argv[2];
        const int64_t n = argc >= 4 ? atoll(argv[3]) : 1000;
        const int64_t seed = argc >= 5 ? atoll(argv[4]) : 0;
        q4e_ngram_table t;
        t.parse(dir);
        printf("== parse ==\n");
        printf("file        : %s\n", t.path().c_str());
        printf("tensor      : %s\n", Q4E_NGRAM_TENSOR);
        printf("dims        : %lld x %lld (row_len x n_rows)\n", (long long) t.row_len(), (long long) t.n_rows());
        printf("size        : %lld bytes\n", (long long) t.size());
        printf("bits/elt    : %.4f (expect 4.5)\n", (double) t.size() * 8.0 / ((double) t.row_len() * (double) t.n_rows()));
        printf("n_entries   : %lld (= n_rows/16)\n", (long long) t.n_entries());
        printf("row_bytes   : %lld B (160 vals x 4.5 bit)\n", (long long) t.row_bytes());
        const auto r = t.bench_random_reads(n, seed);
        printf("== bench_random_reads(n=%lld, seed=%lld) ==\n", (long long) n, (long long) seed);
        printf("{\n");
        printf("  \"n\": %lld,\n", (long long) r.n);
        printf("  \"bytes_per_read\": %lld,\n", (long long) r.bytes_per_read);
        printf("  \"mean_ms\": %.4f,\n", r.mean_ms);
        printf("  \"p50_ms\": %.4f,\n", r.p50_ms);
        printf("  \"p99_ms\": %.4f,\n", r.p99_ms);
        printf("  \"min_ms\": %.4f,\n", r.min_ms);
        printf("  \"max_ms\": %.4f,\n", r.max_ms);
        printf("  \"throughput_mbps\": %.2f,\n", r.throughput_mbps);
        printf("  \"mode\": \"%s\"\n", r.mode.c_str());
        printf("}\n");
        t.close();
        return 0;
    }

    // unit tests: fake GGUF in a repo-local temp dir
    const char * tmpdir = getenv("Q4E_NGRAM_TMPDIR");
    std::string dir = tmpdir ? tmpdir : "_testtmp";
    if (dir.empty()) {
        dir = "_testtmp";
    }
    std::string fake = dir + "\\fake.gguf";
    std::string empty = dir + "\\empty.gguf";

    std::filesystem::create_directories(dir);
    write_fake_gguf(fake);

    test_decode_reference();
    test_decode_d1_range();
    test_decode_short_raises();
    test_fnv_vectors();
    test_parse_synthetic(fake);
    test_row_ids_oracle(fake);
    test_read_row_values(fake);
    test_lookup(fake);
    test_read_sequence_oracle(fake);
    test_bench_synthetic(fake);
    test_parse_missing_tensor(empty);
    test_offload_prefetch(fake);

    // real GGUF (skip if dir missing)
    const char * real = getenv("Q4E_NGRAM_REAL_DIR");
    if (real && *real) {
        test_real_gguf(real);
    }

    if (g_failures) {
        fprintf(stderr, "q4e-ngram: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-ngram: all %d checks passed\n", g_checks);
    return 0;
}
