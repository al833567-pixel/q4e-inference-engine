// Unit tests for the tiered MoE expert cache (common/q4e-expert-cache.{h,cpp}).
// Vectors frozen in this C++ test (v3 provenance: 推理工具/tests/test_cache.py, removed v4; oracle green in both).
#include "q4e-expert-cache.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
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

#define CHECK_EQ_STR(a, b) \
    do { \
        ++g_checks; \
        const char * va = (a); \
        const char * vb = (b); \
        if (va == nullptr || vb == nullptr || strcmp(va, vb) != 0) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s == \"%s\", expected \"%s\"\n", __FILE__, __LINE__, #a, va ? va : "(null)", vb); \
        } \
    } while (0)

#define CHECK_THROWS(expr) \
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

static q4e_expert_cache_config make_cfg(int vram = 3, int ram = 5, int n_layers = 2, bool has_shared = false) {
    q4e_expert_cache_config c;
    c.vram_capacity_per_layer = vram;
    c.ram_capacity_per_layer  = ram;
    c.n_layers                = n_layers;
    c.has_shared              = has_shared;
    return c;
}

// test_first_touch_is_ssd_then_vram_hit
static void test_first_touch_is_ssd_then_vram_hit() {
    q4e_expert_cache c(make_cfg(3, 5));
    const int ids[] = {1, 2, 3};
    q4e_expert_hits b1 = c.touch(0, ids, 3);
    CHECK_EQ_I64(b1.vram, 0);
    CHECK_EQ_I64(b1.ram, 0);
    CHECK_EQ_I64(b1.ssd, 3);
    const int one[] = {1};
    q4e_expert_hits b2 = c.touch(0, one, 1);
    CHECK_EQ_I64(b2.vram, 1);
    CHECK_EQ_I64(b2.ram, 0);
    CHECK_EQ_I64(b2.ssd, 0);
}

// test_vram_lru_evicts_to_ram
static void test_vram_lru_evicts_to_ram() {
    q4e_expert_cache c(make_cfg(2, 4));
    const int i1[] = {1};
    const int i2[] = {2};
    const int i3[] = {3};
    c.touch(0, i1, 1); // vram: [1]
    c.touch(0, i2, 1); // vram: [1,2]
    c.touch(0, i3, 1); // full -> evict 1 to ram; vram: [2,3]
    CHECK_EQ_STR(c.tier_of(0, 1), "ram");
    CHECK_EQ_STR(c.tier_of(0, 2), "vram");
    CHECK_EQ_STR(c.tier_of(0, 3), "vram");
    q4e_expert_hits b = c.touch(0, i1, 1); // ram hit -> promote -> evict 2 (LRU)
    CHECK_EQ_I64(b.vram, 0);
    CHECK_EQ_I64(b.ram, 1);
    CHECK_EQ_I64(b.ssd, 0);
    CHECK_EQ_STR(c.tier_of(0, 1), "vram");
    CHECK_EQ_STR(c.tier_of(0, 2), "ram");
}

// test_ram_lru_evicts_to_ssd
static void test_ram_lru_evicts_to_ssd() {
    q4e_expert_cache c(make_cfg(1, 2));
    const int i1[] = {1};
    const int i2[] = {2};
    const int i3[] = {3};
    const int i4[] = {4};
    c.touch(0, i1, 1); // vram: [1]
    c.touch(0, i2, 1); // evict 1->ram; vram: [2], ram: [1]
    c.touch(0, i3, 1); // evict 2->ram[1,2]; vram: [3]
    c.touch(0, i4, 1); // evict 3->ram full -> evict 1->ssd; ram: [2,3], vram: [4]
    CHECK_EQ_STR(c.tier_of(0, 1), "ssd");
    CHECK_EQ_STR(c.tier_of(0, 2), "ram");
    CHECK_EQ_STR(c.tier_of(0, 3), "ram");
    CHECK_EQ_STR(c.tier_of(0, 4), "vram");
    q4e_expert_hits b = c.touch(0, i1, 1); // re-touch -> ssd hit
    CHECK_EQ_I64(b.ssd, 1);
}

// test_lru_order_respects_reuse
static void test_lru_order_respects_reuse() {
    q4e_expert_cache c(make_cfg(2, 4));
    const int ids_12[] = {1, 2};
    const int i1[] = {1};
    const int i3[] = {3};
    c.touch(0, ids_12, 2); // vram LRU->MRU: 1, 2
    c.touch(0, i1, 1);     // 1 becomes MRU: 2, 1
    c.touch(0, i3, 1);     // evict 2 (least recently used), not 1
    CHECK_EQ_STR(c.tier_of(0, 2), "ram");
    CHECK_EQ_STR(c.tier_of(0, 1), "vram");
    CHECK_EQ_STR(c.tier_of(0, 3), "vram");
}

// test_prefetch_goes_only_to_ram
static void test_prefetch_goes_only_to_ram() {
    q4e_expert_cache c(make_cfg(2, 4));
    const int hints[] = {10, 11};
    c.prefetch(0, hints, 2);
    CHECK_EQ_STR(c.tier_of(0, 10), "ram");
    CHECK_EQ_STR(c.tier_of(0, 11), "ram");
    CHECK_EQ_I64(c.stats().prefetch.issued, 2);
}

// test_prefetch_hit_counted_on_promotion
static void test_prefetch_hit_counted_on_promotion() {
    q4e_expert_cache c(make_cfg(2, 4));
    const int hint[] = {10};
    const int id[] = {10};
    c.prefetch(0, hint, 1);
    q4e_expert_hits b = c.touch(0, id, 1); // ram hit (and prefetched)
    CHECK_EQ_I64(b.ram, 1);
    CHECK_EQ_I64(c.stats().prefetch.hits, 1);
}

// test_prefetch_skipped_when_hot
static void test_prefetch_skipped_when_hot() {
    q4e_expert_cache c(make_cfg(2, 4));
    const int i5[] = {5};
    const int i6[] = {6};
    c.touch(0, i5, 1);
    c.prefetch(0, i5, 1);
    CHECK_EQ_I64(c.stats().prefetch.skipped_in_vram, 1);
    CHECK_EQ_I64(c.stats().prefetch.issued, 0);
    c.prefetch(0, i6, 1); // into ram
    c.prefetch(0, i6, 1); // already staged
    CHECK_EQ_I64(c.stats().prefetch.already_staged, 1);
}

// test_shared_resident_and_pinned
static void test_shared_resident_and_pinned() {
    q4e_expert_cache c(make_cfg(1, 1, 2, true));
    const int shared[] = {Q4E_EXPERT_SHARED_ID};
    q4e_expert_hits b = c.touch(0, shared, 1);
    CHECK_EQ_I64(b.vram, 1);
    for (int i = 0; i < 50; ++i) {
        int id = i;
        int hint = 100 + i;
        c.touch(0, &id, 1);
        c.prefetch(0, &hint, 1);
    }
    CHECK_EQ_STR(c.tier_of(0, Q4E_EXPERT_SHARED_ID), "vram");
    b = c.touch(0, shared, 1);
    CHECK_EQ_I64(b.vram, 1);
}

struct rng64 {
    uint64_t s = 42;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

// deterministic: same input sequence => identical stats (two independent runs)
static void test_deterministic_replay() {
    auto run = []() {
        q4e_expert_cache c(make_cfg(4, 8, 3));
        rng64 rng;
        std::vector<int> pool(64);
        for (int i = 0; i < 64; ++i) pool[i] = i;
        for (int t = 0; t < 300; ++t) {
            for (int l = 0; l < 3; ++l) {
                std::vector<int> ids;
                for (int k = 0; k < 6; ++k) {
                    const int pick = (int) (rng.next() % (uint64_t) (64 - k)) + k;
                    std::swap(pool[k], pool[pick]);
                    ids.push_back(pool[k]);
                }
                std::sort(ids.begin(), ids.end());
                c.touch(l, ids.data(), (int64_t) ids.size());
                std::vector<int> hints;
                for (int k = 0; k < 4; ++k) {
                    const int pick = (int) (rng.next() % (uint64_t) (64 - k)) + k;
                    std::swap(pool[k], pool[pick]);
                    hints.push_back(pool[k]);
                }
                std::sort(hints.begin(), hints.end());
                c.prefetch(l, hints.data(), (int64_t) hints.size());
            }
        }
        return c.stats();
    };
    const q4e_expert_cache_stats a = run();
    const q4e_expert_cache_stats b = run();
    CHECK_EQ_I64(a.touches, b.touches);
    CHECK_EQ_I64(a.hits.vram, b.hits.vram);
    CHECK_EQ_I64(a.hits.ram, b.hits.ram);
    CHECK_EQ_I64(a.hits.ssd, b.hits.ssd);
    CHECK_EQ_I64(a.prefetch.issued, b.prefetch.issued);
    CHECK_EQ_I64(a.prefetch.hits, b.prefetch.hits);
    CHECK_EQ_I64(a.placement_events, b.placement_events);
    CHECK(a.touches > 0);
}

static q4e_expert_cache_stats run_workload(int hot, int tokens = 1000, int n_layers = 48) {
    q4e_expert_cache_config cfg;
    cfg.n_layers = n_layers;
    cfg.n_experts_per_layer = 512;
    cfg.vram_capacity_per_layer = 20;
    cfg.ram_capacity_per_layer  = 142;
    cfg.has_shared = true;
    q4e_expert_cache c(cfg);
    for (int t = 0; t < tokens; ++t) {
        for (int l = 0; l < n_layers; ++l) {
            std::vector<int> ids;
            for (int i = 0; i < 10; ++i) {
                ids.push_back((l * 31 + t * 7 + i * 13) % hot);
            }
            ids.push_back(Q4E_EXPERT_SHARED_ID);
            c.touch(l, ids.data(), (int64_t) ids.size());
        }
    }
    return c.stats();
}

// test_workload_hot_set_fits_vram
static void test_workload_hot_set_fits_vram() {
    const q4e_expert_cache_stats s = run_workload(15);
    CHECK_EQ_I64(s.touches, 1000 * 48 * 11);
    const double sum = s.hit_rate_vram + s.hit_rate_ram + s.hit_rate_ssd;
    CHECK(std::fabs(sum - 1.0) < 1e-6);
    CHECK(s.hit_rate_ssd < 0.05);
    CHECK(s.hit_rate_vram > 0.9);
}

// test_workload_needs_ram_tier
static void test_workload_needs_ram_tier() {
    const q4e_expert_cache_stats s = run_workload(30);
    CHECK(s.hit_rate_ssd < 0.05);
    CHECK(s.hit_rate_vram + s.hit_rate_ram > 0.95);
    CHECK(s.hit_rate_ram > 0.4);
}

// test_stats_and_placement_log
static void test_stats_and_placement_log() {
    q4e_expert_cache c(make_cfg(2, 3));
    const int ids[] = {1, 2, 3, 4};
    const int hint[] = {9};
    c.touch(0, ids, 4);
    c.prefetch(0, hint, 1);
    const q4e_expert_cache_stats s = c.stats();
    CHECK_EQ_I64(s.touches, 4);
    CHECK_EQ_I64(s.hits.vram, 0);
    CHECK_EQ_I64(s.hits.ram, 0);
    CHECK_EQ_I64(s.hits.ssd, 4);
    CHECK(s.placement_events > 0);
    for (const auto & e : c.placement_log()) {
        CHECK(e.layer >= 0 && e.layer < 2);
        CHECK(e.tier >= 0 && e.tier <= 2);
    }
}

// test_range_checks
static void test_range_checks() {
    q4e_expert_cache c(make_cfg());
    const int ids[] = {1};
    const int big[] = {99999};
    CHECK_THROWS(c.touch(99, ids, 1));
    CHECK_THROWS(c.touch(0, big, 1));
    CHECK_THROWS(c.prefetch(99, ids, 1));
}

// test_reset
static void test_reset() {
    q4e_expert_cache c(make_cfg());
    const int ids[] = {1, 2};
    const int hint[] = {3};
    c.touch(0, ids, 2);
    c.prefetch(0, hint, 1);
    c.reset();
    const q4e_expert_cache_stats s = c.stats();
    CHECK_EQ_I64(s.touches, 0);
    CHECK_EQ_I64(s.hits.vram, 0);
    CHECK_EQ_I64(s.hits.ram, 0);
    CHECK_EQ_I64(s.hits.ssd, 0);
    CHECK_EQ_STR(c.tier_of(0, 1), "ssd");
}

struct recording_backend : q4e_expert_cache_backend {
    std::set<std::string> kinds;
    void place(int, int, const char *) override { kinds.insert("place"); }
    void evict(int, int, const char *) override { kinds.insert("evict"); }
};

// test_subclass_override_place_evict
static void test_subclass_override_place_evict() {
    recording_backend be;
    q4e_expert_cache c(make_cfg(2, 2), &be);
    const int ids[] = {1, 2, 3};
    c.touch(0, ids, 3);
    CHECK(be.kinds.count("place") == 1);
    CHECK(be.kinds.count("evict") == 1);
}

// layer name parsing (hook: real router touch)
static void test_layer_from_name() {
    CHECK_EQ_I64(q4e_expert_layer_from_name("blk.12.ffn_up_exps.weight", 48), 12);
    CHECK_EQ_I64(q4e_expert_layer_from_name("blk.0.ffn_up_exps.weight", 48), 0);
    CHECK_EQ_I64(q4e_expert_layer_from_name("blk.47.ffn_gate_exps.weight", 48), 47);
    CHECK_EQ_I64(q4e_expert_layer_from_name("output.weight", 48), -1);
    CHECK_EQ_I64(q4e_expert_layer_from_name("blk.48.ffn_up_exps.weight", 48), -1);
    CHECK_EQ_I64(q4e_expert_layer_from_name("blk.x.weight", 48), -1);
    CHECK_EQ_I64(q4e_expert_layer_from_name(nullptr, 48), -1);
}

// test_occupancy_bounded_by_capacity: evict-then-insert keeps each tier within capacity
static void test_occupancy_bounded_by_capacity() {
    q4e_expert_cache_config cfg;
    cfg.n_layers                = 2;
    cfg.n_experts_per_layer     = 16;
    cfg.vram_capacity_per_layer = 4;
    cfg.ram_capacity_per_layer  = 6;
    cfg.has_shared              = false;
    q4e_expert_cache c(cfg);
    for (int l = 0; l < 2; ++l) {
        for (int e = 0; e < 16; ++e) { // more distinct experts than any capacity -> force eviction
            const int id = e;
            c.touch(l, &id, 1);
        }
    }
    const q4e_expert_cache_stats s = c.stats();
    CHECK(s.occ_vram.max <= cfg.vram_capacity_per_layer);
    CHECK(s.occ_ram.max  <= cfg.ram_capacity_per_layer);
    CHECK(s.occ_vram.min >= 0);
    CHECK(s.occ_ram.min  >= 0);
}

// test_cb_eval_rejects_non_moe: hook filter rejects non-MoE tensors (no touch)
static void test_cb_eval_rejects_non_moe() {
    q4e_expert_cache c(make_cfg(4, 6, 2, false));
    const int64_t before = c.stats().touches;

    // case A: op is not MUL_MAT_ID -> filter rejects
    {
        struct ggml_tensor src0 = {};
        struct ggml_tensor t = {};
        std::snprintf(src0.name, sizeof(src0.name), "blk.0.ffn_up_exps.weight");
        t.op     = GGML_OP_ADD; // not MUL_MAT_ID
        t.src[0] = &src0;
        CHECK(q4e_expert_cache_cb_eval(&t, /*ask=*/false, &c));
        CHECK_EQ_I64(c.stats().touches, before);
    }

    // case B: MUL_MAT_ID but expert name has no "up" -> filter rejects
    {
        struct ggml_tensor src0 = {};
        struct ggml_tensor t = {};
        std::snprintf(src0.name, sizeof(src0.name), "blk.0.ffn_down_exps.weight");
        t.op     = GGML_OP_MUL_MAT_ID;
        t.src[0] = &src0;
        CHECK(q4e_expert_cache_cb_eval(&t, /*ask=*/false, &c));
        CHECK_EQ_I64(c.stats().touches, before);
    }
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;
    test_first_touch_is_ssd_then_vram_hit();
    test_vram_lru_evicts_to_ram();
    test_ram_lru_evicts_to_ssd();
    test_lru_order_respects_reuse();
    test_prefetch_goes_only_to_ram();
    test_prefetch_hit_counted_on_promotion();
    test_prefetch_skipped_when_hot();
    test_shared_resident_and_pinned();
    test_deterministic_replay();
    test_workload_hot_set_fits_vram();
    test_workload_needs_ram_tier();
    test_stats_and_placement_log();
    test_range_checks();
    test_reset();
    test_subclass_override_place_evict();
    test_layer_from_name();
    test_occupancy_bounded_by_capacity();
    test_cb_eval_rejects_non_moe();
    if (g_failures) {
        fprintf(stderr, "q4e-expert-cache: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-expert-cache: all %d checks passed\n", g_checks);
    return 0;
}
