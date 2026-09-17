#ifndef NOMINMAX
#   define NOMINMAX
#endif

// Unit tests for the q4e v5 pool-path MoE FFN builder (A-15.2a static / A-15.2b dynamic).
// CPU oracle: pool path vs full expert tensor path (build_moe_ffn), f32, fixed seed.
#include "llama-graph.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "q4e-expert-warm.h"
// #include "q4e-pool-backend.h" // BISECT: disabled

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
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

// relative check with absolute floor: |a - b| <= tol * (1 + |a|)
#define CHECK_NEAR(a, b, tol) \
    do { \
        ++g_checks; \
        const double va = (a); \
        const double vb = (b); \
        if (!(std::fabs(va - vb) <= (double) (tol) * (1.0 + std::fabs(va)))) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s == %g, expected %g (tol %g)\n", __FILE__, __LINE__, #a, va, vb, (double) (tol)); \
        } \
    } while (0)

// one cur vector + the logits it must produce for all experts
struct token_group {
    std::vector<double> target; // [n_expert]
};

// per-token data + intended top-k, for the host-side oracle
struct case_data {
    int n_embd   = 64;
    int n_ff     = 128;
    int n_expert = 8;
    int k        = 3;
    int n_tokens = 4;

    std::vector<double> cur;      // [n_embd, n_tokens]
    std::vector<double> gate_inp; // [n_expert, n_embd]
    std::vector<double> up_w;     // [n_ff, n_embd, n_expert]
    std::vector<double> gate_w;   // [n_ff, n_embd, n_expert]
    std::vector<double> down_w;   // [n_embd, n_ff, n_expert]
    std::vector<std::vector<double>> target; // [n_tokens][n_expert]
    std::vector<std::vector<int>> sel;        // [n_tokens][k]
};

static void make_case(
      case_data & cd,
      const std::vector<token_group> & groups,
      const std::vector<int> & token_to_group) {
    cd.n_tokens = (int) token_to_group.size();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    auto rnd = [&]() { return dist(rng); };

    // one cur vector per group, made mutually orthogonal
    std::vector<std::vector<double>> v(groups.size(), std::vector<double>(cd.n_embd));
    for (size_t g = 0; g < groups.size(); ++g) {
        for (int j = 0; j < cd.n_embd; ++j) {
            v[g][j] = rnd();
        }
        for (size_t h = 0; h < g; ++h) {
            double dot = 0.0;
            double nrm = 0.0;
            for (int j = 0; j < cd.n_embd; ++j) {
                dot += v[g][j] * v[h][j];
                nrm += v[h][j] * v[h][j];
            }
            for (int j = 0; j < cd.n_embd; ++j) {
                v[g][j] -= (dot / nrm) * v[h][j];
            }
        }
    }

    // cur[t] = v of its group
    cd.cur.assign(cd.n_embd * cd.n_tokens, 0.0);
    for (int t = 0; t < cd.n_tokens; ++t) {
        const std::vector<double> & vg = v[(size_t) token_to_group[t]];
        for (int j = 0; j < cd.n_embd; ++j) {
            cd.cur[j + t * cd.n_embd] = vg[j];
        }
    }

    // gate_inp[e] . v[g] = target[g][e] for every group g (orthogonal basis)
    cd.gate_inp.assign(cd.n_expert * cd.n_embd, 0.0);
    for (size_t g = 0; g < groups.size(); ++g) {
        double nrm = 0.0;
        for (int j = 0; j < cd.n_embd; ++j) {
            nrm += v[g][j] * v[g][j];
        }
        for (int e = 0; e < cd.n_expert; ++e) {
            for (int j = 0; j < cd.n_embd; ++j) {
                cd.gate_inp[j + e * cd.n_embd] += groups[g].target[e] * v[g][j] / nrm;
            }
        }
    }

    // expert weights, random
    cd.up_w.assign((size_t) cd.n_ff * cd.n_embd * cd.n_expert, 0.0);
    cd.gate_w.assign((size_t) cd.n_ff * cd.n_embd * cd.n_expert, 0.0);
    cd.down_w.assign((size_t) cd.n_embd * cd.n_ff * cd.n_expert, 0.0);
    for (double & x : cd.up_w)   { x = rnd(); }
    for (double & x : cd.gate_w) { x = rnd(); }
    for (double & x : cd.down_w) { x = rnd(); }

    // intended top-k per token (logits are distinct, so the selection is unique)
    for (int t = 0; t < cd.n_tokens; ++t) {
        const std::vector<double> & l = groups[(size_t) token_to_group[t]].target;
        std::vector<std::pair<double, int>> order(cd.n_expert);
        for (int e = 0; e < cd.n_expert; ++e) {
            order[e] = {l[e], e};
        }
        std::sort(order.begin(), order.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        });
        cd.target.push_back(l);
        cd.sel.push_back({});
        for (int i = 0; i < cd.k; ++i) {
            cd.sel[t].push_back(order[i].second);
        }
    }
}

// host-side oracle: expert e output on cur_t, then silu(gate) * up, then down
static std::vector<double> host_expert_out(const case_data & cd, int e, int t) {
    std::vector<double> up_v(cd.n_ff);
    std::vector<double> h(cd.n_ff);
    for (int a = 0; a < cd.n_ff; ++a) {
        double su = 0.0;
        double sg = 0.0;
        // weight layout matches ggml tensor {n_embd, n_ff, n_expert}: embd fastest
        for (int j = 0; j < cd.n_embd; ++j) {
            su += cd.up_w[((size_t) e * cd.n_ff + a) * cd.n_embd + j] * cd.cur[j + t * cd.n_embd];
            sg += cd.gate_w[((size_t) e * cd.n_ff + a) * cd.n_embd + j] * cd.cur[j + t * cd.n_embd];
        }
        up_v[a] = su;
        h[a] = (sg / (1.0 + std::exp(-sg))) * su;
    }
    std::vector<double> out(cd.n_embd);
    for (int i = 0; i < cd.n_embd; ++i) {
        double s = 0.0;
        for (int b = 0; b < cd.n_ff; ++b) {
            s += cd.down_w[((size_t) e * cd.n_embd + i) * cd.n_ff + b] * h[b];
        }
        out[i] = s;
    }
    return out;
}

// expected token output; experts with id >= R contribute 0
static std::vector<double> host_token_out(const case_data & cd, int t, int R, float w_scale) {
    const std::vector<double> & l = cd.target[t];
    double m = -1e30;
    for (int e = 0; e < cd.n_expert; ++e) {
        m = std::max(m, l[e]);
    }
    std::vector<double> p(cd.n_expert);
    double s = 0.0;
    for (int e = 0; e < cd.n_expert; ++e) {
        p[e] = std::exp(l[e] - m);
        s += p[e];
    }
    double S = 0.0;
    for (int i = 0; i < cd.k; ++i) {
        S += p[cd.sel[t][i]];
    }
    std::vector<double> out(cd.n_embd, 0.0);
    for (int i = 0; i < cd.k; ++i) {
        const int e = cd.sel[t][i];
        if (e >= R) {
            continue;
        }
        const double w = p[e] / S * w_scale;
        const std::vector<double> eo = host_expert_out(cd, e, t);
        for (int j = 0; j < cd.n_embd; ++j) {
            out[j] += w * eo[j];
        }
    }
    return out;
}

struct moe_env {
    ggml_context * ctx_d = nullptr;
    llama_adapter_loras loras;
    llm_graph_result res{1024};
    llm_graph_params params{};
    std::unique_ptr<llm_graph_context> gctx;

    ggml_tensor * cur       = nullptr;
    ggml_tensor * gate_inp  = nullptr;
    ggml_tensor * up        = nullptr;
    ggml_tensor * gate      = nullptr;
    ggml_tensor * down      = nullptr;
    ggml_tensor * up_pool   = nullptr;
    ggml_tensor * gate_pool = nullptr;
    ggml_tensor * down_pool = nullptr;

    ~moe_env() {
        if (ctx_d) {
            ggml_free(ctx_d);
        }
    }
};

static void fill_tensor(ggml_tensor * t, const std::vector<double> & data) {
    CHECK_EQ_I64(ggml_nelements(t), (int64_t) data.size());
    float * p = (float *) ggml_get_data(t);
    for (size_t i = 0; i < data.size(); ++i) {
        p[i] = float(data[i]);
    }
}

static void env_setup(moe_env & env, const case_data & cd, int R) {
    // this ggml has fixed-size context pools (no growth): pre-size for 8 tensor objects + data
    const size_t mem_size = (size_t) ggml_tensor_overhead() * 8
        + (size_t) cd.n_ff * cd.n_embd * cd.n_expert * 4 * 3
        + (size_t) cd.n_embd * cd.n_tokens * 4
        + (size_t) cd.n_expert * cd.n_embd * 4
        + 64;
    ggml_init_params ip;
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    env.ctx_d = ggml_init(ip);

    // mul_mat contract: ne[0] is the reduction dim, so expert weights are [n_embd, n_out, n_expert]
    env.cur      = ggml_new_tensor_2d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_tokens);
    env.gate_inp = ggml_new_tensor_2d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_expert);
    env.up       = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_ff, cd.n_expert);
    env.gate     = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_ff, cd.n_expert);
    env.down     = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_ff, cd.n_embd, cd.n_expert);
    env.up_pool   = ggml_view_3d(env.ctx_d, env.up, cd.n_embd, cd.n_ff, R, env.up->nb[1], env.up->nb[2], 0);
    env.gate_pool = ggml_view_3d(env.ctx_d, env.gate, cd.n_embd, cd.n_ff, R, env.gate->nb[1], env.gate->nb[2], 0);
    env.down_pool = ggml_view_3d(env.ctx_d, env.down, cd.n_ff, cd.n_embd, R, env.down->nb[1], env.down->nb[2], 0);

    CHECK_EQ_I64(env.up_pool->ne[2], R);
    CHECK_EQ_I64(env.gate_pool->ne[2], R);
    CHECK_EQ_I64(env.down_pool->ne[2], R);

    fill_tensor(env.cur, cd.cur);
    fill_tensor(env.gate_inp, cd.gate_inp);
    fill_tensor(env.up, cd.up_w);
    fill_tensor(env.gate, cd.gate_w);
    fill_tensor(env.down, cd.down_w);

    env.params.arch = LLM_ARCH_QWEN4EXP;
    env.params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    env.params.n_outputs = (uint32_t) cd.n_tokens;
    env.params.hparams.n_embd = cd.n_embd;
    env.params.hparams.n_expert = cd.n_expert;
    env.params.hparams.n_expert_used = cd.k;
    env.params.hparams.n_layer_all = 2;
    env.params.hparams.swiglu_clamp_exp.fill(0.0f);
    env.params.ubatch.n_tokens = cd.n_tokens;
    env.params.res = &env.res;
    env.params.loras = &env.loras;

    env.gctx.reset(new llm_graph_context(env.params));
}

static std::vector<float> read_out(ggml_tensor * t) {
    std::vector<float> out((size_t) t->ne[0] * (size_t) t->ne[1]);
    const char * d = (const char *) ggml_get_data(t);
    for (int64_t j = 0; j < t->ne[1]; ++j) {
        for (int64_t i = 0; i < t->ne[0]; ++i) {
            out[(size_t) i + (size_t) j * t->ne[0]] = *(const float *) (d + i * t->nb[0] + j * t->nb[1]);
        }
    }
    return out;
}

// run both builders on the same data, then check against the host oracle
static void test_graph_case(
      const case_data & cd,
      int R,
      float w_scale,
      const char * name,
      int64_t expected_out_of_pool) {
    moe_env env;
    env_setup(env, cd, R);

    ggml_tensor * ref = env.gctx->build_moe_ffn(
            env.cur, env.gate_inp,
            env.up, env.gate, env.down,
            nullptr,
            cd.n_expert, cd.k,
            LLM_FFN_SILU, true, w_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, 0,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    CHECK(ref != nullptr);

    ggml_tensor * pool = env.gctx->build_moe_ffn_pool(
            env.cur, env.gate_inp,
            env.up_pool, env.gate_pool, env.down_pool,
            cd.n_expert, cd.k,
            LLM_FFN_SILU, true, w_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, 0);
    CHECK(pool != nullptr);

    // compute through the CPU backend scheduler (same path as production);
    // the compute context is no_alloc, so the backend owns the data buffers
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    ggml_backend_t backends[1] = {cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    CHECK(sched != nullptr);
    const ggml_status st = ggml_backend_sched_graph_compute(sched, env.res.get_gf());
    CHECK(st == GGML_STATUS_SUCCESS);

    // read outputs before freeing the scheduler (it owns the output buffers)
    std::vector<float> ref_out = read_out(ref);
    std::vector<float> pool_out = read_out(pool);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    // out-of-pool count via the static remap helper
    int64_t count = 0;
    for (int t = 0; t < cd.n_tokens; ++t) {
        count += q4e_pool_count_out_of_pool(cd.sel[t].data(), cd.k, R);
    }
    CHECK_EQ_I64(count, expected_out_of_pool);

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int t = 0; t < cd.n_tokens; ++t) {
        std::vector<double> expect = host_token_out(cd, t, R, w_scale);
        std::vector<double> expect_full = host_token_out(cd, t, cd.n_expert, w_scale);
        for (int i = 0; i < cd.n_embd; ++i) {
            const float gv = pool_out[(size_t) i + (size_t) t * cd.n_embd];
            const float rv = ref_out[(size_t) i + (size_t) t * cd.n_embd];
            const double ov = expect[(size_t) i];
            const double of = expect_full[(size_t) i];
            CHECK_NEAR(gv, ov, 1e-4);
            CHECK_NEAR(rv, of, 1e-4);
            const double d = std::fabs(gv - rv);
            max_abs = std::max(max_abs, d);
            if (std::fabs(rv) > 1e-6) {
                max_rel = std::max(max_rel, d / std::fabs(rv));
            }
            // pool and full must match only when no selected expert is out of pool
            if (expected_out_of_pool == 0) {
                CHECK(d <= 1e-4 * (1.0 + std::fabs(rv)));
            }
        }
    }
    fprintf(stderr, "%s: pool vs full: max_abs_err = %.3e, max_rel_err = %.3e, out_of_pool = %lld\n",
            name, max_abs, max_rel, (long long) expected_out_of_pool);
}

// static slot map + count helper
static void test_slot_of_helper() {
    CHECK_EQ_I64(q4e_pool_slot_of(0, 5), 0);
    CHECK_EQ_I64(q4e_pool_slot_of(3, 5), 3);
    CHECK_EQ_I64(q4e_pool_slot_of(4, 5), 4);
    CHECK_EQ_I64(q4e_pool_slot_of(5, 5), 0); // out-of-pool marker
    CHECK_EQ_I64(q4e_pool_slot_of(7, 5), 0);
    const int32_t ids[] = {0, 1, 2, 4, 5, 6, 7, 3}; // 5, 6, 7 out of pool
    CHECK_EQ_I64(q4e_pool_count_out_of_pool(ids, 8, 5), 3);
    CHECK_EQ_I64(q4e_pool_count_out_of_pool(ids, 8, 8), 0);
}

// all top-k experts in pool: pool path must match the full path (oracle gate)
static void test_all_in_pool_oracle() {
    case_data cd;
    std::vector<token_group> groups(1);
    groups[0].target = {4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5}; // top-3 = {0,1,2}, all < R = 5
    std::vector<int> token_to_group = {0, 0, 0, 0};
    make_case(cd, groups, token_to_group);
    test_graph_case(cd, 5, 2.5f, "all_in_pool", 0);
}

// two top-k experts out of pool: their contribution must be 0
static void test_out_of_pool_zeroed() {
    case_data cd;
    std::vector<token_group> groups(1);
    groups[0].target = {4.0, 0.5, 0.4, 0.3, 0.2, 3.5, 3.0, 0.1}; // top-3 = {0,5,6}, 5 and 6 >= R = 5
    std::vector<int> token_to_group = {0, 0, 0, 0};
    make_case(cd, groups, token_to_group);
    test_graph_case(cd, 5, 2.5f, "out_of_pool", 8); // 2 per token x 4 tokens
}

// per-token mix: some tokens all in pool, some with out-of-pool experts
static void test_mixed_per_token() {
    case_data cd;
    std::vector<token_group> groups(2);
    groups[0].target = {4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5}; // top-3 = {0,1,2}
    groups[1].target = {4.0, 0.5, 0.4, 0.3, 0.2, 3.5, 3.0, 0.1}; // top-3 = {0,5,6}
    std::vector<int> token_to_group = {0, 0, 1, 1};
    make_case(cd, groups, token_to_group);
    test_graph_case(cd, 5, 2.5f, "mixed_per_token", 4); // 2 per token x 2 tokens
}

#define CHECK_THROWS(expr) \
    do { \
        ++g_checks; \
        bool threw = false; \
        try { \
            (void) (expr); \
        } catch (...) { \
            threw = true; \
        } \
        if (!threw) { \
            ++g_failures; \
            fprintf(stderr, "FAIL %s:%d: %s did not throw\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

// ---- A-15.2b: dynamic slot map + promotion (arbitrary access sequence oracle) ----

// shared multi-round env: full tensors (copy source + reference path) and a separate pool buffer
struct dyn_env {
    ggml_context * ctx_d = nullptr;
    llama_adapter_loras loras;
    llm_graph_result res{2048};
    llm_graph_params params{};
    std::unique_ptr<llm_graph_context> gctx;

    ggml_tensor * cur = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * up_pool = nullptr;
    ggml_tensor * gate_pool = nullptr;
    ggml_tensor * down_pool = nullptr;
    ggml_tensor * gate_inp[5] = {nullptr};
    ggml_tensor * table[5] = {nullptr};

    std::unique_ptr<q4e_expert_pool> pool;
    q4e_pool_dynamic * dyn = nullptr;
    q4e_expert_h2d * h2d = nullptr;

    // one persistent backend + sched for all rounds (production pattern: the gallocr
    // must outlive the op-node data pointers it sets; the graph is rebuilt per round)
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    void pool_setup() {
        q4e_expert_pool_config pcfg;
        pcfg.n_layers = 1;
        pcfg.n_experts_per_layer = 8;
        pcfg.l1_slots_per_layer = 5; // R = 5
        pcfg.l2_slots_per_layer = 8; // all experts can be L2
        pool.reset(new q4e_expert_pool(pcfg));
        dyn = new q4e_pool_dynamic(*pool);
        q4e_h2d_config cfg;
        cfg.expert_bytes = int64_t(3) * 128 * 64 * 4; // gate + up + down rows, f32
        cfg.queue_cap = 64;
        cfg.copy_latency = 1;
        h2d = new q4e_expert_h2d(dyn, cfg);
        h2d->bind(pool.get());
        dyn->set_h2d(h2d);
    }

    void sched_setup() {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        const char * st = std::getenv("Q4E_TEST_SINGLE_THREAD");
        fprintf(stderr, "DBG sched_setup: Q4E_TEST_SINGLE_THREAD=%s\n", st ? st : "(null)");
        if (st != nullptr) {
            ggml_backend_cpu_set_n_threads(cpu, 1); // BISECT: force single-threaded compute
        }
        ggml_backend_t backends[1] = {cpu};
        sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    }

    ~dyn_env() {
        fprintf(stderr, "DBG D1 pre-sched-free\n");
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        fprintf(stderr, "DBG D2 pre-cpu-free\n");
        if (cpu) {
            ggml_backend_free(cpu);
        }
        fprintf(stderr, "DBG D3 pre-ctx-free\n");
        if (ctx_d) {
            ggml_free(ctx_d);
        }
        fprintf(stderr, "DBG D4 pre-delete\n");
        delete h2d;
        delete dyn;
        fprintf(stderr, "DBG D5 done\n");
    }
};

static void env_setup_dyn(dyn_env & env, const case_data & cd) {
    // 17 tensor objects: cur, 3 full, 3 pool, 5 gate_inp, 5 tables
    const size_t mem_size = (size_t) ggml_tensor_overhead() * 17
        + (size_t) cd.n_ff * cd.n_embd * cd.n_expert * 4 * 3
        + (size_t) cd.n_ff * cd.n_embd * 5 * 4 * 3
        + (size_t) cd.n_embd * cd.n_tokens * 4
        + (size_t) cd.n_expert * cd.n_embd * 4 * 5
        + (size_t) cd.n_expert * 4 * 5
        + 64;
    ggml_init_params ip;
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    env.ctx_d = ggml_init(ip);

    env.cur  = ggml_new_tensor_2d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_tokens);
    env.up   = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_ff, cd.n_expert);
    env.gate = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_ff, cd.n_expert);
    env.down = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_ff, cd.n_embd, cd.n_expert);
    // separate pool allocation; the full tensors stay pristine (copy source + reference path)
    env.up_pool   = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_ff, 5);
    env.gate_pool = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_embd, cd.n_ff, 5);
    env.down_pool = ggml_new_tensor_3d(env.ctx_d, GGML_TYPE_F32, cd.n_ff, cd.n_embd, 5);

    fill_tensor(env.cur, cd.cur);
    fill_tensor(env.up, cd.up_w);
    fill_tensor(env.gate, cd.gate_w);
    fill_tensor(env.down, cd.down_w);

    env.params.arch = LLM_ARCH_QWEN4EXP;
    env.params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    env.params.n_outputs = (uint32_t) cd.n_tokens;
    env.params.hparams.n_embd = cd.n_embd;
    env.params.hparams.n_expert = cd.n_expert;
    env.params.hparams.n_expert_used = cd.k;
    env.params.hparams.n_layer_all = 2;
    env.params.hparams.swiglu_clamp_exp.fill(0.0f);
    env.params.ubatch.n_tokens = cd.n_tokens;
    env.params.res = &env.res;
    env.params.loras = &env.loras;

    env.gctx.reset(new llm_graph_context(env.params));
}

// one round: access sequence (promotions), optional in-flight overlap probe, drain,
// fill the slot table, then pool path vs full path vs host oracle
static double dyn_round(
      dyn_env & env,
      const case_data & cd,
      ggml_tensor * gate_inp,
      ggml_tensor * table,
      bool probe) {
    env.dyn->begin_round(0);
    std::vector<int> promoted;
    for (int i = 0; i < cd.k; ++i) {
        const int e = cd.sel[0][i];
        if (env.dyn->access(0, e) != 0) {
            promoted.push_back(e);
        }
    }
    if (probe && !promoted.empty()) {
        // issue the copies, then re-access while inflight (A-15.3 overlap caliber)
        env.h2d->drain_once();
        CHECK(env.h2d->has_pending());
        for (int e : promoted) {
            env.dyn->access(0, e);
        }
    }
    env.dyn->drain();
    CHECK(!env.h2d->has_pending());
    env.dyn->end_round(0);
    fprintf(stderr, "DBG M1 round-end\n");

    // every selected expert must now be mapped (in the pool buffer)
    for (int i = 0; i < cd.k; ++i) {
        CHECK(env.dyn->slot_of(0, cd.sel[0][i]) >= 0);
    }

    // slot table: mapped -> slot, unmapped -> R sentinel (weight zeroed in graph)
    const int R = env.pool->cfg().l1_slots_per_layer;
    float * tb = (float *) ggml_get_data(table);
    for (int e = 0; e < cd.n_expert; ++e) {
        const int s = env.dyn->slot_of(0, e);
        tb[e] = (s >= 0) ? float(s) : float(R);
    }

    fprintf(stderr, "DBG M2 table-filled\n");
    // fresh graph per round on the persistent sched (same pattern as llama_eval)
    env.res.reset();
    ggml_backend_sched_reset(env.sched);
    fprintf(stderr, "DBG M3 reset-done\n");

    ggml_tensor * ref = env.gctx->build_moe_ffn(
            env.cur, gate_inp,
            env.up, env.gate, env.down,
            nullptr,
            cd.n_expert, cd.k,
            LLM_FFN_SILU, true, 2.5f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, 0,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    CHECK(ref != nullptr);
    fprintf(stderr, "DBG M4 ref-built\n");

    ggml_tensor * pool = env.gctx->build_moe_ffn_pool(
            env.cur, gate_inp,
            env.up_pool, env.gate_pool, env.down_pool,
            cd.n_expert, cd.k,
            LLM_FFN_SILU, true, 2.5f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, 0, table);
    CHECK(pool != nullptr);
    fprintf(stderr, "DBG M5 pool-built\n");

    const ggml_status st = ggml_backend_sched_graph_compute(env.sched, env.res.get_gf());
    CHECK(st == GGML_STATUS_SUCCESS);
    fprintf(stderr, "DBG M6 computed\n");

    std::vector<float> ref_out = read_out(ref);
    fprintf(stderr, "DBG M7 ref-read\n");
    std::vector<float> pool_out = read_out(pool);
    fprintf(stderr, "DBG M8 pool-read\n");

    double max_abs = 0.0;
    for (int t = 0; t < cd.n_tokens; ++t) {
        std::vector<double> expect = host_token_out(cd, t, cd.n_expert, 2.5f);
        for (int i = 0; i < cd.n_embd; ++i) {
            const float pv = pool_out[(size_t) i + (size_t) t * cd.n_embd];
            const float rv = ref_out[(size_t) i + (size_t) t * cd.n_embd];
            CHECK_NEAR(pv, expect[(size_t) i], 1e-4); // core acceptance gate
            CHECK_NEAR(rv, expect[(size_t) i], 1e-4);
            max_abs = std::max(max_abs, std::fabs((double) pv - (double) rv));
        }
    }
    CHECK(max_abs <= 1e-4 * 2.0);
    fprintf(stderr, "DBG M9 checks-done\n");
    return max_abs;
}

// arbitrary access sequence: forced LRU eviction, L3 two-stage and L2 re-promotion,
// overlap probe, exact promotion/dedup/hit counts
static void test_dynamic_oracle_sequence() {
    case_data cds[5];
    const double targets[5][8] = {
        {4.0, 3.5, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0}, // top-3 {0,1,2}  all L1 hits
        {3.0, 1.0, 0.9, 0.8, 0.7, 4.0, 3.5, 0.0}, // top-3 {0,5,6}  5,6 L3 promotion
        {0.5, 4.0, 0.4, 3.5, 0.3, 0.2, 0.1, 3.9}, // top-3 {1,7,3}  7 L3, 3 L2 re-promote
        {4.0, 0.5, 3.5, 0.4, 3.0, 0.2, 0.1, 0.0}, // top-3 {0,2,4}  all L2 re-promote
        {0.5, 0.4, 3.0, 0.3, 0.2, 4.0, 3.5, 0.1}, // top-3 {5,6,2}  5,6 second stage, 2 hit
    };
    std::vector<token_group> groups(1);
    std::vector<int> token_to_group = {0, 0, 0, 0};
    for (int r = 0; r < 5; ++r) {
        groups[0].target.assign(targets[r], targets[r] + 8);
        make_case(cds[r], groups, token_to_group);
        CHECK_EQ_I64(cds[r].sel[0].size(), 3);
    }

    dyn_env env;
    env.pool_setup();
    env_setup_dyn(env, cds[0]);
    env.sched_setup();
    for (int r = 0; r < 5; ++r) {
        env.gate_inp[r] = ggml_new_tensor_2d(env.ctx_d, GGML_TYPE_F32, cds[r].n_embd, cds[r].n_expert);
        fill_tensor(env.gate_inp[r], cds[r].gate_inp);
        env.table[r] = ggml_new_tensor_1d(env.ctx_d, GGML_TYPE_F32, cds[r].n_expert);
    }
    // initial pool load: slot i holds expert i (matches the dynamic map seed)
    for (int i = 0; i < 5; ++i) {
        q4e_pool_promote_into(env.up_pool, env.gate_pool, env.down_pool,
                              env.up, env.gate, env.down, i, i);
    }
    env.dyn->set_layer_tensors(0, env.up_pool, env.gate_pool, env.down_pool,
                               env.up, env.gate, env.down);

    for (int r = 0; r < 5; ++r) {
        const bool probe = (r == 1); // in-flight overlap probe (A-15.3 caliber)
        const double max_abs = dyn_round(env, cds[r], env.gate_inp[r], env.table[r], probe);
        fprintf(stderr, "dyn round %d: pool vs full max_abs_err = %.3e\n", r, max_abs);
        CHECK(max_abs <= 1e-4);
    }

    // exact counters: first out-of-pool per episode issues once; dedup hits do not re-issue
    CHECK_EQ_I64(env.dyn->hits_l1(), 6);
    CHECK_EQ_I64(env.dyn->misses_l2(), 6);
    CHECK_EQ_I64(env.dyn->misses_l3(), 5);
    CHECK_EQ_I64(env.dyn->promotions(), 8);
    CHECK_EQ_I64(env.h2d->h2d_issued(), 9);
    CHECK_EQ_I64(env.h2d->h2d_completed(), 9);
    CHECK_EQ_I64(env.h2d->h2d_bytes(), 9 * int64_t(3) * 128 * 64 * 4);
    CHECK_EQ_I64(env.h2d->overlap_hits(), 2);
    CHECK_EQ_I64(env.h2d->dedup_skips(), 0);
    CHECK_EQ_I64(env.h2d->h2d_dropped(), 0);
    CHECK_EQ_I64(env.h2d->accesses(), 17);

    // pool manager view: LRU evictions, two-stage L3, M count per round
    q4e_expert_pool_stats ps = env.pool->stats();
    CHECK_EQ_I64(ps.accesses, 11); // seed 5 + L1 hits 6
    CHECK_EQ_I64(ps.hits_l1, 6);
    CHECK_EQ_I64(ps.hits_l3, 5); // seed only
    CHECK_EQ_I64(ps.l2_fills, 8); // seed 5 + 5,6,7
    CHECK_EQ_I64(ps.l2_to_l1, 6); // 3,0,2,4,5,6
    CHECK_EQ_I64(ps.l1_evictions, 7); // 3,4,2,0,1,3,0
    CHECK_EQ_I64(ps.rounds, 5);
    CHECK_EQ_I64(ps.m_sum, 15);
    CHECK_EQ_I64(ps.m_max, 3);

    // LRU order check: the evicted expert is the longest-unhit resident
    CHECK(std::strcmp(env.pool->state_of(0, 5), "l1") == 0);
    CHECK(std::strcmp(env.pool->state_of(0, 6), "l1") == 0);
    CHECK(std::strcmp(env.pool->state_of(0, 2), "l1") == 0);
    CHECK_EQ_I64(env.dyn->slot_of(0, 3), -1);
}

// per-layer independence + L3 two-stage + range guards (no ggml tensors involved)
static void test_dynamic_multilayer() {
    fprintf(stderr, "DBG T2 multilayer-start\n");
    q4e_expert_pool_config pcfg;
    pcfg.n_layers = 2;
    pcfg.n_experts_per_layer = 8;
    pcfg.l1_slots_per_layer = 5;
    pcfg.l2_slots_per_layer = 8;
    q4e_expert_pool pool(pcfg);
    q4e_pool_dynamic dyn(pool);
    q4e_h2d_config cfg;
    cfg.expert_bytes = 100;
    cfg.queue_cap = 8;
    cfg.copy_latency = 1;
    q4e_expert_h2d h2d(&dyn, cfg);
    h2d.bind(&pool);
    dyn.set_h2d(&h2d);

    // layer 1 promotes independently of layer 0
    dyn.begin_round(1);
    CHECK_EQ_I64(dyn.access(1, 6), 2); // fresh L3
    CHECK_EQ_I64(dyn.access(1, 7), 2);
    dyn.end_round(1);
    dyn.drain();

    CHECK(dyn.slot_of(1, 6) >= 0);
    CHECK(dyn.slot_of(1, 7) >= 0);
    CHECK_EQ_I64(dyn.slot_of(1, 0), -1); // LRU victim (slot 0)
    CHECK_EQ_I64(dyn.slot_of(1, 1), -1); // LRU victim (slot 1)
    CHECK_EQ_I64(dyn.slot_of(0, 0), 0); // layer 0 untouched
    CHECK_EQ_I64(dyn.slot_of(0, 4), 4);
    CHECK_EQ_I64(dyn.promotions(), 2);
    CHECK_EQ_I64(h2d.h2d_issued(), 2);
    CHECK_EQ_I64(h2d.h2d_completed(), 2);
    // L3 -> L1 takes two stages: after one job the expert sits in L2
    CHECK(std::strcmp(pool.state_of(1, 6), "l2") == 0);
    CHECK(std::strcmp(pool.state_of(1, 7), "l2") == 0);

    // range guards
    CHECK_THROWS(dyn.access(2, 0));
    CHECK_THROWS(dyn.access(1, 8));
    CHECK_EQ_I64(dyn.slot_of(1, 8), -1);
}

#if 0 // BISECT: Phase 2 v6 tests disabled
// ---- q4e v6 global expert pool (A-16 phase 2) ----

// 2b: round-robin slot map math (no ggml)
static void test_v6_slot_map() {
    const int n_layers = 2, n_expert = 8;

    auto tables = q4e_v6_slot_map(6, n_layers, n_expert);
    CHECK_EQ_I64(tables.size(), n_layers);
    CHECK_EQ_I64(tables[0].size(), n_expert);
    CHECK_EQ_I64(tables[1].size(), n_expert);
    // slot s -> layer s%2, expert (s/2)%8
    CHECK_EQ_I64(tables[0][0], 0);
    CHECK_EQ_I64(tables[0][1], 2);
    CHECK_EQ_I64(tables[0][2], 4);
    for (int e = 3; e < n_expert; ++e) {
        CHECK_EQ_I64(tables[0][e], 6); // sentinel = n_slots
    }
    CHECK_EQ_I64(tables[1][0], 1);
    CHECK_EQ_I64(tables[1][1], 3);
    CHECK_EQ_I64(tables[1][2], 5);
    for (int e = 3; e < n_expert; ++e) {
        CHECK_EQ_I64(tables[1][e], 6);
    }
    // bijection: every slot has exactly one (layer, expert) owner
    std::vector<int> slot_owner(6, -1);
    for (int l = 0; l < n_layers; ++l) {
        for (int e = 0; e < n_expert; ++e) {
            const int s = tables[l][e];
            if (s < 6) {
                CHECK(slot_owner[s] < 0);
                slot_owner[s] = l * n_expert + e;
            }
        }
    }
    for (int s = 0; s < 6; ++s) {
        CHECK(slot_owner[s] >= 0);
    }
    // full coverage: no sentinel
    auto full = q4e_v6_slot_map(16, n_layers, n_expert);
    int n_sentinel = 0;
    for (const auto & tl : full) {
        for (int s : tl) {
            if (s >= 16) ++n_sentinel;
        }
    }
    CHECK_EQ_I64(n_sentinel, 0);
    // N beyond n_layers*n_expert: wraps, superseded slots never referenced
    auto over = q4e_v6_slot_map(17, n_layers, n_expert);
    std::vector<int> used(17, 0);
    for (const auto & tl : over) {
        for (int s : tl) {
            if (s < 17) ++used[s];
        }
    }
    CHECK_EQ_I64(used[0], 0); // slot 0 superseded by slot 16 (wrap)
    int n_used = 0;
    for (int u : used) {
        if (u > 0) ++n_used;
    }
    CHECK_EQ_I64(n_used, 16); // all (layer, expert) pairs covered
}

// 2b: static fill from full expert tensors (incl. fused gate_up views) into pool slots
static void test_v6_static_fill() {
    const int n_embd = 16, n_ff = 32, n_expert = 8, n_layers = 2, n_slots = 17;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto rnd = [&]() { return dist(rng); };

    const size_t n_up   = (size_t) n_embd * n_ff * n_expert;
    const size_t n_down = (size_t) n_ff * n_embd * n_expert;
    const size_t mem_size = (size_t) ggml_tensor_overhead() * 32
        + n_up * 4 * 2 + n_down * 4 * 2 // L0 separate up/gate/down
        + (size_t) n_embd * (2 * n_ff) * n_expert * 4 // L1 fused gate_up
        + (size_t) n_ff * n_embd * n_expert * 4 // L1 down
        + (size_t) n_embd * n_ff * n_slots * 4 * 2 // up/gate pool
        + (size_t) n_ff * n_embd * n_slots * 4 // down pool
        + 64;
    ggml_init_params ip;
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);

    auto rand_fill = [&](ggml_tensor * t) {
        float * p = (float *) ggml_get_data(t);
        for (size_t i = 0; i < (size_t) t->ne[0] * t->ne[1] * t->ne[2]; ++i) {
            p[i] = rnd();
        }
    };

    // layer 0: separate up/gate tensors
    ggml_tensor * up0   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * gate0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * down0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_expert);
    rand_fill(up0); rand_fill(gate0); rand_fill(down0);

    // layer 1: fused gate_up tensor [n_embd, 2*n_ff, n_expert]
    ggml_tensor * fused1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 2 * n_ff, n_expert);
    ggml_tensor * down1  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_expert);
    rand_fill(fused1); rand_fill(down1);

    // pool with one extra slot: slot 0 is superseded by slot 16 (wrap), so it stays untouched
    ggml_tensor * up_pool   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_slots);
    ggml_tensor * gate_pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_slots);
    ggml_tensor * down_pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_slots);
    const float sentinel = 123456.75f;
    for (auto * pool : {up_pool, gate_pool, down_pool}) {
        float * p = (float *) ggml_get_data(pool);
        for (size_t i = 0; i < (size_t) pool->ne[0] * pool->ne[1] * pool->ne[2]; ++i) {
            p[i] = sentinel;
        }
    }

    auto tables = q4e_v6_slot_map(n_slots, n_layers, n_expert);
    CHECK_EQ_I64(tables[0][0], 16);
    CHECK_EQ_I64(tables[1][0], 1);

    // gate = first half of the fused tensor, up = second half
    const auto & gu = fused1;
    ggml_tensor * up1   = ggml_view_3d(ctx, gu, gu->ne[0], n_ff, gu->ne[2], gu->nb[1], gu->nb[2], (size_t) n_ff * gu->nb[1]);
    ggml_tensor * gate1 = ggml_view_3d(ctx, gu, gu->ne[0], n_ff, gu->ne[2], gu->nb[1], gu->nb[2], 0);

    q4e_v6_pool_fill(up_pool, gate_pool, down_pool,
        {up0, up1}, {gate0, gate1}, {down0, down1}, tables);

    // pool slot must hold a byte-exact copy of the source expert slice
    auto check_block = [&](ggml_tensor * pool, int slot, ggml_tensor * src, int expert) {
        const float * p = (const float *) ggml_get_data(pool) + (size_t) slot * pool->nb[2] / 4;
        const float * s = (const float *) ggml_get_data(src)  + (size_t) expert * src->nb[2] / 4;
        for (int64_t a = 0; a < src->ne[1]; ++a) {
            for (int64_t b = 0; b < src->ne[0]; ++b) {
                CHECK(p[a * (pool->nb[1] / 4) + b] == s[a * (src->nb[1] / 4) + b * (src->nb[0] / 4)]);
            }
        }
    };
    for (int e = 0; e < n_expert; ++e) {
        const int s0 = tables[0][e];
        check_block(up_pool, s0, up0, e);
        check_block(gate_pool, s0, gate0, e);
        check_block(down_pool, s0, down0, e);
        const int s1 = tables[1][e];
        check_block(up_pool, s1, up1, e);
        check_block(gate_pool, s1, gate1, e);
        check_block(down_pool, s1, down1, e);
    }
    // unused slot 0: untouched by the fill
    for (auto * pool : {up_pool, gate_pool, down_pool}) {
        const float * p = (const float *) ggml_get_data(pool);
        bool all_sentinel = true;
        for (int64_t a = 0; a < pool->ne[1]; ++a) {
            for (int64_t b = 0; b < pool->ne[0]; ++b) {
                if (p[a * (pool->nb[1] / 4) + b] != sentinel) {
                    all_sentinel = false;
                }
            }
        }
        CHECK(all_sentinel);
    }
    ggml_free(ctx);
}

// expected token output for the v6 pool path: experts with table[e] >= n_slots contribute 0
static std::vector<double> host_token_out_v6(const case_data & cd, int t, const std::vector<int> & table, int n_slots, float w_scale) {
    const std::vector<double> & l = cd.target[t];
    double m = -1e30;
    for (int e = 0; e < cd.n_expert; ++e) {
        m = std::max(m, l[e]);
    }
    std::vector<double> p(cd.n_expert);
    double s = 0.0;
    for (int e = 0; e < cd.n_expert; ++e) {
        p[e] = std::exp(l[e] - m);
        s += p[e];
    }
    double S = 0.0;
    for (int i = 0; i < cd.k; ++i) {
        S += p[cd.sel[t][i]];
    }
    std::vector<double> out(cd.n_embd, 0.0);
    for (int i = 0; i < cd.k; ++i) {
        const int e = cd.sel[t][i];
        if (table[e] >= n_slots) {
            continue;
        }
        const double w = p[e] / S * w_scale;
        const std::vector<double> eo = host_expert_out(cd, e, t);
        for (int j = 0; j < cd.n_embd; ++j) {
            out[j] += w * eo[j];
        }
    }
    return out;
}

// 2a/2b: one shared N-slot pool + per-layer round-robin tables;
// all top-k in pool: pool path == full path (oracle gate); out-of-pool experts contribute 0
static void test_v6_global_pool_oracle() {
    const int n_layers = 2, n_slots = 6, k = 3;

    case_data cds[2];
    // layer 0: top-3 {0,1,2}, all in pool (L0 slots: 0->0, 1->2, 2->4)
    std::vector<token_group> groups0(1);
    groups0[0].target = {4.0, 3.5, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0};
    make_case(cds[0], groups0, {0, 0, 0, 0});
    // layer 1: top-3 {0,1,6}, expert 6 out of pool (L1 slots: 0->1, 1->3, 2->5)
    std::vector<token_group> groups1(1);
    groups1[0].target = {4.0, 3.5, 0.5, 0.4, 0.3, 0.2, 1.0, 0.9};
    make_case(cds[1], groups1, {0, 0, 0, 0});
    CHECK(cds[0].sel[0] == (std::vector<int>{0, 1, 2}));
    CHECK(cds[1].sel[0] == (std::vector<int>{0, 1, 6}));

    const int n_embd   = cds[0].n_embd;
    const int n_ff     = cds[0].n_ff;
    const int n_expert = cds[0].n_expert;
    const int n_tokens = cds[0].n_tokens;
    const size_t mem_size = (size_t) ggml_tensor_overhead() * 32
        + (size_t) n_embd * n_ff * n_expert * 4 * 4 // up/gate full, 2 layers
        + (size_t) n_ff * n_embd * n_expert * 4 * 2 // down full, 2 layers
        + (size_t) n_embd * n_ff * n_slots * 4 * 2 // up/gate pool
        + (size_t) n_ff * n_embd * n_slots * 4 // down pool
        + (size_t) n_embd * n_tokens * 4
        + (size_t) n_expert * n_embd * 4 * 2
        + 64;
    ggml_init_params ip;
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);

    ggml_tensor * cur    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * gate_inp[2];
    ggml_tensor * up_full[2], * gate_full[2], * down_full[2];
    for (int li = 0; li < n_layers; ++li) {
        gate_inp[li]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_expert);
        up_full[li]   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
        gate_full[li] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
        down_full[li] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_expert);
        fill_tensor(gate_inp[li], cds[li].gate_inp);
        fill_tensor(up_full[li], cds[li].up_w);
        fill_tensor(gate_full[li], cds[li].gate_w);
        fill_tensor(down_full[li], cds[li].down_w);
    }
    fill_tensor(cur, cds[0].cur); // same cur for both layers

    // one shared pool for both layers, filled round-robin
    ggml_tensor * up_pool   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_slots);
    ggml_tensor * gate_pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_slots);
    ggml_tensor * down_pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_slots);
    auto tables = q4e_v6_slot_map(n_slots, n_layers, n_expert);
    q4e_v6_pool_fill(up_pool, gate_pool, down_pool,
        {up_full[0], up_full[1]}, {gate_full[0], gate_full[1]}, {down_full[0], down_full[1]}, tables);

    ggml_tensor * table[2];
    for (int li = 0; li < n_layers; ++li) {
        table[li] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
        float * tp = (float *) ggml_get_data(table[li]);
        for (int e = 0; e < n_expert; ++e) {
            tp[e] = (float) tables[li][e];
        }
    }
    {
        const float * t0 = (const float *) ggml_get_data(table[0]);
        CHECK(t0[0] == 0.0f && t0[1] == 2.0f && t0[2] == 4.0f && t0[3] == 6.0f);
        const float * t1 = (const float *) ggml_get_data(table[1]);
        CHECK(t1[0] == 1.0f && t1[1] == 3.0f && t1[2] == 5.0f && t1[3] == 6.0f);
    }

    llama_adapter_loras loras;
    llm_graph_result res{1024};
    llm_graph_params params{};
    params.arch = LLM_ARCH_QWEN4EXP;
    params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    params.n_outputs = (uint32_t) n_tokens;
    params.hparams.n_embd = n_embd;
    params.hparams.n_expert = n_expert;
    params.hparams.n_expert_used = k;
    params.hparams.n_layer_all = n_layers;
    params.hparams.swiglu_clamp_exp.fill(0.0f);
    params.ubatch.n_tokens = n_tokens;
    params.res = &res;
    params.loras = &loras;
    std::unique_ptr<llm_graph_context> gctx(new llm_graph_context(params));
    const float w_scale = 2.0f;

    ggml_tensor * ref[2], * pool[2];
    for (int li = 0; li < n_layers; ++li) {
        ref[li] = gctx->build_moe_ffn(
                cur, gate_inp[li],
                up_full[li], gate_full[li], down_full[li],
                nullptr,
                n_expert, k,
                LLM_FFN_SILU, true, w_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, li,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        CHECK(ref[li] != nullptr);

        pool[li] = gctx->build_moe_ffn_pool(
                cur, gate_inp[li],
                up_pool, gate_pool, down_pool,
                n_expert, k,
                LLM_FFN_SILU, true, w_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, li, table[li]);
        CHECK(pool[li] != nullptr);
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    ggml_backend_t backends[1] = {cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    CHECK(sched != nullptr);
    const ggml_status st = ggml_backend_sched_graph_compute(sched, res.get_gf());
    CHECK(st == GGML_STATUS_SUCCESS);

    std::vector<float> ref_out[2], pool_out[2];
    for (int li = 0; li < n_layers; ++li) {
        ref_out[li] = read_out(ref[li]);
        pool_out[li] = read_out(pool[li]);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    // layer 0: all top-k in pool -> pool path == full path == host oracle (2a gate)
    for (int t = 0; t < n_tokens; ++t) {
        std::vector<double> expect = host_token_out(cds[0], t, n_expert, w_scale);
        for (int i = 0; i < n_embd; ++i) {
            const float gv = pool_out[0][(size_t) i + (size_t) t * n_embd];
            const float rv = ref_out[0][(size_t) i + (size_t) t * n_embd];
            CHECK_NEAR(gv, expect[(size_t) i], 1e-4);
            CHECK_NEAR(rv, expect[(size_t) i], 1e-4);
            CHECK(std::fabs((double) gv - (double) rv) <= 1e-4 * (1.0 + std::fabs(rv)));
        }
    }

    // layer 1: expert 6 out of pool -> pool path zero-weights it, full path keeps it
    double max_diff = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        std::vector<double> expect_pool = host_token_out_v6(cds[1], t, tables[1], n_slots, w_scale);
        std::vector<double> expect_full = host_token_out(cds[1], t, n_expert, w_scale);
        for (int i = 0; i < n_embd; ++i) {
            const float gv = pool_out[1][(size_t) i + (size_t) t * n_embd];
            const float rv = ref_out[1][(size_t) i + (size_t) t * n_embd];
            CHECK_NEAR(gv, expect_pool[(size_t) i], 1e-4);
            CHECK_NEAR(rv, expect_full[(size_t) i], 1e-4);
            max_diff = std::max(max_diff, std::fabs((double) gv - (double) rv));
        }
    }
    CHECK(max_diff > 1e-3); // expert 6 must be visible in the difference
    fprintf(stderr, "v6-global-pool-oracle: layer0 (all in pool) pool==full==oracle, layer1 (1 out of pool) diff = %.3e\n", max_diff);
    ggml_free(ctx);
}
#endif // BISECT: Phase 2 v6 tests disabled

#if 0 // BISECT: Phase 3 test disabled
// 3a: runtime promotion round. Decode sequence: 3 steps, 2 layers, one shared 6-slot
// pool with LRU eviction between steps. The backend drives the round (reserve the
// round's top-k, promote the missing through the pipeline, verify residency); its L1
// is the model pool (attach_l1). After each round every top-k expert is resident, so
// the pool path must equal the full-tensor path (build_moe_ffn) and the host oracle.
static void test_v6_runtime_decode_sequence() {
    const int n_layers = 2, n_slots = 6, n_embd = 64, n_ff = 128, n_expert = 8, n_tokens = 1;
    const int k = 3;

    // attach_l1 error paths (external_l1 unset, null, double attach) + slot routing
    {
        q4e_pool_backend_config c0;
        c0.n_ff = 4;
        c0.n_embd = 4;
        c0.n_l1_slots = 2;
        c0.n_l2_slots = 2;
        c0.pool.n_experts_per_layer = 4;
        c0.pool.n_layers = 1;
        c0.pool.n_slots = 2;
        c0.pool.floor = 0;
        q4e_pool_backend be0(c0);
        CHECK_THROWS(be0.attach_l1(nullptr, nullptr, nullptr)); // external_l1 not set
        q4e_pool_backend_config c1 = c0;
        c1.external_l1 = true;
        q4e_pool_backend be1(c1);
        CHECK_THROWS(be1.attach_l1(nullptr, nullptr, nullptr)); // null pointer
        float a[32] = { 0 };
        float b[32] = { 0 };
        float c[32] = { 0 };
        be1.attach_l1(a, b, c);
        CHECK_THROWS(be1.attach_l1(a, b, c)); // already attached
        CHECK(be1.up_slot(1) == a + 16);      // slot 1: the 16-float block after slot 0
        CHECK_THROWS(be1.up_slot(2));         // out of range
        CHECK(be1.graph_slot(0, 0) == 2);     // out-of-pool -> n_l1_slots sentinel
    }

    // one token per decode step; seed 42 keeps the weights identical across steps,
    // so the model is fixed and only the token (gate_inp/target) changes per step
    case_data cd0[3], cd1[3]; // [step][layer]
    const double t0[] = { 4.0, 3.5, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0 }; // sel {0,1,2}
    const double t1[] = { 2.0, 1.5, 3.0, 4.0, 3.5, 1.0, 0.5, 0.0 }; // sel {3,4,2}
    const double t2a[] = { 1.0, 0.5, 0.4, 0.3, 0.2, 4.0, 3.5, 3.0 }; // sel {5,6,7}
    const double t2b[] = { 4.0, 0.5, 0.4, 0.3, 0.2, 0.1, 3.5, 3.0 }; // sel {0,6,7}
    for (int s = 0; s < 3; ++s) {
        const double * ta = (s == 0) ? t0 : (s == 1) ? t1 : t2a;
        const double * tb = (s == 2) ? t2b : ta;
        std::vector<token_group> ga(1);
        std::vector<token_group> gb(1);
        ga[0].target.assign(ta, ta + n_expert);
        gb[0].target.assign(tb, tb + n_expert);
        make_case(cd0[s], ga, { 0 });
        make_case(cd1[s], gb, { 0 });
    }
    CHECK(cd0[0].sel[0] == (std::vector<int> { 0, 1, 2 }));
    CHECK(cd1[0].sel[0] == (std::vector<int> { 0, 1, 2 }));
    CHECK(cd0[1].sel[0] == (std::vector<int> { 3, 4, 2 }));
    CHECK(cd1[1].sel[0] == (std::vector<int> { 3, 4, 2 }));
    CHECK(cd0[2].sel[0] == (std::vector<int> { 5, 6, 7 }));
    CHECK(cd1[2].sel[0] == (std::vector<int> { 0, 6, 7 }));
    CHECK(cd0[0].up_w == cd0[2].up_w); // same seed: the model weights are step-invariant
    CHECK(cd0[0].up_w == cd1[2].up_w);

    const int64_t ebytes = (int64_t) 3 * n_ff * n_embd * (int64_t) sizeof(float);
    q4e_h2d_pipeline_config pcfg;
    pcfg.l2_ram_mb = 4;
    pcfg.expert_bytes = ebytes;
    pcfg.max_inflight = 2; // below the missing count per round: the busy path must fire
    q4e_h2d_pipeline pipe(pcfg);

    q4e_pool_backend_config cfg;
    cfg.n_ff = n_ff;
    cfg.n_embd = n_embd;
    cfg.n_l1_slots = n_slots;
    cfg.n_l2_slots = (int) pipe.l2_slots() + 2; // must cover the pipeline slot range
    cfg.external_l1 = true; // L1 is the model pool below
    cfg.pool.n_experts_per_layer = n_expert;
    cfg.pool.n_layers = n_layers;
    cfg.pool.n_slots = n_slots;
    cfg.pool.floor = 0;
    q4e_pool_backend be(cfg);

    size_t mem_size = (size_t) ggml_tensor_overhead() * 256
        + (size_t) n_embd * n_ff * n_expert * 4 * 6 // up/gate/down full, 2 layers
        + (size_t) n_embd * n_ff * n_slots * 4 * 2 // up/gate pool
        + (size_t) n_ff * n_embd * n_slots * 4 // down pool
        + (size_t) n_embd * n_tokens * 4 * 3 // cur per step
        + (size_t) n_expert * n_embd * 4 * 6 // gate_inp per step/layer
        + (size_t) n_expert * 4 * 2 // tables
        + 4096;
    ggml_init_params ip;
    ip.mem_size = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);

    // the model: fixed expert weights + the shared pool (the backend's external L1)
    ggml_tensor * up_full[2], * gate_full[2], * down_full[2];
    for (int li = 0; li < n_layers; ++li) {
        up_full[li] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
        gate_full[li] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
        down_full[li] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_expert);
        fill_tensor(up_full[li], cd0[0].up_w);
        fill_tensor(gate_full[li], cd0[0].gate_w);
        fill_tensor(down_full[li], cd0[0].down_w);
    }
    ggml_tensor * cur[3];
    ggml_tensor * gate_inp[3][2];
    for (int s = 0; s < 3; ++s) {
        cur[s] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        fill_tensor(cur[s], cd0[s].cur);
        for (int li = 0; li < n_layers; ++li) {
            gate_inp[s][li] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_expert);
            fill_tensor(gate_inp[s][li], (li == 0) ? cd0[s].gate_inp : cd1[s].gate_inp);
        }
    }
    ggml_tensor * up_pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_slots);
    ggml_tensor * gate_pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_slots);
    ggml_tensor * down_pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_slots);
    std::memset(ggml_get_data(up_pool), 0, (size_t) n_embd * n_ff * n_slots * 4);
    std::memset(ggml_get_data(gate_pool), 0, (size_t) n_embd * n_ff * n_slots * 4);
    std::memset(ggml_get_data(down_pool), 0, (size_t) n_ff * n_embd * n_slots * 4);
    ggml_tensor * table[2];
    for (int li = 0; li < n_layers; ++li) {
        table[li] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
        std::memset(ggml_get_data(table[li]), 0, (size_t) n_expert * 4);
    }
    be.attach_l1(
        (float *) ggml_get_data(up_pool),
        (float *) ggml_get_data(gate_pool),
        (float *) ggml_get_data(down_pool));
    be.set_layer_source(0, (const float *) ggml_get_data(up_full[0]), (const float *) ggml_get_data(gate_full[0]), (const float *) ggml_get_data(down_full[0]));
    be.set_layer_source(1, (const float *) ggml_get_data(up_full[1]), (const float *) ggml_get_data(gate_full[1]), (const float *) ggml_get_data(down_full[1]));
    be.wire(pipe);

    llama_adapter_loras loras;
    const float w_scale = 2.0f;
    for (int s = 0; s < 3; ++s) {
        // the router top-k of this decode step, as the graph router computes it
        std::vector<std::vector<int>> topk(n_layers);
        topk[0] = cd0[s].sel[0];
        topk[1] = cd1[s].sel[0];

        be.sync_round(pipe, topk);

        // 3a invariant: every top-k expert of the round is resident after the round
        for (int li = 0; li < n_layers; ++li) {
            for (int e : topk[li]) {
                CHECK_EQ_I64(be.slot_of(li, e) >= 0, 1);
            }
        }
        // graph_slot mapping: in-pool -> slot id, out-of-pool -> the n_slots sentinel
        if (s == 1) {
            CHECK_EQ_I64(be.graph_slot(0, 0), n_slots);
            CHECK_EQ_I64(be.graph_slot(0, 2), be.slot_of(0, 2));
        }

        // slot tables synced to the graph convention before the pool compute
        for (int li = 0; li < n_layers; ++li) {
            float * tp = (float *) ggml_get_data(table[li]);
            for (int e = 0; e < n_expert; ++e) {
                tp[e] = (float) be.graph_slot(li, e);
            }
        }

        // the LRU trace so far: which experts hold a slot at this step
        if (s == 0) {
            for (int li = 0; li < n_layers; ++li) {
                for (int e = 0; e < n_expert; ++e) {
                    CHECK_EQ_I64(be.slot_of(li, e) >= 0, (e < 3) ? 1 : 0); // pool exactly filled
                }
            }
        }
        if (s == 1) {
            CHECK_EQ_I64(be.slot_of(0, 0), -1); // evicted by the round-1 inserts
            CHECK_EQ_I64(be.slot_of(0, 1), -1);
            CHECK_EQ_I64(be.slot_of(1, 0), -1);
            CHECK_EQ_I64(be.slot_of(1, 1), -1);
            for (int li = 0; li < n_layers; ++li) {
                CHECK_EQ_I64(be.slot_of(li, 2) >= 0, 1);
                CHECK_EQ_I64(be.slot_of(li, 3) >= 0, 1);
                CHECK_EQ_I64(be.slot_of(li, 4) >= 0, 1);
            }
        }
        if (s == 2) {
            CHECK_EQ_I64(be.slot_of(0, 2), -1);
            CHECK_EQ_I64(be.slot_of(0, 3), -1);
            CHECK_EQ_I64(be.slot_of(0, 4), -1);
            CHECK_EQ_I64(be.slot_of(1, 2), -1);
            CHECK_EQ_I64(be.slot_of(1, 3), -1);
            CHECK_EQ_I64(be.slot_of(1, 4), -1);
            CHECK_EQ_I64(be.slot_of(0, 5) >= 0, 1); // the round-2 union fills the pool
            CHECK_EQ_I64(be.slot_of(0, 6) >= 0, 1);
            CHECK_EQ_I64(be.slot_of(0, 7) >= 0, 1);
            CHECK_EQ_I64(be.slot_of(1, 0) >= 0, 1); // L2 hit: refilled from L2, no re-fill
            CHECK_EQ_I64(be.slot_of(1, 6) >= 0, 1);
            CHECK_EQ_I64(be.slot_of(1, 7) >= 0, 1);
        }

        llm_graph_result res{1024};
        llm_graph_params params{};
        params.arch = LLM_ARCH_QWEN4EXP;
        params.gtype = LLM_GRAPH_TYPE_DEFAULT;
        params.n_outputs = (uint32_t) n_tokens;
        params.hparams.n_embd = n_embd;
        params.hparams.n_expert = n_expert;
        params.hparams.n_expert_used = k;
        params.hparams.n_layer_all = n_layers;
        params.hparams.swiglu_clamp_exp.fill(0.0f);
        params.ubatch.n_tokens = n_tokens;
        params.res = &res;
        params.loras = &loras;
        std::unique_ptr<llm_graph_context> gctx(new llm_graph_context(params));
        ggml_tensor * ref[2], * pool[2];
        for (int li = 0; li < n_layers; ++li) {
            ref[li] = gctx->build_moe_ffn(
                    cur[s], gate_inp[s][li],
                    up_full[li], gate_full[li], down_full[li],
                    nullptr,
                    n_expert, k,
                    LLM_FFN_SILU, true, w_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, li,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            CHECK(ref[li] != nullptr);
            pool[li] = gctx->build_moe_ffn_pool(
                    cur[s], gate_inp[s][li],
                    up_pool, gate_pool, down_pool,
                    n_expert, k,
                    LLM_FFN_SILU, true, w_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, li, table[li]);
            CHECK(pool[li] != nullptr);
        }
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        CHECK(cpu != nullptr);
        ggml_backend_t backends[1] = {cpu};
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        CHECK(sched != nullptr);
        const ggml_status st = ggml_backend_sched_graph_compute(sched, res.get_gf());
        CHECK(st == GGML_STATUS_SUCCESS);

        // dynamic promotion keeps the pool correct: pool path == full path == oracle
        for (int li = 0; li < n_layers; ++li) {
            std::vector<float> pool_out = read_out(pool[li]);
            std::vector<float> ref_out = read_out(ref[li]);
            const case_data & cd = (li == 0) ? cd0[s] : cd1[s];
            std::vector<double> expect = host_token_out(cd, 0, n_expert, w_scale);
            for (int i = 0; i < n_embd; ++i) {
                const float gv = pool_out[i];
                const float rv = ref_out[i];
                CHECK_NEAR(gv, expect[i], 1e-4);
                CHECK_NEAR(rv, expect[i], 1e-4);
                CHECK(std::fabs((double) gv - (double) rv) <= 1e-4 * (1.0 + std::fabs(rv)));
            }
        }
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
    }

    // the pipeline really carried the promotions: 16 submits (6+4+6), 15 fresh fills
    // (L1:0 at step 2 is an L2 hit), all copies done, the in-flight cap fired 5 times,
    // and 10 LRU evictions across the two full rounds
    CHECK_EQ_I64(pipe.stats().fills, 15);
    CHECK_EQ_I64(pipe.stats().completed, 16);
    CHECK_EQ_I64(pipe.stats().failed, 0);
    CHECK_EQ_I64(pipe.stats().busy, 5);
    CHECK_EQ_I64(be.stats().evictions, 10);
    CHECK_EQ_I64(be.stats().rounds, 6); // one round per layer per step
    fprintf(stderr, "v6-runtime-decode: 3-step decode sequence, 16 promotions (15 fills + 1 L2 hit), 10 evictions, pool==full==oracle every step\n");
    ggml_free(ctx);
}
#endif // BISECT: Phase 3 test disabled

int main() {
    test_slot_of_helper();
    test_all_in_pool_oracle();
    test_out_of_pool_zeroed();
    test_mixed_per_token();
    test_dynamic_oracle_sequence();
    test_dynamic_multilayer();
    // test_v6_slot_map(); // BISECT
    // test_v6_static_fill(); // BISECT
    // test_v6_global_pool_oracle(); // BISECT
    // test_v6_runtime_decode_sequence(); // BISECT: disabled
    if (g_failures) {
        fprintf(stderr, "q4e-moe-pool: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-moe-pool: all %d checks passed\n", g_checks);
    return 0;
}
