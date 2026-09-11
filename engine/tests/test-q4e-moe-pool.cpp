// Unit tests for the q4e v5 pool-path MoE FFN builder (A-15.2a, llama-graph.cpp build_moe_ffn_pool).
// CPU oracle: pool path vs full expert tensor path (build_moe_ffn), f32, fixed seed.
#include "llama-graph.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

int main() {
    test_slot_of_helper();
    test_all_in_pool_oracle();
    test_out_of_pool_zeroed();
    test_mixed_per_token();
    if (g_failures) {
        fprintf(stderr, "q4e-moe-pool: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("q4e-moe-pool: all %d checks passed\n", g_checks);
    return 0;
}
