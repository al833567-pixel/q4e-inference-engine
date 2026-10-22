// q4e-kbench: throughput of the CPU expert dot kernels on this machine, independent of the graph and the scheduler.
// It streams N MiB of weight rows (random bytes in the layout of the type) against q8 activation rows with T threads
// and prints GB/s of weights, for ggml's vec_dot and for the multi-row kernels.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

extern "C" {
void ggml_vec_dot_iq2_s_q8_K_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc);
void ggml_vec_dot_q2_0_q8_0_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc);
}

static double run(int n_threads, size_t n_rows, size_t row_bytes, const uint8_t * w, int ne, const std::vector<std::vector<uint8_t>> & act,
                  int nc, int kernel, ggml_type type) {
    const auto * tr = ggml_get_type_traits_cpu(type);
    std::atomic<int> go{0};
    std::vector<std::thread> th;
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < n_threads; ++t) {
        th.emplace_back([&, t]() {
            const size_t r0 = n_rows*t/n_threads, r1 = n_rows*(t + 1)/n_threads;
            float s[4];
            const void * y[4];
            for (int c = 0; c < nc; ++c) {
                y[c] = act[c].data();
            }
            float sink = 0;
            for (size_t r = r0; r < r1; ++r) {
                const uint8_t * row = w + r*row_bytes;
                if (kernel == 0) {
                    for (int c = 0; c < nc; ++c) {
                        tr->vec_dot(ne, &s[c], 0, row, 0, y[c], 0, 1);
                    }
                } else if (type == GGML_TYPE_IQ2_S) {
                    ggml_vec_dot_iq2_s_q8_K_avx512_nc(ne, s, row, y, nc);
                } else {
                    ggml_vec_dot_q2_0_q8_0_avx512_nc(ne, s, row, y, nc);
                }
                sink += s[0];
            }
            if (sink == 12345.678f) {
                printf("x");
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return n_rows*row_bytes/sec/1e9;
}

// one mul_mat_id of a MoE layer (512 experts of [ne0, ne1], 10 used, 1-4 tokens) through ggml_graph_compute,
// like the CPU split of a layer; reports ms per op and GB/s of the experts read
static void bench_mmid(ggml_type type, int64_t ne0, int64_t ne1, int n_tokens, int n_threads, int reps) {
    const int64_t n_expert = 512, n_used = 10;
    ggml_init_params ip = { (size_t) 64*1024*1024 + ggml_row_size(type, ne0)*ne1*n_expert, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, type, ne0, ne1, n_expert);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    std::mt19937 rng(7);
    uint8_t * wd = (uint8_t *) w->data;
    for (size_t i = 0; i < ggml_nbytes(w); ++i) {
        wd[i] = (uint8_t) rng();
    }
    for (size_t i = 0; i < ggml_nbytes(w); i += ggml_type_size(type)) {
        const uint16_t one = 0x3c00;
        memcpy(wd + i, &one, 2);
    }
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        ((float *) x->data)[i] = (float) (rng() % 2000) / 1000.0f - 1.0f;
    }
    ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    double sec = 0;
    for (int r = 0; r < reps; ++r) {
        // new experts each time, like new tokens
        for (int64_t i = 0; i < ggml_nelements(ids); ++i) {
            ((int32_t *) ids->data)[i] = (int32_t) ((r*37 + i*53) % n_expert);
        }
        const auto t0 = std::chrono::steady_clock::now();
        ggml_graph_compute(gf, &plan);
        sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    const double bytes = (double) ggml_row_size(type, ne0)*ne1*n_used*n_tokens; // upper bound: distinct experts
    printf("  mul_mat_id %s [%lld x %lld] x %d tokens, %2d threads: %.3f ms/op, %.1f GB/s\n", ggml_type_name(type),
        (long long) ne0, (long long) ne1, n_tokens, n_threads, 1000.0*sec/reps, bytes/(sec/reps)/1e9);
    ggml_threadpool_free(tp);
    ggml_free(ctx);
}

// GPU mode: one weight [ne0, ne1] times 64 activation vectors as 64 independent mul_mats in one graph on the first
// GPU backend, so the time per op is the kernel time without the launch of a graph; prints us/op and GB/s
static void bench_gpu_mm(ggml_backend_t be, ggml_type type, int64_t ne0, int64_t ne1, int n_tokens) {
    const int n_ops = 64;
    ggml_init_params ip = { (size_t) (3*n_ops + 8)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    std::vector<ggml_tensor *> xs, outs;
    for (int i = 0; i < n_ops; ++i) {
        xs.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, n_tokens));
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int i = 0; i < n_ops; ++i) {
        outs.push_back(ggml_mul_mat(ctx, w, xs[i]));
        ggml_build_forward_expand(gf, outs.back());
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(3);
    std::vector<float> f(ne0*ne1);
    for (auto & v : f) {
        v = (float) (rng() % 2000) / 1000.0f - 1.0f;
    }
    std::vector<uint8_t> q(ggml_nbytes(w));
    if (type == GGML_TYPE_F32) {
        memcpy(q.data(), f.data(), q.size());
    } else if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) q.data(), ne0*ne1);
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) q.data(), ne0*ne1);
    } else {
        ggml_quantize_chunk(type, f.data(), q.data(), 0, ne1, ne0, nullptr);
    }
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> xv(ne0*n_tokens, 0.5f);
    for (auto * x : xs) {
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
    }
    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(be, gf);
    }
    const int reps = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        ggml_backend_graph_compute(be, gf);
    }
    ggml_backend_synchronize(be);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/(reps*n_ops);
    printf("  %-7s [%6lld x %6lld] x %d: %7.2f us/op, %6.1f GB/s\n", ggml_type_name(type), (long long) ne0, (long long) ne1,
        n_tokens, us, ggml_nbytes(w)/us/1e3);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// GPU q4e expert layers: the GPU tier of build_moe_ffn_q4e (Q4E_ROUTE ids -> MUL_MAT_ID gate, up -> SWIGLU ->
// MUL_MAT_ID down -> Q4E_ROUTE weights -> Q4E_WSUM) for n_layers layers in one graph on the first GPU backend.
// Pools of n_pool blocks of [n_embd, n_ff] (gate/up) and [n_ff, n_embd] (down), 10 experts x 2 blocks per token,
// ~30% of the blocks not in VRAM (id -1), the experts of token t+1 keep half of those of token t (MTP verification).
// Prints us per layer for gate+up+swiglu alone and for the whole layer. With a reference file it writes the outputs
// (first run) or compares against them (later runs, e.g. another build or GGML_CUDA_Q4E_MMID=0).
static void bench_gpu_mmid(ggml_backend_t be, ggml_type type_gu, ggml_type type_down, int n_tokens, int n_pool, const char * ref) {
    const int64_t n_embd = 2560, n_ff = 320, n_blocks = 2, n_used = 10, n_expert = 256;
    const int     n_layers = 48;
    const int64_t n_rows = n_used*n_blocks;
    std::mt19937 rng(11);

    ggml_init_params ip = { (size_t) (24*n_layers + 16)*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, type_gu,   n_embd, n_ff, n_pool);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, type_gu,   n_embd, n_ff, n_pool);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, type_down, n_ff, n_embd, n_pool);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    std::vector<ggml_tensor *> sels, wts, tables, outs, hs;
    ggml_cgraph * gf_gu  = ggml_new_graph(ctx);
    ggml_cgraph * gf_all = ggml_new_graph(ctx);
    for (int l = 0; l < n_layers; ++l) {
        ggml_tensor * sel   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_tensor * w     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tokens);
        ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_expert*n_blocks);
        ggml_tensor * ids   = ggml_q4e_route(ctx, sel, w, table, (int) n_blocks, 0, false, true);
        ggml_tensor * g     = ggml_mul_mat_id(ctx, gate, x, ids);
        ggml_tensor * u     = ggml_mul_mat_id(ctx, up,   x, ids);
        ggml_tensor * h     = ggml_swiglu_split(ctx, g, u);
        ggml_tensor * y     = ggml_mul_mat_id(ctx, down, h, ids);
        ggml_tensor * wr    = ggml_q4e_route(ctx, sel, w, table, (int) n_blocks, 0, true, false);
        ggml_tensor * o     = ggml_q4e_wsum(ctx, y, wr);
        ggml_build_forward_expand(gf_gu, h);
        ggml_build_forward_expand(gf_all, o);
        sels.push_back(sel); wts.push_back(w); tables.push_back(table); outs.push_back(o); hs.push_back(h);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        printf("  out of memory for a pool of %d blocks\n", n_pool);
        ggml_free(ctx);
        return;
    }

    // random quants with sane scales: the fp16 d at the start of a block (iq1_m: the scale in the top nibbles)
    auto fill = [&](ggml_tensor * t) {
        const size_t bs = ggml_type_size(t->type);
        std::vector<uint8_t> q(ggml_nbytes(t)/n_pool*64);
        for (size_t off = 0; off < ggml_nbytes(t); off += q.size()) {
            for (size_t i = 0; i < q.size(); i += 4) {
                const uint32_t v = rng();
                memcpy(&q[i], &v, 4);
            }
            for (size_t i = 0; i < q.size(); i += bs) {
                if (t->type == GGML_TYPE_IQ1_M) {
                    uint16_t * sc = (uint16_t *) (q.data() + i + bs - 8); // 0x2c00 = 1/16
                    sc[0] &= 0x0fff; sc[1] &= 0x0fff; sc[2] = (sc[2] & 0x0fff) | 0xc000; sc[3] = (sc[3] & 0x0fff) | 0x2000;
                } else {
                    const uint16_t d = 0x2c00;
                    memcpy(&q[i], &d, 2);
                }
            }
            ggml_backend_tensor_set(t, q.data(), off, std::min(q.size(), ggml_nbytes(t) - off));
        }
    };
    fill(gate);
    fill(up);
    fill(down);
    {
        std::vector<float> xv(n_embd*n_tokens);
        for (auto & v : xv) {
            v = (float) (rng() % 2000) / 1000.0f - 1.0f;
        }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
    }
    int n_hit = 0, n_ids = 0, n_uniq = 0;
    for (int l = 0; l < n_layers; ++l) {
        std::vector<int32_t> sel(n_used*n_tokens);
        std::vector<float>   w(n_used*n_tokens);
        for (int t = 0; t < n_tokens; ++t) {
            std::vector<int32_t> cur;
            if (t > 0) {
                for (int k = 0; k < n_used; ++k) {
                    if (rng() % 2) {
                        cur.push_back(sel[(t - 1)*n_used + k]);
                    }
                }
            }
            while ((int) cur.size() < n_used) {
                const int32_t e = (int32_t) (rng() % n_expert);
                if (std::find(cur.begin(), cur.end(), e) == cur.end()) {
                    cur.push_back(e);
                }
            }
            for (int k = 0; k < n_used; ++k) {
                sel[t*n_used + k] = cur[k];
                w[t*n_used + k]   = 0.05f + (float) (rng() % 1000) / 4000.0f;
            }
        }
        // distinct random slots for the blocks of this layer, 30% not in VRAM
        std::vector<float> table(n_expert*n_blocks);
        std::vector<int32_t> perm(n_pool);
        for (int i = 0; i < n_pool; ++i) {
            perm[i] = i;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (size_t v = 0; v < table.size(); ++v) {
            table[v] = rng() % 10 < 3 ? -1.0f : (float) perm[v % n_pool];
        }
        std::vector<int32_t> seen;
        for (int t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < n_used; ++k) {
                for (int b = 0; b < n_blocks; ++b) {
                    const int32_t v = sel[t*n_used + k]*(int32_t) n_blocks + b;
                    n_ids++;
                    if (table[v] >= 0.0f) {
                        n_hit++;
                        if (std::find(seen.begin(), seen.end(), v) == seen.end()) {
                            seen.push_back(v);
                            n_uniq++;
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(sels[l], sel.data(), 0, sel.size()*sizeof(int32_t));
        ggml_backend_tensor_set(wts[l], w.data(), 0, w.size()*sizeof(float));
        ggml_backend_tensor_set(tables[l], table.data(), 0, table.size()*sizeof(float));
    }
    // bytes of the distinct hit blocks per layer: the floor of the weight reads
    const double blk_bytes = 2.0*ggml_row_size(type_gu, n_embd)*n_ff + ggml_row_size(type_down, n_ff)*n_embd;
    const double uniq_bytes = blk_bytes*n_uniq/n_layers;

    auto time_graph = [&](ggml_cgraph * gf) {
        for (int i = 0; i < 3; ++i) {
            ggml_backend_graph_compute(be, gf);
        }
        ggml_backend_synchronize(be);
        const int reps = 30;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            ggml_backend_graph_compute(be, gf);
        }
        ggml_backend_synchronize(be);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/(reps*n_layers);
    };
    const double us_gu  = time_graph(gf_gu);
    const double us_all = time_graph(gf_all);
    printf("  %s/%s pool %d, %d tok, %d ids (%.0f%% hit, %.1f distinct blocks/layer): gate+up+swiglu %7.1f us | layer %7.1f us"
        " (down+wsum %6.1f) | %5.1f GB/s of distinct blocks\n",
        ggml_type_name(type_gu), ggml_type_name(type_down), n_pool, n_tokens, (int) (n_rows*n_tokens),
        100.0*n_hit/n_ids, (double) n_uniq/n_layers, us_gu, us_all, us_all - us_gu, uniq_bytes/us_all/1e3);

    if (ref) {
        std::vector<float> res;
        for (int l = 0; l < n_layers; ++l) {
            std::vector<float> o(ggml_nelements(outs[l]));
            ggml_backend_tensor_get(outs[l], o.data(), 0, o.size()*sizeof(float));
            res.insert(res.end(), o.begin(), o.end());
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s.%s.%d.%d", ref, ggml_type_name(type_gu), n_tokens, n_pool);
        FILE * f = fopen(path, "rb");
        if (!f) {
            f = fopen(path, "wb");
            if (f) {
                fwrite(res.data(), sizeof(float), res.size(), f);
                fclose(f);
                printf("    wrote reference %s\n", path);
            }
        } else {
            std::vector<float> r(res.size());
            const size_t n = fread(r.data(), sizeof(float), r.size(), f);
            fclose(f);
            double max_ref = 0.0, max_diff = 0.0;
            int n_bad = 0;
            for (size_t i = 0; i < std::min(n, r.size()); ++i) {
                max_ref  = std::max(max_ref, (double) fabsf(r[i]));
                max_diff = std::max(max_diff, (double) fabsf(r[i] - res[i]));
                n_bad += !std::isfinite(res[i]);
            }
            printf("    vs reference: %zu values, max |ref| %.4g, max |diff| %.4g (%.2e relative), %d non-finite\n",
                n, max_ref, max_diff, max_diff/std::max(max_ref, 1e-30), n_bad);
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "gpu-mmid") == 0) {
        // q4e-kbench gpu-mmid [n_pool] [reference file prefix]
        ggml_backend_load_all();
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) {
            printf("no GPU\n");
            return 1;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        printf("%s\n", ggml_backend_dev_description(dev));
        const int n_pool = argc > 2 ? atoi(argv[2]) : 2048;
        const char * ref = argc > 3 ? argv[3] : nullptr;
        for (ggml_type t : { GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_M }) {
            for (int nt : { 1, 2, 4 }) {
                bench_gpu_mmid(be, t, GGML_TYPE_Q2_0, nt, n_pool, ref);
            }
        }
        ggml_backend_free(be);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "gpu") == 0) {
        ggml_backend_load_all();
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) {
            printf("no GPU\n");
            return 1;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        printf("%s\n", ggml_backend_dev_description(dev));
        const int nt = argc > 2 ? atoi(argv[2]) : 1;
        struct shape { ggml_type t; int64_t a, b; };
        for (shape s : { shape{ GGML_TYPE_Q8_0, 10240, 320 }, shape{ GGML_TYPE_Q8_0, 320, 10240 }, shape{ GGML_TYPE_BF16, 10240, 320 },
                         shape{ GGML_TYPE_BF16, 320, 10240 }, shape{ GGML_TYPE_Q8_0, 10240, 4 }, shape{ GGML_TYPE_BF16, 2560, 512 },
                         shape{ GGML_TYPE_Q8_0, 2560, 512 }, shape{ GGML_TYPE_IQ4_XS, 2560, 4096 }, shape{ GGML_TYPE_IQ4_XS, 4096, 2560 },
                         shape{ GGML_TYPE_Q8_0, 2560, 4096 }, shape{ GGML_TYPE_Q8_0, 2560, 1 }, shape{ GGML_TYPE_BF16, 2560, 1 } }) {
            bench_gpu_mm(be, s.t, s.a, s.b, nt);
        }
        ggml_backend_free(be);
        return 0;
    }
    const size_t mib = argc > 1 ? (size_t) atoi(argv[1]) : 1024;
    ggml_cpu_init();
    if (argc > 2) {
        for (int nt : { 8, 12, 16 }) {
            bench_mmid(GGML_TYPE_IQ2_S, 2560, 640, 1, nt, 400);
            bench_mmid(GGML_TYPE_Q2_0,  640, 2560, 1, nt, 400);
            bench_mmid(GGML_TYPE_IQ2_S, 2560, 640, 4, nt, 200);
        }
        return 0;
    }
    for (ggml_type type : { GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0 }) {
        const int ne = type == GGML_TYPE_IQ2_S ? 2560 : 640;
        const size_t row_bytes = ggml_row_size(type, ne);
        const size_t n_rows = mib*1048576/row_bytes;
        std::vector<uint8_t> w(n_rows*row_bytes);
        std::mt19937 rng(1);
        for (size_t i = 0; i < w.size(); i += 4) {
            const uint32_t v = rng();
            memcpy(&w[i], &v, std::min<size_t>(4, w.size() - i));
        }
        // keep the scales sane: fp16 d of every block = 1.0 (block starts with the half)
        const size_t blk = ggml_type_size(type);
        for (size_t i = 0; i < w.size(); i += blk) {
            const uint16_t one = 0x3c00;
            memcpy(&w[i], &one, 2);
        }
        const ggml_type vt = ggml_get_type_traits_cpu(type)->vec_dot_type;
        std::vector<std::vector<uint8_t>> act(4, std::vector<uint8_t>(ggml_row_size(vt, ne)));
        std::vector<float> x(ne);
        for (auto & v : x) {
            v = (float) (rng() % 2000) / 1000.0f - 1.0f;
        }
        for (auto & a : act) {
            ggml_get_type_traits_cpu(vt)->from_float(x.data(), a.data(), ne);
        }
        printf("%s: %zu MiB of rows of %d, %zu bytes each\n", ggml_type_name(type), mib, ne, row_bytes);
        for (int nt : { 1, 6, 8, 12, 16 }) {
            for (int nc : { 1, 4 }) {
                const double a = run(nt, n_rows, row_bytes, w.data(), ne, act, nc, 0, type);
                const double b = run(nt, n_rows, row_bytes, w.data(), ne, act, nc, 1, type);
                printf("  threads %2d, %d rows of activations: vec_dot %6.1f GB/s | avx512 multi-row %6.1f GB/s\n", nt, nc, a, b);
            }
        }
    }
    return 0;
}
