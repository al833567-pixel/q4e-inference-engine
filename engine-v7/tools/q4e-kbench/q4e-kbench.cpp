// q4e-kbench: throughput of the CPU expert dot kernels on this machine, independent of the graph and the scheduler.
// It streams N MiB of weight rows (random bytes in the layout of the type) against q8 activation rows with T threads
// and prints GB/s of weights, for ggml's vec_dot and for the multi-row kernels.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
// cycles this thread ran: robust to a busy machine (other processes preempting the bench)
static uint64_t thread_cycles() { ULONG64 c = 0; QueryThreadCycleTime(GetCurrentThread(), &c); return c; }
#else
static uint64_t thread_cycles() { return 0; }
#endif

extern "C" {
void ggml_vec_dot_iq2_s_q8_K_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc);
void ggml_vec_dot_q2_0_q8_0_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc);
void ggml_vec_dot_iq2_s_q8_K_avx512_nc_ref(int n, float * s, const void * vx, const void * const * vy, int nc);
void ggml_vec_dot_iq2_xxs_q8_K_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc);
void ggml_vec_dot_iq2_xxs_q8_K_avx512_nc_ref(int n, float * s, const void * vx, const void * const * vy, int nc);
}

typedef void (*nc_kernel_t)(int n, float * s, const void * vx, const void * const * vy, int nc);

// thread cycles of a dependent chain of 1-cycle adds: the ratio of the cycle counter to core clocks
static void calibrate() {
    volatile uint64_t seed = 1;
    uint64_t x = seed;
    const uint64_t n = 200000000;
    const uint64_t c0 = thread_cycles();
    for (uint64_t i = 0; i < n; ++i) {
        x += i;
        __asm__ volatile("" : "+r"(x));
    }
    const uint64_t c1 = thread_cycles();
    seed = x;
    printf("calibration: %.2f counter cycles per core cycle (dependent adds)\n", (double) (c1 - c0)/n);
}

// two multi-row kernels of the same type: the results must be bit-identical, the throughput is measured
// interleaved (the machine may be busy) over n_mib of random weight rows
static void bench_cmp(const char * name, ggml_type type, int ne, nc_kernel_t a, nc_kernel_t b, size_t n_mib, int n_threads) {
    const size_t row_bytes = ggml_row_size(type, ne);
    const size_t n_rows = n_mib*1048576/row_bytes;
    std::vector<uint8_t> w(n_rows*row_bytes);
    std::mt19937 rng(11);
    for (size_t i = 0; i < w.size(); i += 4) {
        const uint32_t v = rng();
        memcpy(&w[i], &v, std::min<size_t>(4, w.size() - i));
    }
    for (size_t i = 0; i < w.size(); i += ggml_type_size(type)) {
        const uint16_t one = 0x3c00;
        memcpy(&w[i], &one, 2);
    }
    const ggml_type vt = ggml_get_type_traits_cpu(type)->vec_dot_type;
    std::vector<std::vector<uint8_t>> act(4, std::vector<uint8_t>(ggml_row_size(vt, ne)));
    std::vector<float> x(ne);
    for (auto & v : act) {
        for (auto & f : x) {
            f = (float) (rng() % 2000) / 1000.0f - 1.0f;
        }
        ggml_get_type_traits_cpu(vt)->from_float(x.data(), v.data(), ne);
    }
    const void * y[4] = { act[0].data(), act[1].data(), act[2].data(), act[3].data() };
    // exactness on the first rows
    size_t n_diff = 0;
    for (int nc = 1; nc <= 4; ++nc) {
        for (size_t r = 0; r < std::min<size_t>(n_rows, 4096); ++r) {
            float sa[4], sb[4];
            a(ne, sa, w.data() + r*row_bytes, y, nc);
            b(ne, sb, w.data() + r*row_bytes, y, nc);
            n_diff += memcmp(sa, sb, nc*sizeof(float)) != 0;
        }
    }
    printf("%s: %zu rows of %d, %s\n", name, n_rows, ne, n_diff ? "RESULTS DIFFER" : "bit-identical");
    for (int nc : { 1, 2, 4 }) {
        double best[2] = { 0, 0 };
        double cyc[2] = { 1e30, 1e30 };
        const size_t inner = std::max<size_t>(1, 64/n_mib);
        for (int rep = 0; rep < 5; ++rep) {
            for (int k = 0; k < 2; ++k) {
                nc_kernel_t f = k == 0 ? a : b;
                std::vector<std::thread> th;
                std::atomic<uint64_t> cycles{0};
                const auto t0 = std::chrono::steady_clock::now();
                for (int t = 0; t < n_threads; ++t) {
                    th.emplace_back([&, t]() {
                        float s[4];
                        float sink = 0;
                        const uint64_t c0 = thread_cycles();
                        for (size_t it = 0; it < inner; ++it) {
                            for (size_t r = n_rows*t/n_threads; r < n_rows*(t + 1)/n_threads; ++r) {
                                f(ne, s, w.data() + r*row_bytes, y, nc);
                                sink += s[0];
                            }
                        }
                        cycles += thread_cycles() - c0;
                        if (sink == 12345.678f) {
                            printf("x");
                        }
                    });
                }
                for (auto & x : th) {
                    x.join();
                }
                cyc[k] = std::min(cyc[k], (double) cycles/(inner*n_rows*(ne/256.0)));
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                best[k] = std::max(best[k], inner*n_rows*row_bytes/sec/1e9);
            }
        }
        printf("  %d threads, %d activation rows: old %6.1f GB/s %6.1f cyc/256 | new %6.1f GB/s %6.1f cyc/256 (best of 5)\n", n_threads, nc, best[0], cyc[0], best[1], cyc[1]);
    }
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

// the CPU tier of a q4e layer: mul_mat_id gate/up + swiglu + mul_mat_id down + q4e_wsum against GGML_OP_Q4E_EXPERTS,
// n_tokens tokens of 10 experts x 2 blocks of [2560 x 320], about n_miss of their rows on the CPU (the rest -1, as if
// in VRAM), new random blocks each repetition (cold in the caches). Checks that both give the same bits.
static void bench_fused(ggml_type type_gu, ggml_type type_d, int n_tokens, int n_miss, int n_threads, int reps) {
    const int64_t n_embd = 2560, n_ff = 320, n_as = 1024, n_rows = 20;
    ggml_init_params ip = { (size_t) 256*1024*1024 + 2*ggml_row_size(type_gu, n_embd)*n_ff*n_as + ggml_row_size(type_d, n_ff)*n_embd*n_as, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, type_gu, n_embd, n_ff, n_as);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, type_gu, n_embd, n_ff, n_as);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, type_d, n_ff, n_embd, n_as);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_rows, n_tokens);
    ggml_tensor * wt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_rows, n_tokens);
    std::mt19937 rng(5);
    for (ggml_tensor * t : { gate, up, down }) {
        uint8_t * d = (uint8_t *) t->data;
        for (size_t i = 0; i < ggml_nbytes(t); i += 4) {
            const uint32_t v = rng();
            memcpy(d + i, &v, std::min<size_t>(4, ggml_nbytes(t) - i));
        }
        const uint16_t scale = 0x2000; // ~0.0078
        for (size_t i = 0; i < ggml_nbytes(t); i += ggml_type_size(t->type)) {
            memcpy(d + i, &scale, 2);
        }
    }
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        ((float *) x->data)[i] = (float) (rng() % 2000) / 1000.0f - 1.0f;
    }

    // separate ops
    ggml_tensor * g  = ggml_mul_mat_id(ctx, gate, x, ids);
    ggml_tensor * u  = ggml_mul_mat_id(ctx, up,   x, ids);
    ggml_tensor * h  = ggml_swiglu_split(ctx, g, u);
    ggml_tensor * y  = ggml_mul_mat_id(ctx, down, h, ids);
    ggml_tensor * o1 = ggml_q4e_wsum(ctx, y, wt);
    ggml_cgraph * gf1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf1, o1);
    // fused
    ggml_tensor * o2 = ggml_q4e_experts(ctx, gate, up, down, x, ids, wt);
    ggml_cgraph * gf2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf2, o2);

    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    ggml_cplan p1 = ggml_graph_plan(gf1, n_threads, tp);
    ggml_cplan p2 = ggml_graph_plan(gf2, n_threads, tp);
    std::vector<uint8_t> w1(p1.work_size), w2(p2.work_size);
    p1.work_data = w1.data();
    p2.work_data = w2.data();

    std::vector<double> t1, t2;
    size_t n_diff = 0, n_blocks = 0, bytes = 0;
    for (int r = 0; r < reps; ++r) {
        // n_miss rows of the n_tokens*n_rows on the CPU: tokens share some experts (as in a verify batch)
        int32_t * id = (int32_t *) ids->data;
        float * wv = (float *) wt->data;
        std::vector<int32_t> pool;
        for (int i = 0; i < n_miss; ++i) {
            pool.push_back((int32_t) (rng() % n_as));
        }
        std::vector<char> used(n_as, 0);
        for (int64_t i = 0; i < n_rows*n_tokens; ++i) {
            id[i] = -1;
            wv[i] = 0.0f;
        }
        for (int i = 0; i < n_miss; ++i) {
            const int64_t t = rng() % n_tokens, k = rng() % n_rows;
            id[t*n_rows + k] = pool[rng() % pool.size()];
            wv[t*n_rows + k] = (float) (rng() % 1000 + 1) / 4000.0f;
        }
        for (int64_t i = 0; i < n_rows*n_tokens; ++i) {
            if (id[i] >= 0 && !used[id[i]]) {
                used[id[i]] = 1;
                ++n_blocks;
                bytes += 2*ggml_row_size(type_gu, n_embd)*n_ff + ggml_row_size(type_d, n_ff)*n_embd;
            }
        }
        for (int k = 0; k < 2; ++k) {
            const bool fused = (r + k) % 2 == 1;
            const auto t0 = std::chrono::steady_clock::now();
            ggml_graph_compute(fused ? gf2 : gf1, fused ? &p2 : &p1);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            (fused ? t2 : t1).push_back(us);
        }
        // random IQ1_M blocks may carry inf/nan scales: a nan is equal to a nan here
        for (int64_t i = 0; i < ggml_nelements(o1); ++i) {
            const float a = ((const float *) o1->data)[i], b = ((const float *) o2->data)[i];
            if (memcmp(&a, &b, 4) != 0 && !(std::isnan(a) && std::isnan(b))) {
                if (n_diff++ == 0) {
                    printf("    first difference at %lld: %.9g vs %.9g\n", (long long) i, a, b);
                }
                break;
            }
        }
    }
    std::sort(t1.begin(), t1.end());
    std::sort(t2.begin(), t2.end());
    const double mb = (double) bytes/reps;
    printf("  %s/%s %d tokens, %.1f blocks/layer, %d threads: separate %7.1f us (min %7.1f, %5.1f GB/s) | fused %7.1f us (min %7.1f, %5.1f GB/s) | %s\n",
        ggml_type_name(type_gu), ggml_type_name(type_d), n_tokens, (double) n_blocks/reps, n_threads,
        t1[t1.size()/2], t1[0], mb/t1[0]/1e3, t2[t2.size()/2], t2[0], mb/t2[0]/1e3, n_diff ? "OUTPUTS DIFFER" : "bit-identical");
    ggml_threadpool_free(tp);
    ggml_free(ctx);
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

int main(int argc, char ** argv) {
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
                         shape{ GGML_TYPE_Q8_0, 2560, 4096 }, shape{ GGML_TYPE_Q8_0, 2560, 1 }, shape{ GGML_TYPE_BF16, 2560, 1 },
                         shape{ GGML_TYPE_IQ2_S, 2560, 4096 }, shape{ GGML_TYPE_IQ2_S, 2560, 320 }, shape{ GGML_TYPE_Q2_0, 320, 2560 }, shape{ GGML_TYPE_Q2_0, 2560, 4096 } }) {
            bench_gpu_mm(be, s.t, s.a, s.b, nt);
        }
        ggml_backend_free(be);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "fused") == 0) {
        // q4e-kbench fused [threads] [reps]
        ggml_cpu_init();
        const int nt   = argc > 2 ? atoi(argv[2]) : 8;
        const int reps = argc > 3 ? atoi(argv[3]) : 200;
        bench_fused(GGML_TYPE_IQ2_S,   GGML_TYPE_Q2_0, 4, 20, nt, reps);
        bench_fused(GGML_TYPE_IQ2_S,   GGML_TYPE_Q2_0, 1, 6,  nt, reps);
        bench_fused(GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, 4, 20, nt, reps);
        bench_fused(GGML_TYPE_IQ1_M,   GGML_TYPE_Q2_0, 4, 20, nt, reps);
        bench_fused(GGML_TYPE_IQ2_S,   GGML_TYPE_Q2_0, 3, 1,  nt, reps);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "cmp") == 0) {
        ggml_cpu_init();
        const int nt = argc > 2 ? atoi(argv[2]) : 8;
        const size_t cmp_mib = argc > 3 ? (size_t) atoi(argv[3]) : 256;
        calibrate();
        bench_cmp("iq2_s",   GGML_TYPE_IQ2_S,   2560, ggml_vec_dot_iq2_s_q8_K_avx512_nc_ref,   ggml_vec_dot_iq2_s_q8_K_avx512_nc,   cmp_mib, nt);
        bench_cmp("iq2_xxs", GGML_TYPE_IQ2_XXS, 2560, ggml_vec_dot_iq2_xxs_q8_K_avx512_nc_ref, ggml_vec_dot_iq2_xxs_q8_K_avx512_nc, cmp_mib, nt);
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
