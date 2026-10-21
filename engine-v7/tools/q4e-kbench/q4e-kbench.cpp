// q4e-kbench: throughput of the CPU expert dot kernels on this machine, independent of the graph and the scheduler.
// It streams N MiB of weight rows (random bytes in the layout of the type) against q8 activation rows with T threads
// and prints GB/s of weights, for ggml's vec_dot and for the multi-row kernels.
#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
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

int main(int argc, char ** argv) {
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
