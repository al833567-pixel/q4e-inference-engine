// q4e-bench v6 micro-benchmark implementation (A-16.7a)
#include "q4e-bench-v6.h"

#include "q4e-global-pool.h"
#include "q4e-staging.h"
#include "q4e-h2d-pipeline.h"
#include "q4e-graph-map.h"
#include "q4e-kv-paging.h"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

using clock = std::chrono::steady_clock;

using bench_fn = int (*)(int iters, bool kv100k);

int bench_pool_insert(int iters, bool kv100k);
int bench_staging_place(int iters, bool kv100k);
int bench_pipe_submit_poll(int iters, bool kv100k);
int bench_map_promo_queue(int iters, bool kv100k);
int bench_kv_ensure(int iters, bool kv100k);

struct section_def {
    const char * name;
    bench_fn    fn;
};

const section_def kSections[] = {
    { "pool_insert",      &bench_pool_insert },
    { "staging_place",    &bench_staging_place },
    { "pipe_submit_poll", &bench_pipe_submit_poll },
    { "map_promo_queue",  &bench_map_promo_queue },
    { "kv_ensure",        &bench_kv_ensure },
};

// pool: 2 layers x 8 experts, 16 global slots, floor 2 per layer
int bench_pool_insert(int iters, bool) {
    q4e_global_pool_config cfg;
    cfg.n_experts_per_layer = 8;
    cfg.n_layers = 2;
    cfg.n_slots = 16;
    cfg.floor = 2;
    q4e_global_pool pool(cfg);
    for (int i = 0; i < iters; ++i) {
        pool.insert(i % 2, i % 8);
    }
    return iters;
}

// staging: 4 MB total, 512 KB chunk slots
int bench_staging_place(int iters, bool) {
    q4e_staging_config cfg;
    cfg.total_mb = 4;
    cfg.chunk_bytes = 512 * 1024;
    q4e_staging st(cfg);
    for (int i = 0; i < iters; ++i) {
        const int buf = i % 2;
        st.begin_fill(buf);
        const int64_t off = st.place(buf, 128 * 1024);
        if (off < 0) {
            throw std::runtime_error("staging place overflow");
        }
        st.publish(buf);
        st.consume_begin(buf);
        st.release(buf);
    }
    return iters;
}

// pipe: L2 8 MB, 256 KB expert slots, 4 in-flight
int bench_pipe_submit_poll(int iters, bool) {
    q4e_h2d_pipeline_config cfg;
    cfg.l2_ram_mb = 8;
    cfg.expert_bytes = 256 * 1024;
    cfg.max_inflight = 4;
    q4e_h2d_pipeline pipe(cfg);
    pipe.set_fill_backend([](int, int, void *, int64_t b) { return b; });
    pipe.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });
    for (int i = 0; i < iters; ++i) {
        (void) pipe.submit(i % 2, i % 8);
        (void) pipe.poll();
    }
    return iters;
}

// map: 512 experts, top_k 8, 16 tokens
int bench_map_promo_queue(int iters, bool) {
    constexpr int n_experts = 512;
    constexpr int top_k = 8;
    constexpr int n_tokens = 16;
    q4e_graph_map map(n_experts, top_k, n_tokens);
    std::vector<int> ids(n_tokens * top_k);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = int(i % n_experts);
    }
    const int staged[4] = { 0, 1, 2, 3 };
    std::vector<int> promo;
    for (int i = 0; i < iters; ++i) {
        map.set_router_choices(ids.data(), int(ids.size()));
        map.set_staged(staged, 4);
        map.promotion_queue(promo);
    }
    return iters;
}

// kv: ctx 100k, page 32 tokens, L1 2000 pages, L2 3000 pages
// kv100k = 300-step sliding window (4096 tokens) across the full ctx
int bench_kv_ensure(int iters, bool kv100k) {
    q4e_kv_paging kv(100000, 32, 2000, 3000, 65536);
    if (!kv100k) {
        for (int i = 0; i < iters; ++i) {
            const int lo = (i * 64) % (100000 - 256);
            kv.ensure_resident(lo, lo + 256);
        }
        return iters;
    }
    constexpr int steps = 300;
    const int win = 4096;
    const int stride = (100000 - win) / (steps - 1);
    for (int s = 0; s < steps; ++s) {
        const int lo = s * stride;
        kv.ensure_resident(lo, lo + win);
    }
    return steps;
}

} // namespace

q4e_bench_v6_result q4e_bench_v6_run(const q4e_bench_v6_opts & opt) {
    if (opt.iters < 1) {
        throw std::invalid_argument("q4e_bench_v6_run: iters must be >= 1");
    }
    q4e_bench_v6_result r;
    for (const section_def & d : kSections) {
        q4e_bench_v6_section s;
        s.name = d.name;
        try {
            const auto t0 = clock::now();
            const int n_ops = d.fn(opt.iters, opt.kv100k);
            const auto t1 = clock::now();
            s.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            s.n_ops = n_ops;
            s.ops_per_sec = s.elapsed_ms > 0.0 ? double(n_ops) / (s.elapsed_ms * 1e-3) : 1e9;
        } catch (const std::exception & e) {
            r.ok = false;
            r.note = d.name;
            r.note += ": ";
            r.note += e.what();
            r.sections.push_back(std::move(s));
            return r;
        }
        r.sections.push_back(std::move(s));
    }
    r.ok = true;
    r.note = "all sections ok";
    return r;
}

std::string q4e_bench_v6_text(const q4e_bench_v6_result & r) {
    std::ostringstream os;
    os << "q4e-bench v6 sections (small configs, bookkeeping throughput)\n";
    for (const q4e_bench_v6_section & s : r.sections) {
        os << "  " << s.name << ": ops/sec=" << s.ops_per_sec
           << "  n_ops=" << s.n_ops
           << "  elapsed_ms=" << s.elapsed_ms << "\n";
    }
    os << "ok=" << (r.ok ? "true" : "false") << "  note=" << r.note << "\n";
    return os.str();
}
