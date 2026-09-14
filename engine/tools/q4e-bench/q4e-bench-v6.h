// v6 component throughput micro-benchmark for q4e-bench (A-16.7a)
// Five small-config sections over the common v6 bookkeeping components:
// pool insert, staging place, h2d submit/poll, graph map promotion queue, kv ensure.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct q4e_bench_v6_opts {
    int  iters = 2000;  // per-section loop count, >= 1
    bool kv100k = false; // kv section: 300-step sliding window over a 100k ctx
};

struct q4e_bench_v6_section {
    std::string name;
    double  ops_per_sec = 0.0;
    int64_t n_ops = 0;
    double  elapsed_ms = 0.0;
};

struct q4e_bench_v6_result {
    bool ok = false; // all sections completed
    std::string note;
    std::vector<q4e_bench_v6_section> sections; // fixed order: pool, staging, pipe, map, kv
};

// Runs all five sections. A section failure stops the run: ok = false, note = section name + reason.
// Throws std::invalid_argument when iters < 1.
q4e_bench_v6_result q4e_bench_v6_run(const q4e_bench_v6_opts & opt);

// Human-readable multi-line report: section names, ops/sec, ok/note.
std::string q4e_bench_v6_text(const q4e_bench_v6_result & r);
