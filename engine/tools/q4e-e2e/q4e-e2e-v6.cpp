// v6 CPU end-to-end gates for q4e-e2e (A-16.7b): five gates over the common
// v6 bookkeeping components with fake backends - pool assembly, coverage path
// (oracle), pipeline stress, KV sliding window, stats/JSON. Pure CPU, no server.
#include "q4e-e2e-v6.h"

#include "q4e-global-pool.h"
#include "q4e-staging.h"
#include "q4e-h2d-pipeline.h"
#include "q4e-graph-map.h"
#include "q4e-promotion.h"
#include "q4e-kv-paging.h"
#include "q4e-kv-bridge.h"
#include "q4e-stats-v6.h"
#include "q4e-server-v6.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

struct gate_out { bool ok = true; const char * note = ""; };

constexpr int64_t kSlot = 3216896; // one expert slot bytes (~3.07 MB)
constexpr int kGuard = 1000;       // promotion loop iteration cap

void set_backends(q4e_h2d_pipeline & p, bool h2d_fail) {
    p.set_fill_backend([](int, int, void *, int64_t bytes) { return bytes; });
    if (h2d_fail) {
        p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return false; });
    } else {
        p.set_h2d_backend([](int64_t, void *, void *, int64_t) { return true; });
    }
}

gate_out gate1_pool_assembly() {
    gate_out g;
    const q4e_global_pool pool;
    for (int l = 0; l < pool.cfg().n_layers; ++l) {
        if (pool.occupancy_of(l) != 0) { g.ok = false; g.note = "occupancy not zero at init"; return g; }
    }
    const q4e_global_pool_stats s = pool.stats();
    if (s.hits || s.misses || s.evictions || s.rounds || s.m_sum || s.m_max) {
        g.ok = false; g.note = "stats not zero at init"; return g;
    }
    return g;
}

// random router choices -> promotion queue -> run/reap to completion ->
// the staging resident set covers the whole queue (oracle)
gate_out gate2_coverage_path(bool h2d_fail) {
    gate_out g;
    q4e_graph_map map(512, 8, 32);
    std::mt19937 rng(1234567);
    std::vector<int> choices(256);
    for (auto & c : choices) c = (int) (rng() % 512u);
    map.set_router_choices(choices.data(), 256);
    std::vector<int> queue;
    map.promotion_queue(queue);
    if (queue.empty()) { g.ok = false; g.note = "empty promotion queue"; return g; }

    q4e_h2d_pipeline_config pcfg;
    pcfg.l2_ram_mb = 64; pcfg.expert_bytes = (uint64_t) kSlot; pcfg.max_inflight = 4;
    q4e_h2d_pipeline pipe(pcfg);
    set_backends(pipe, h2d_fail);
    q4e_staging_config scfg;
    scfg.total_mb = 4096; // bookkeeping only; one buffer holds the full 512 set
    scfg.chunk_bytes = (uint64_t) kSlot;
    q4e_staging st(scfg);
    q4e_promotion_op op(map, pipe, st, 0, kSlot);

    bool done = false;
    for (int it = 0; !done && it < kGuard; ++it) {
        const q4e_promotion_run pr = op.run();
        (void) op.reap();
        std::vector<int> retry;
        op.deferred_retry(retry);
        done = (pr.n_submitted == 0 && pr.n_busy_deferred == 0 && retry.empty());
    }
    if (!done) { g.ok = false; g.note = h2d_fail ? "h2d failed (injected)" : "promotion loop did not converge"; return g; }
    int n_resident = 0;
    for (const int e : queue) if (op.resident(e)) ++n_resident;
    if (n_resident != (int) queue.size()) { g.ok = false; g.note = "staging resident set does not cover the queue"; return g; }
    return g;
}

// max_inflight 1 with 6 missing experts: busy count is 5+4+3+2+1 = 15, retry to done
gate_out gate3_pipeline_stress(bool h2d_fail) {
    gate_out g;
    q4e_graph_map map(512, 6, 1); // 1 token, top_k 6 -> exactly 6 distinct missing
    const int choices[] = { 10, 11, 12, 13, 14, 15 };
    map.set_router_choices(choices, 6);
    q4e_h2d_pipeline_config pcfg;
    pcfg.l2_ram_mb = 64; pcfg.expert_bytes = (uint64_t) kSlot; pcfg.max_inflight = 1;
    q4e_h2d_pipeline pipe(pcfg);
    set_backends(pipe, h2d_fail);
    q4e_staging st; // default 128 MB A/B, chunk 3216896; holds 6 easily
    q4e_promotion_op op(map, pipe, st, 0, kSlot);

    bool done = false;
    for (int it = 0; !done && it < kGuard; ++it) {
        const q4e_promotion_run pr = op.run();
        (void) op.reap();
        done = (pr.n_submitted == 0 && pr.n_busy_deferred == 0);
    }
    if (!done) { g.ok = false; g.note = h2d_fail ? "h2d failed (injected)" : "stress loop did not converge"; return g; }
    const q4e_h2d_stats ps = pipe.stats();
    if (ps.failed != 0) { g.ok = false; g.note = "expected failed=0"; return g; }
    if (ps.completed != 6) { g.ok = false; g.note = "expected completed=6"; return g; }
    if (ps.busy != 15) { g.ok = false; g.note = "expected busy=15"; return g; }
    int n_resident = 0;
    for (const int e : choices) if (op.resident(e)) ++n_resident;
    if (n_resident != 6) { g.ok = false; g.note = "not all 6 experts resident"; return g; }
    return g;
}

// 300 sliding 4096-token windows over a 100k ctx; kv100k adds 100 large-stride jumps
gate_out gate4_kv_sliding_window(bool kv100k) {
    gate_out g;
    q4e_kv_cfg cfg;
    cfg.ctx_tokens = 100000; cfg.page_tokens = 32; cfg.l1_tokens = 64000;
    cfg.l2_mb = 6000; cfg.kv_bytes_per_token = 1000;
    q4e_kv_paging kv = q4e_kv_bridge::build(cfg);
    const int ctx = cfg.ctx_tokens, span = 4096, n_pages = kv.n_pages();
    auto step = [&](int pos) -> gate_out {
        const q4e_kv_bridge::win w = q4e_kv_bridge::window(ctx, pos, span);
        kv.ensure_resident(w.lo, w.hi);
        for (int t = w.lo; t < w.hi; ++t) {
            if (kv.tier_of(kv.page_of(t)) != q4e_kv_paging::tier::L1) return { false, "window page not L1" };
        }
        if (kv.l1_pages_now() + kv.l2_pages_now() + kv.l3_pages_now() != n_pages) return { false, "tier page sum != n_pages" };
        return { true, "" };
    };
    for (int s = 0; s < 300; ++s) {
        const gate_out r = step(4096 + s * 300); // slide 300 tokens per step, stays < ctx
        if (!r.ok) return r;
    }
    if (kv100k) {
        for (int s = 0; s < 100; ++s) {
            const gate_out r = step(4096 + (s % 2) * 50000); // large-stride jump between two windows
            if (!r.ok) return r;
        }
    }
    return g;
}

// six components -> stats json with the seven group keys + the server v6 body wrapper
gate_out gate5_stats_json() {
    gate_out g;
    q4e_global_pool pool;
    q4e_staging st;
    q4e_h2d_pipeline pipe;
    q4e_graph_map map(8, 2, 1);
    const int choices[] = { 0, 1 };
    map.set_router_choices(choices, 2);
    q4e_promotion_op promo(map, pipe, st, 0, st.chunk_bytes());
    q4e_kv_paging kv(1000, 100, 2, 1, 1000);
    const std::string j = q4e_stats_v6_json(q4e_stats_v6_collect(&pool, &st, &pipe, &map, &promo, &kv, 1234567));
    const char * groups[] = { "\"pool\":", "\"staging\":", "\"pipe\":", "\"map\":", "\"promo\":", "\"kv\":", "\"meta\":" };
    for (const char * key : groups) {
        if (j.find(key) == std::string::npos) { g.ok = false; g.note = "missing json group key"; return g; }
    }
    const q4e_server_v6_cfg cfg6;              // default config
    const q4e_server_v6_state st6(cfg6);       // assembled component set
    if (q4e_server_v6_body(&st6, 777).find("\"v6\":") == std::string::npos) {
        g.ok = false; g.note = "server body missing \"v6\""; return g;
    }
    return g;
}

void run_gate(q4e_e2e_v6_gate & gate, const gate_out & o) {
    gate.ok = o.ok;
    gate.note = o.ok ? "" : o.note;
}

} // namespace

q4e_e2e_v6_result q4e_e2e_v6_run_impl(bool kv100k, bool h2d_fail) {
    q4e_e2e_v6_result r;
    q4e_e2e_v6_gate g1 = { 1, "pool_assembly", false, "" };
    q4e_e2e_v6_gate g2 = { 2, "coverage_path", false, "" };
    q4e_e2e_v6_gate g3 = { 3, "pipeline_stress", false, "" };
    q4e_e2e_v6_gate g4 = { 4, "kv_sliding_window", false, "" };
    q4e_e2e_v6_gate g5 = { 5, "stats_json", false, "" };
    auto safe = [](q4e_e2e_v6_gate & gate, const gate_out & o) {
        try { run_gate(gate, o); }
        catch (const std::exception &) { gate.ok = false; gate.note = "exception"; }
        catch (...) { gate.ok = false; gate.note = "unknown exception"; }
    };
    safe(g1, gate1_pool_assembly());
    safe(g2, gate2_coverage_path(h2d_fail));
    safe(g3, gate3_pipeline_stress(h2d_fail));
    safe(g4, gate4_kv_sliding_window(kv100k));
    safe(g5, gate5_stats_json());
    r.gates = { g1, g2, g3, g4, g5 };
    r.all_ok = r.gates[0].ok && r.gates[1].ok && r.gates[2].ok && r.gates[3].ok && r.gates[4].ok;
    return r;
}

q4e_e2e_v6_result q4e_e2e_v6_run(bool kv100k) {
    return q4e_e2e_v6_run_impl(kv100k, false);
}

q4e_e2e_v6_result q4e_e2e_v6_run_h2d_fail(bool kv100k) {
    return q4e_e2e_v6_run_impl(kv100k, true);
}

std::string q4e_e2e_v6_text(const q4e_e2e_v6_result & r) {
    std::string out;
    for (const q4e_e2e_v6_gate & g : r.gates) {
        out += "gate " + std::to_string(g.id) + " " + g.name + " " + (g.ok ? "ok" : "FAIL") + "\n";
        if (g.note && g.note[0]) out += "  note: " + std::string(g.note) + "\n";
    }
    out += "all_ok=" + std::string(r.all_ok ? "true" : "false") + "\n";
    return out;
}
