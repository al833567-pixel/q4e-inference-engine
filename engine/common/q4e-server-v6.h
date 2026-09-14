// v6 /q4e/stats server wiring (A-16.6b): assemble the six stats components from one cfg
// and expose a handler shim over q4e-stats-v6. Pure C++, no httplib include:
// the server only calls q4e_server_v6_http / q4e_server_v6_body.
#pragma once

#include "q4e-global-pool.h"
#include "q4e-graph-map.h"
#include "q4e-h2d-pipeline.h"
#include "q4e-kv-paging.h"
#include "q4e-promotion.h"
#include "q4e-staging.h"

#include <cstdint>
#include <memory>
#include <string>

// cfg -> component wiring; defaults per 01 spec
// kv_bytes_per_token / expert_bytes must be > 0; the staging chunk slot equals expert_bytes
struct q4e_server_v6_cfg {
    int     n_experts      = 512;     // experts per layer
    int     n_layers       = 48;      // MoE layers
    int     moe_vram       = 5000;    // L1 global pool slots; 0 = pool off
    int     layer_floor    = 10;      // min residents per layer
    int     staging_mb     = 128;     // A/B staging buffer MB; 0 = off
    int     moe_ram        = 4000;    // L2 pinned expert pool MB; 0 = off
    int     max_inflight   = 4;       // in-flight H2D cap
    int     kv_ctx         = 100000;  // KV context tokens
    int     kv_page_tokens = 32;      // tokens per KV page
    int     kv_l1_tokens   = 64000;   // L1 KV token budget
    int     kv_l2_mb       = 6000;    // L2 KV RAM MB; 0 = off
    int64_t kv_bytes_per_token = 16384;   // model KV bytes per token, > 0
    int64_t expert_bytes         = 3216896; // one expert slot bytes, > 0
};

// Assembled component set; non-copyable (q4e_promotion_op holds references).
// ctor propagates std::invalid_argument from any component ctor on an illegal cfg
class q4e_server_v6_state {
public:
    explicit q4e_server_v6_state(const q4e_server_v6_cfg & cfg);

    // raw stats json (q4e_stats_v6_json of the six components); the {"v6": ...}
    // wrapper is applied by q4e_server_v6_body, not here
    std::string stats_json(int64_t now_us) const;

    // moe_vram > 0 (L1 expert pool on) or kv_l2_mb > 0 (KV paging on)
    bool enabled() const;

    const q4e_server_v6_cfg & cfg() const { return cfg_; }

    // component access (the engine binds backends / router choices through these)
    q4e_global_pool & pool() { return pool_; }
    q4e_staging & staging() { return staging_; }
    q4e_h2d_pipeline & pipe() { return pipe_; }
    q4e_graph_map & map() { return map_; }
    q4e_promotion_op & promo() { return *promo_; }
    q4e_kv_paging & kv() { return kv_; }

    // reset every component (stats cleared; kv pages back to L3)
    void reset();

private:
    q4e_global_pool pool_;
    q4e_staging staging_;
    q4e_h2d_pipeline pipe_;
    q4e_graph_map map_;
    std::unique_ptr<q4e_promotion_op> promo_;
    q4e_kv_paging kv_;
    q4e_server_v6_cfg cfg_;
};

// /q4e/stats contract: q4e_server_v6_body returns the FULL response body directly:
//   valid state -> {"v6": <stats json>}
//   null state  -> empty body; the server maps it to HTTP 500
// returns 200 on a valid state, 500 on a null state
int q4e_server_v6_http(const q4e_server_v6_state * st);

// full body for the status above; now_us is injected by the server (no clock read here)
// null state -> empty body
std::string q4e_server_v6_body(const q4e_server_v6_state * st, int64_t now_us);
