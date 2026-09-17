#include "q4e-server-v6.h"

#include "q4e-global-pool.h"
#include "q4e-graph-map.h"
#include "q4e-h2d-pipeline.h"
#include "q4e-kv-bridge.h"
#include "q4e-promotion.h"
#include "q4e-staging.h"
#include "q4e-stats-v6.h"

#include <stdexcept>

namespace {

q4e_global_pool make_pool(const q4e_server_v6_cfg & cfg) {
    q4e_global_pool_config pc;
    pc.n_experts_per_layer = cfg.n_experts;
    pc.n_layers            = cfg.n_layers;
    pc.n_slots             = cfg.moe_vram;
    pc.floor               = cfg.layer_floor;
    return q4e_global_pool(pc);
}

q4e_staging make_staging(const q4e_server_v6_cfg & cfg) {
    q4e_staging_config sc;
    sc.total_mb    = (uint32_t) cfg.staging_mb;
    sc.chunk_bytes = (uint64_t) cfg.expert_bytes; // q4e_promotion_op requires expert_bytes <= chunk
    return q4e_staging(sc);
}

q4e_h2d_pipeline make_pipe(const q4e_server_v6_cfg & cfg) {
    q4e_h2d_pipeline_config hc;
    hc.l2_ram_mb    = (uint32_t) cfg.moe_ram;
    hc.expert_bytes = (uint64_t) cfg.expert_bytes;
    hc.max_inflight = (uint32_t) cfg.max_inflight;
    return q4e_h2d_pipeline(hc);
}

q4e_graph_map make_map(const q4e_server_v6_cfg & cfg) {
    // stats-only shim: one router token; top_k never read by the stats path
    return q4e_graph_map(cfg.n_experts, 8, 1);
}

q4e_kv_paging make_kv(const q4e_server_v6_cfg & cfg) {
    q4e_kv_cfg kc;
    kc.ctx_tokens         = cfg.kv_ctx;
    kc.page_tokens        = cfg.kv_page_tokens;
    kc.l1_tokens          = cfg.kv_l1_tokens;
    kc.l2_mb              = cfg.kv_l2_mb;
    kc.kv_bytes_per_token = cfg.kv_bytes_per_token;
    return q4e_kv_bridge::build(kc);
}

} // namespace

q4e_server_v6_state::q4e_server_v6_state(const q4e_server_v6_cfg & cfg)
    : pool_(make_pool(cfg)),
      staging_(make_staging(cfg)),
      pipe_(make_pipe(cfg)),
      map_(make_map(cfg)),
      promo_(std::make_unique<q4e_promotion_op>(map_, pipe_, staging_, 0, cfg.expert_bytes)),
      kv_(make_kv(cfg)),
      cfg_(cfg) {
}

std::string q4e_server_v6_state::stats_json(int64_t now_us) const {
    return q4e_stats_v6_json(q4e_stats_v6_collect(&pool_, &staging_, &pipe_, &map_, promo_.get(), &kv_, now_us));
}

bool q4e_server_v6_state::enabled() const {
    return cfg_.moe_vram > 0 || cfg_.kv_l2_mb > 0;
}

void q4e_server_v6_state::reset() {
    pool_.reset();
    staging_.reset();
    pipe_.reset();
    map_.reset();
    promo_.reset();
    kv_.reset();
}

int q4e_server_v6_http(const q4e_server_v6_state * st) {
    return st ? 200 : 500;
}

std::string q4e_server_v6_body(const q4e_server_v6_state * st, int64_t now_us) {
    if (!st) {
        return std::string();
    }
    return "{\"v6\": " + st->stats_json(now_us) + "}";
}
