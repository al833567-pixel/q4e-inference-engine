#include "q4e-kv-bridge.h"

#include <algorithm>
#include <stdexcept>

namespace {
constexpr int64_t MB = 1048576;
}

q4e_kv_paging q4e_kv_bridge::build(const q4e_kv_cfg & cfg) {
    if (cfg.ctx_tokens <= 0)           throw std::invalid_argument("q4e_kv_bridge::build: ctx_tokens <= 0");
    if (cfg.page_tokens <= 0)          throw std::invalid_argument("q4e_kv_bridge::build: page_tokens <= 0");
    if (cfg.kv_bytes_per_token <= 0)   throw std::invalid_argument("q4e_kv_bridge::build: kv_bytes_per_token <= 0");
    if (cfg.l2_mb < 0)                 throw std::invalid_argument("q4e_kv_bridge::build: l2_mb < 0");

    const int n_pages = (cfg.ctx_tokens + cfg.page_tokens - 1) / cfg.page_tokens;
    const int64_t page_bytes = (int64_t) cfg.page_tokens * cfg.kv_bytes_per_token;

    int l2_pages = 0; // l2_mb = 0 -> L2 off
    if (cfg.l2_mb > 0) {
        l2_pages = (int) std::min<int64_t>((int64_t) cfg.l2_mb * MB / page_bytes, (int64_t) n_pages);
    }

    int l1_pages = n_pages; // l1_tokens <= 0 -> fully resident
    if (cfg.l1_tokens > 0) {
        l1_pages = (int) std::min<int64_t>((cfg.l1_tokens + cfg.page_tokens - 1) / cfg.page_tokens, (int64_t) n_pages);
    }

    return q4e_kv_paging(cfg.ctx_tokens, cfg.page_tokens, l1_pages, l2_pages, (uint64_t) page_bytes);
}

q4e_kv_bridge::win q4e_kv_bridge::window(int ctx_tokens, int pos, int span) {
    if (ctx_tokens <= 0) throw std::invalid_argument("q4e_kv_bridge::window: ctx_tokens <= 0");
    if (span <= 0)       throw std::invalid_argument("q4e_kv_bridge::window: span <= 0");
    if (pos < 0 || pos >= ctx_tokens) throw std::invalid_argument("q4e_kv_bridge::window: pos out of range");

    const int lo = std::max(0, pos - span + 1);
    return { lo, pos + 1 };
}
