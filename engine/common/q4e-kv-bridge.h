// v6 KV bridge: common_params v6 fields -> q4e_kv_paging assembly (A-16.5b-1).
// Pure C++, no common.h include; params are passed explicitly via q4e_kv_cfg.
// Defaults per 01 spec: L1 64k tokens, L2 32k tokens (--q4e-kv-ram 6000MB,
// page count derived from model kv_bytes per token), L3 = remaining SSD.
#pragma once

#include "q4e-kv-paging.h"

#include <cstdint>

struct q4e_kv_cfg {
    int      ctx_tokens       = 0;    // context size in tokens
    int      page_tokens      = 0;    // fixed tokens per KV page
    int      l1_tokens        = 0;    // L1 VRAM token budget; <= 0 = fully resident (paging off)
    int      l2_mb            = 0;    // L2 pinned RAM budget in MB; 0 = L2 off
    int64_t  kv_bytes_per_token = 0;  // model KV cache bytes per token
};

class q4e_kv_bridge {
public:
    struct win { int lo; int hi; };

    // l2_pages = min(floor(l2_mb * 1048576 / page_bytes), n_pages); l2_mb = 0 -> 0
    // l1_pages = min(ceil(l1_tokens / page_tokens), n_pages); l1_tokens <= 0 -> n_pages
    // throws std::invalid_argument: ctx_tokens <= 0, page_tokens <= 0,
    // kv_bytes_per_token <= 0, l2_mb < 0
    static q4e_kv_paging build(const q4e_kv_cfg & cfg);

    // decode sliding window: [pos - span + 1, pos + 1) clamped to [0, ctx_tokens)
    // hi is exclusive, so the result feeds ensure_resident directly
    // throws std::invalid_argument: ctx_tokens <= 0, span <= 0, pos outside [0, ctx_tokens)
    static win window(int ctx_tokens, int pos, int span);
};
