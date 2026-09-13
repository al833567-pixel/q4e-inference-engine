// v6 100k KV cache paging manager: fixed-token pages across three tiers (A-16.5a)
// L1 = VRAM active window, L2 = pinned RAM, L3 = SSD backing. Each page holds
// exactly one tier; L1/L2 keep LRU order (list front = LRU). Initial: all L3.
// Pure C++ bookkeeping; physical moves go through an injectable tier backend.
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct q4e_kv_paging_stats {
    uint64_t promote_l2_to_l1 = 0;
    uint64_t promote_l3_to_l1 = 0;
    uint64_t demote_l1_to_l2  = 0;
    uint64_t demote_l1_to_l3  = 0;
    uint64_t demote_l2_to_l3  = 0;
};

class q4e_kv_paging {
public:
    enum class tier : int { L1 = 0, L2 = 1, L3 = 2 };

    // n_pages = ceil(ctx_tokens / page_tokens); the last page holds the remainder
    // ctx_tokens > 0, page_tokens > 0, l1_pages > 0 (clamped to n_pages if larger),
    // l2_pages >= 0 (0 = L2 off), kv_bytes_per_page > 0
    // throws std::invalid_argument on any illegal value
    q4e_kv_paging(int ctx_tokens, int page_tokens, int l1_pages, int l2_pages, uint64_t kv_bytes_per_page);

    // physical move hook: return false -> std::runtime_error; unset = bookkeeping only
    void set_tier_backend(std::function<bool(int page, int from, int to)> backend);

    // token -> page id, token in [0, ctx_tokens); throws std::out_of_range
    int page_of(int token) const;

    struct token_range { int lo; int hi; };
    // page -> [lo, hi) token range, hi clamped to ctx_tokens; throws std::out_of_range
    token_range token_range_of(int page) const;

    // all pages of [lo, hi) end up in L1. Promotion order: L2 pages in L2 LRU order,
    // then L3 pages ascending. L1 full: LRU L1 page demoted to L2; L2 full: its LRU
    // page goes to L3 first; L2 off: directly to L3.
    // throws std::invalid_argument: bad range or more pages than the L1 capacity
    void ensure_resident(int token_lo, int token_hi);

    // L1 pages of [lo, hi) demoted to L2 in L1 LRU order; L2 full: chain to L3 (L2 off: L3)
    // throws std::invalid_argument: bad range
    void demote_range(int token_lo, int token_hi);

    tier tier_of(int page) const; // throws std::out_of_range
    int  l1_pages_now() const;
    int  l2_pages_now() const;
    int  l3_pages_now() const; // invariant: l1 + l2 + l3 == n_pages
    int  n_pages() const;
    int  page_tokens() const;
    int  ctx_tokens() const;
    int  l1_capacity() const;
    int  l2_capacity() const; // clamped to n_pages
    uint64_t kv_bytes_per_page() const;

    q4e_kv_paging_stats stats() const;
    void reset(); // all pages back to L3, stats cleared

private:
    void check_page(int page) const;
    void check_range(int lo, int hi) const;
    void make_room_l1();
    void move_page(int page, tier from, tier to); // backend first, then bookkeeping + stats

    int  ctx_ = 0, page_tok_ = 0, l1_cap_ = 0, l2_cap_ = 0;
    uint64_t kv_bytes_ = 0;
    std::vector<int> tier_; // per page: tier index
    std::list<int> l1_, l2_; // LRU order, front = LRU
    std::unordered_map<int, std::list<int>::iterator> l1_it_, l2_it_;
    std::function<bool(int, int, int)> backend_;
    q4e_kv_paging_stats stats_;
};
