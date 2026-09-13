#include "q4e-kv-paging.h"

#include <algorithm>
#include <set>
#include <string>

q4e_kv_paging::q4e_kv_paging(int ctx_tokens, int page_tokens, int l1_pages, int l2_pages, uint64_t kv_bytes_per_page) {
    if (ctx_tokens <= 0) {
        throw std::invalid_argument("q4e_kv_paging: ctx_tokens must be > 0");
    }
    if (page_tokens <= 0) {
        throw std::invalid_argument("q4e_kv_paging: page_tokens must be > 0");
    }
    if (l1_pages <= 0) {
        throw std::invalid_argument("q4e_kv_paging: l1_pages must be > 0");
    }
    if (l2_pages < 0) {
        throw std::invalid_argument("q4e_kv_paging: l2_pages must be >= 0");
    }
    if (kv_bytes_per_page == 0) {
        throw std::invalid_argument("q4e_kv_paging: kv_bytes_per_page must be > 0");
    }
    const int n = (ctx_tokens + page_tokens - 1) / page_tokens;
    ctx_      = ctx_tokens;
    page_tok_ = page_tokens;
    l1_cap_   = std::min(l1_pages, n);
    l2_cap_   = std::min(l2_pages, n);
    kv_bytes_ = kv_bytes_per_page;
    tier_.assign((size_t) n, (int) tier::L3);
}

void q4e_kv_paging::set_tier_backend(std::function<bool(int, int, int)> backend) {
    backend_ = std::move(backend);
}

void q4e_kv_paging::check_page(int page) const {
    if (page < 0 || page >= (int) tier_.size()) {
        throw std::out_of_range("q4e_kv_paging: page out of range");
    }
}

void q4e_kv_paging::check_range(int lo, int hi) const {
    if (lo < 0 || hi <= lo || hi > ctx_) {
        throw std::invalid_argument("q4e_kv_paging: bad token range [lo, hi)");
    }
}

int q4e_kv_paging::page_of(int token) const {
    if (token < 0 || token >= ctx_) {
        throw std::out_of_range("q4e_kv_paging: token out of range");
    }
    return token / page_tok_;
}

q4e_kv_paging::token_range q4e_kv_paging::token_range_of(int page) const {
    check_page(page);
    const int lo = page * page_tok_;
    return { lo, std::min(lo + page_tok_, ctx_) };
}

q4e_kv_paging::tier q4e_kv_paging::tier_of(int page) const {
    check_page(page);
    return (tier) tier_[page];
}

int q4e_kv_paging::l1_pages_now() const  { return (int) l1_.size(); }
int q4e_kv_paging::l2_pages_now() const  { return (int) l2_.size(); }
int q4e_kv_paging::l3_pages_now() const  { return (int) tier_.size() - (int) l1_.size() - (int) l2_.size(); }
int q4e_kv_paging::n_pages() const       { return (int) tier_.size(); }
int q4e_kv_paging::page_tokens() const   { return page_tok_; }
int q4e_kv_paging::ctx_tokens() const    { return ctx_; }
int q4e_kv_paging::l1_capacity() const   { return l1_cap_; }
int q4e_kv_paging::l2_capacity() const   { return l2_cap_; }
uint64_t q4e_kv_paging::kv_bytes_per_page() const { return kv_bytes_; }

q4e_kv_paging_stats q4e_kv_paging::stats() const {
    return stats_;
}

void q4e_kv_paging::reset() {
    tier_.assign(tier_.size(), (int) tier::L3);
    l1_.clear();
    l2_.clear();
    l1_it_.clear();
    l2_it_.clear();
    stats_ = q4e_kv_paging_stats();
}

void q4e_kv_paging::move_page(int page, tier from, tier to) {
    if (from == to) {
        return;
    }
    if (backend_ && !backend_(page, (int) from, (int) to)) {
        throw std::runtime_error("q4e_kv_paging: tier backend failed to move page " + std::to_string(page));
    }
    if (from == tier::L1) {
        const auto it = l1_it_.find(page);
        l1_.erase(it->second);
        l1_it_.erase(it);
    } else if (from == tier::L2) {
        const auto it = l2_it_.find(page);
        l2_.erase(it->second);
        l2_it_.erase(it);
    }
    tier_[page] = (int) to;
    if (to == tier::L1) {
        l1_.push_back(page);
        l1_it_[page] = std::prev(l1_.end());
    } else if (to == tier::L2) {
        l2_.push_back(page);
        l2_it_[page] = std::prev(l2_.end());
    }
    if (to == tier::L1) {
        if (from == tier::L2) ++stats_.promote_l2_to_l1;
        else if (from == tier::L3) ++stats_.promote_l3_to_l1;
    } else if (from == tier::L1) {
        if (to == tier::L2) ++stats_.demote_l1_to_l2;
        else if (to == tier::L3) ++stats_.demote_l1_to_l3;
    } else if (from == tier::L2) {
        ++stats_.demote_l2_to_l3;
    }
}

void q4e_kv_paging::make_room_l1() {
    while ((int) l1_.size() >= l1_cap_) {
        const int victim = l1_.front();
        if (l2_cap_ == 0) {
            move_page(victim, tier::L1, tier::L3);
        } else if ((int) l2_.size() >= l2_cap_) {
            move_page(l2_.front(), tier::L2, tier::L3);
            move_page(victim, tier::L1, tier::L2);
        } else {
            move_page(victim, tier::L1, tier::L2);
        }
    }
}

void q4e_kv_paging::ensure_resident(int token_lo, int token_hi) {
    check_range(token_lo, token_hi);
    const int p_lo = page_of(token_lo);
    const int p_hi = page_of(token_hi - 1);
    if (p_hi - p_lo + 1 > l1_cap_) {
        throw std::invalid_argument("q4e_kv_paging: range does not fit the L1 capacity");
    }
    std::set<int> need_l2, need_l3;
    for (int p = p_lo; p <= p_hi; ++p) {
        const int t = tier_[p];
        if (t == (int) tier::L1) {
            l1_.splice(l1_.end(), l1_, l1_it_.at(p)); // touch: make most recent
        } else if (t == (int) tier::L2) {
            need_l2.insert(p);
        } else {
            need_l3.insert(p);
        }
    }
    std::vector<int> order;
    for (int p : l2_) {
        if (need_l2.count(p)) order.push_back(p);
    }
    for (int p : need_l3) {
        order.push_back(p); // std::set iterates in ascending page order
    }
    for (int p : order) {
        make_room_l1();
        move_page(p, (tier) tier_[p], tier::L1);
    }
}

void q4e_kv_paging::demote_range(int token_lo, int token_hi) {
    check_range(token_lo, token_hi);
    const int p_lo = page_of(token_lo);
    const int p_hi = page_of(token_hi - 1);
    std::vector<int> victims;
    for (int p : l1_) {
        if (p >= p_lo && p <= p_hi) victims.push_back(p); // L1 LRU order
    }
    for (int p : victims) {
        if (l2_cap_ == 0) {
            move_page(p, tier::L1, tier::L3);
        } else if ((int) l2_.size() >= l2_cap_) {
            move_page(l2_.front(), tier::L2, tier::L3);
            move_page(p, tier::L1, tier::L2);
        } else {
            move_page(p, tier::L1, tier::L2);
        }
    }
}
