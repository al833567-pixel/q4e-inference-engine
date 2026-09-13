#include "q4e-graph-map.h"

q4e_graph_map::q4e_graph_map(int n_experts, int top_k, int n_tokens) :
    n_experts_(n_experts),
    top_k_(top_k),
    n_tokens_(n_tokens) {
    if (n_experts_ <= 0 || top_k_ <= 0 || n_tokens_ <= 0) {
        throw std::invalid_argument("q4e_graph_map: n_experts, top_k, n_tokens must be > 0");
    }
    staged_flags_.assign((std::size_t) n_experts_, 0);
}

void q4e_graph_map::set_router_choices(const int * ids, int n) {
    if (ids == nullptr) {
        throw std::invalid_argument("q4e_graph_map: ids must not be null");
    }
    if (n != n_tokens_ * top_k_) {
        throw std::invalid_argument("q4e_graph_map: n must be n_tokens * top_k");
    }
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= n_experts_) {
            throw std::invalid_argument("q4e_graph_map: expert id out of range");
        }
    }
    choices_.assign(ids, ids + n);
    have_choices_ = true;
    recompute();
    ++maps_;
}

void q4e_graph_map::set_staged(const int * experts, int n) {
    if (n < 0) {
        throw std::invalid_argument("q4e_graph_map: n must be >= 0");
    }
    if (n > 0 && experts == nullptr) {
        throw std::invalid_argument("q4e_graph_map: experts must not be null");
    }
    staged_flags_.assign((std::size_t) n_experts_, 0);
    staged_list_.clear();
    for (int i = 0; i < n; ++i) {
        const int e = experts[i];
        if (e < 0 || e >= n_experts_) {
            throw std::invalid_argument("q4e_graph_map: expert id out of range");
        }
        if (!staged_flags_[(std::size_t) e]) {
            staged_flags_[(std::size_t) e] = 1;
            staged_list_.push_back(e);
        }
    }
    if (have_choices_) {
        recompute();
    }
}

q4e_token_view q4e_graph_map::token_view(int i) {
    if (i < 0 || i >= n_tokens_) {
        throw std::out_of_range("q4e_graph_map: token index out of range");
    }
    if (!have_choices_) {
        throw std::logic_error("q4e_graph_map: router choices not set");
    }
    ++queries_;
    q4e_token_view v;
    v.staged  = token_staged_[(std::size_t) i].data();
    v.n_staged  = (int) token_staged_[(std::size_t) i].size();
    v.missing = token_missing_[(std::size_t) i].data();
    v.n_missing = (int) token_missing_[(std::size_t) i].size();
    return v;
}

void q4e_graph_map::promotion_queue(std::vector<int> & out) const {
    if (!have_choices_) {
        throw std::logic_error("q4e_graph_map: router choices not set");
    }
    out = promo_;
}

q4e_graph_map_agg q4e_graph_map::aggregate() const {
    if (!have_choices_) {
        throw std::logic_error("q4e_graph_map: router choices not set");
    }
    q4e_graph_map_agg a;
    a.n_pairs = n_tokens_ * top_k_;
    a.n_unique_staged  = (int) staged_list_.size();
    a.n_unique_missing = (int) promo_.size();
    for (int t = 0; t < n_tokens_; ++t) {
        a.n_staged_pairs  += (int) token_staged_[(std::size_t) t].size();
        a.n_missing_pairs += (int) token_missing_[(std::size_t) t].size();
    }
    return a;
}

q4e_graph_map_stats q4e_graph_map::stats() const {
    q4e_graph_map_stats s;
    s.maps = maps_;
    s.queries = queries_;
    return s;
}

void q4e_graph_map::reset() {
    choices_.clear();
    have_choices_ = false;
    staged_flags_.assign((std::size_t) n_experts_, 0);
    staged_list_.clear();
    token_staged_.clear();
    token_missing_.clear();
    promo_.clear();
    maps_ = 0;
    queries_ = 0;
}

void q4e_graph_map::recompute() {
    token_staged_.assign((std::size_t) n_tokens_, {});
    token_missing_.assign((std::size_t) n_tokens_, {});
    promo_.clear();
    std::vector<char> seen((std::size_t) n_experts_, 0);
    for (int t = 0; t < n_tokens_; ++t) {
        for (int k = 0; k < top_k_; ++k) {
            const int e = choices_[(std::size_t) t * top_k_ + (std::size_t) k];
            if (staged_flags_[(std::size_t) e]) {
                token_staged_[(std::size_t) t].push_back(e);
            } else {
                token_missing_[(std::size_t) t].push_back(e);
                if (!seen[(std::size_t) e]) {
                    seen[(std::size_t) e] = 1;
                    promo_.push_back(e);
                }
            }
        }
    }
}
