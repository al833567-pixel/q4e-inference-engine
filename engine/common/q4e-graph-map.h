// v6 graph dynamic slot map (A-16.4a): split each token's router top_k experts into
// staged (already in the published staging view) and missing (needs H2D promotion).
// promotion_queue gives the deduplicated, first-seen order of the missing experts.
// Pure C++ bookkeeping, no ggml/CUDA dependency, owns no buffer memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct q4e_token_view {
    const int * staged  = nullptr; // pointers into internal storage; valid until the next set_* call
    const int * missing = nullptr;
    int n_staged  = 0;
    int n_missing = 0;
};

struct q4e_graph_map_agg {
    int n_pairs = 0;          // n_tokens * top_k
    int n_staged_pairs = 0;
    int n_missing_pairs = 0;
    int n_unique_staged = 0;  // distinct experts in the staged set
    int n_unique_missing = 0; // distinct experts that must be promoted
};

struct q4e_graph_map_stats {
    uint64_t maps = 0;    // set_router_choices calls
    uint64_t queries = 0; // token_view calls
};

class q4e_graph_map {
public:
    q4e_graph_map(int n_experts, int top_k, int n_tokens);
    // throws std::invalid_argument: n_experts <= 0, top_k <= 0, n_tokens <= 0

    void set_router_choices(const int * ids, int n); // flat n_tokens * top_k expert ids
    // throws std::invalid_argument: null ids, n != n_tokens * top_k, id outside [0, n_experts)

    void set_staged(const int * experts, int n); // dedup set of experts in the staging view
    // throws std::invalid_argument: n < 0, null experts with n > 0, id outside [0, n_experts)

    q4e_token_view token_view(int i); // staged/missing split for token i
    // throws std::out_of_range: i outside [0, n_tokens)
    // throws std::logic_error: router choices not set

    void promotion_queue(std::vector<int> & out) const; // dedup, first-seen order of missing
    // throws std::logic_error: router choices not set

    q4e_graph_map_agg aggregate() const;
    // throws std::logic_error: router choices not set

    q4e_graph_map_stats stats() const;
    void reset(); // clear choices, staged set, split lists, and stats

private:
    void recompute();

    int n_experts_ = 0;
    int top_k_ = 0;
    int n_tokens_ = 0;

    std::vector<int> choices_;
    bool have_choices_ = false;

    std::vector<char> staged_flags_;
    std::vector<int> staged_list_;

    std::vector<std::vector<int>> token_staged_;
    std::vector<std::vector<int>> token_missing_;
    std::vector<int> promo_;

    uint64_t maps_ = 0;
    uint64_t queries_ = 0;
};
