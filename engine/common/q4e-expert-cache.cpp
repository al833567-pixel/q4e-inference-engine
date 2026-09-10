#include "q4e-expert-cache.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

//
// deterministic tiered MoE expert cache; spec: v2 contract (v3 cache.py, removed v4); frozen in test-q4e-expert-cache.cpp
//

namespace {

void lru_move_to_end(std::list<int> & order, std::unordered_map<int, std::list<int>::iterator> & pos, int e) {
    order.erase(pos[e]);
    order.push_back(e);
    pos[e] = std::prev(order.end());
}

void lru_insert(std::list<int> & order, std::unordered_map<int, std::list<int>::iterator> & pos, int e) {
    order.push_back(e);
    pos[e] = std::prev(order.end());
}

int lru_pop_front(std::list<int> & order, std::unordered_map<int, std::list<int>::iterator> & pos) {
    const int e = order.front();
    order.pop_front();
    pos.erase(e);
    return e;
}

}

q4e_expert_cache::q4e_expert_cache(const q4e_expert_cache_config & cfg) :
    q4e_expert_cache(cfg, nullptr) {
}

q4e_expert_cache::q4e_expert_cache(const q4e_expert_cache_config & cfg, q4e_expert_cache_backend * backend) :
    cfg_(cfg),
    backend_(backend),
    layers_(cfg.n_layers),
    layer_touched_(cfg.n_layers, 0) {
}

void q4e_expert_cache::check_layer(int layer) const {
    if (layer < 0 || layer >= cfg_.n_layers) {
        throw std::invalid_argument("q4e_expert_cache: layer out of range");
    }
}

void q4e_expert_cache::ram_insert(int layer, int expert) {
    layer_state & ls = layers_[layer];
    if (ls.ram_pos.count(expert)) {
        lru_move_to_end(ls.ram_order, ls.ram_pos, expert);
        return;
    }
    if (ls.ram_order.size() >= (std::size_t) cfg_.ram_capacity_per_layer && !ls.ram_order.empty()) {
        const int old = lru_pop_front(ls.ram_order, ls.ram_pos);
        ls.prefetched.erase(old);
        ++pref_.evicted_from_ram;
        if (backend_) {
            backend_->evict(layer, old, "ssd");
        }
        placement_log_.push_back({layer, old, 2});
    }
    lru_insert(ls.ram_order, ls.ram_pos, expert);
}

void q4e_expert_cache::vram_insert(int layer, int expert) {
    layer_state & ls = layers_[layer];
    if (ls.vram_pos.count(expert)) {
        lru_move_to_end(ls.vram_order, ls.vram_pos, expert);
        return;
    }
    if (ls.vram_order.size() >= (std::size_t) cfg_.vram_capacity_per_layer && !ls.vram_order.empty()) {
        // vram -> ram only (one tier down, never straight to ssd)
        const int old = lru_pop_front(ls.vram_order, ls.vram_pos);
        ram_insert(layer, old);
        if (backend_) {
            backend_->evict(layer, old, "ram");
        }
        placement_log_.push_back({layer, old, 1});
    }
    lru_insert(ls.vram_order, ls.vram_pos, expert);
    if (backend_) {
        backend_->place(layer, expert, "vram");
    }
    placement_log_.push_back({layer, expert, 0});
}

q4e_expert_hits q4e_expert_cache::touch(int layer, const int * expert_ids, int64_t n_ids) {
    check_layer(layer);
    layer_state & ls = layers_[layer];
    if (!layer_touched_[layer]) {
        layer_touched_[layer] = 1;
    }
    q4e_expert_hits b;
    for (int64_t i = 0; i < n_ids; ++i) {
        const int e = expert_ids[i];
        ++touches_;
        if (e == Q4E_EXPERT_SHARED_ID) {
            if (cfg_.has_shared) {
                // shared expert is resident, does not count against capacity
                ++b.vram;
            } else {
                ++b.ssd;
                vram_insert(layer, e);
            }
            continue;
        }
        if (e < 0 || e >= cfg_.n_experts_per_layer) {
            throw std::invalid_argument("q4e_expert_cache: expert id out of range");
        }
        if (ls.vram_pos.count(e)) {
            ++b.vram;
            lru_move_to_end(ls.vram_order, ls.vram_pos, e);
            ls.prefetched.erase(e);
        } else if (ls.ram_pos.count(e)) {
            ++b.ram;
            if (ls.prefetched.erase(e) > 0) {
                ++pref_.hits;
            }
            ls.ram_order.erase(ls.ram_pos[e]);
            ls.ram_pos.erase(e);
            // promote ram -> vram (may evict the vram LRU entry to ram)
            vram_insert(layer, e);
        } else {
            ++b.ssd;
            vram_insert(layer, e);
        }
    }
    hits_.vram += b.vram;
    hits_.ram  += b.ram;
    hits_.ssd  += b.ssd;
    return b;
}

void q4e_expert_cache::prefetch(int layer, const int * hint_ids, int64_t n_hints) {
    check_layer(layer);
    layer_state & ls = layers_[layer];
    for (int64_t i = 0; i < n_hints; ++i) {
        const int e = hint_ids[i];
        if (e == Q4E_EXPERT_SHARED_ID || ls.vram_pos.count(e)) {
            ++pref_.skipped_in_vram;
            continue;
        }
        if (ls.ram_pos.count(e)) {
            ++pref_.already_staged;
            continue;
        }
        ram_insert(layer, e);
        ls.prefetched.insert(e);
        ++pref_.issued;
    }
}

q4e_expert_cache_stats q4e_expert_cache::stats() const {
    q4e_expert_cache_stats s;
    s.layers = cfg_.n_layers;
    s.touches = touches_;
    s.hits = hits_;
    const auto rate = [this](int64_t x) -> double {
        return touches_ ? std::round((double) x / touches_ * 1e4) / 1e4 : 0.0;
    };
    s.hit_rate_vram = rate(hits_.vram);
    s.hit_rate_ram  = rate(hits_.ram);
    s.hit_rate_ssd  = rate(hits_.ssd);
    s.prefetch = pref_;
    const auto occ = [this](bool vram) {
        q4e_expert_occupancy o;
        if (cfg_.n_layers == 0) {
            return o;
        }
        int mn = INT32_MAX;
        int mx = 0;
        int64_t sum = 0;
        for (const auto & ls : layers_) {
            const int len = (int) (vram ? ls.vram_order.size() : ls.ram_order.size());
            sum += len;
            mn = std::min(mn, len);
            mx = std::max(mx, len);
        }
        o.avg = std::round((double) sum / cfg_.n_layers * 10) / 10;
        o.min = mn;
        o.max = mx;
        return o;
    };
    s.occ_vram = occ(true);
    s.occ_ram  = occ(false);
    s.placement_events = (int64_t) placement_log_.size();
    for (const char v : layer_touched_) {
        s.layers_touched += (v != 0) ? 1 : 0;
    }
    s.hook_skips_device_ids = hook_skips_device_;
    return s;
}

const char * q4e_expert_cache::tier_of(int layer, int expert) const {
    if (expert == Q4E_EXPERT_SHARED_ID && cfg_.has_shared) {
        return "vram";
    }
    if (layer < 0 || layer >= cfg_.n_layers) {
        return "ssd";
    }
    const layer_state & ls = layers_[layer];
    if (ls.vram_pos.count(expert)) {
        return "vram";
    }
    if (ls.ram_pos.count(expert)) {
        return "ram";
    }
    return "ssd";
}

void q4e_expert_cache::reset() {
    for (auto & ls : layers_) {
        ls.vram_order.clear();
        ls.vram_pos.clear();
        ls.ram_order.clear();
        ls.ram_pos.clear();
        ls.prefetched.clear();
    }
    touches_ = 0;
    hits_ = q4e_expert_hits();
    pref_ = q4e_expert_prefetch_stats();
    placement_log_.clear();
    std::fill(layer_touched_.begin(), layer_touched_.end(), (char) 0);
    hook_skips_device_ = 0;
}

//
// v5 pool residency manager (A-15.1); pure CPU bookkeeping, H2D/SSD moves are A-15.3
//

q4e_expert_pool::q4e_expert_pool(const q4e_expert_pool_config & cfg) :
    cfg_(cfg),
    layers_((std::size_t) std::max(cfg.n_layers, 0)) {
    if (cfg_.l1_slots_per_layer > cfg_.n_experts_per_layer) {
        cfg_.l1_slots_per_layer = cfg_.n_experts_per_layer;
    }
    if (cfg_.l2_slots_per_layer > cfg_.n_experts_per_layer) {
        cfg_.l2_slots_per_layer = cfg_.n_experts_per_layer;
    }
}

void q4e_expert_pool::check_layer(int layer) const {
    if (layer < 0 || layer >= cfg_.n_layers) {
        throw std::invalid_argument("q4e_expert_pool: layer out of range");
    }
}

void q4e_expert_pool::check_expert(int expert) const {
    if (expert < 0 || expert >= cfg_.n_experts_per_layer) {
        throw std::invalid_argument("q4e_expert_pool: expert id out of range");
    }
}

void q4e_expert_pool::l1_insert(int layer, int expert) {
    if (cfg_.l1_slots_per_layer <= 0) {
        return;
    }
    layer_state & ls = layers_[(std::size_t) layer];
    if (ls.l1_pos.count(expert)) {
        lru_move_to_end(ls.l1_order, ls.l1_pos, expert);
        return;
    }
    if (ls.l1_order.size() >= (std::size_t) cfg_.l1_slots_per_layer && !ls.l1_order.empty()) {
        // evict L1 LRU; metadata no-op, L2 copy kept if present
        lru_pop_front(ls.l1_order, ls.l1_pos);
        ++l1_evictions_;
    }
    lru_insert(ls.l1_order, ls.l1_pos, expert);
}

void q4e_expert_pool::l2_insert(int layer, int expert) {
    if (cfg_.l2_slots_per_layer <= 0) {
        return;
    }
    layer_state & ls = layers_[(std::size_t) layer];
    if (ls.l2_pos.count(expert)) {
        lru_move_to_end(ls.l2_order, ls.l2_pos, expert);
        return;
    }
    if (ls.l2_order.size() >= (std::size_t) cfg_.l2_slots_per_layer && !ls.l2_order.empty()) {
        // free L2 LRU slot; expert may stay L1, else falls to L3
        lru_pop_front(ls.l2_order, ls.l2_pos);
        ++l2_evictions_;
    }
    lru_insert(ls.l2_order, ls.l2_pos, expert);
    ++l2_fills_;
}

int64_t q4e_expert_pool::access(int layer, int expert) {
    check_layer(layer);
    check_expert(expert);
    layer_state & ls = layers_[(std::size_t) layer];
    ++accesses_;
    if (ls.round_active) {
        ls.round_ids.insert(expert);
    }
    if (ls.l1_pos.count(expert)) {
        lru_move_to_end(ls.l1_order, ls.l1_pos, expert);
        ++hits_l1_;
        return 0;
    }
    if (ls.l2_pos.count(expert)) {
        ++hits_l2_;
        lru_move_to_end(ls.l2_order, ls.l2_pos, expert); // H2D read refreshes L2
        l1_insert(layer, expert);
        ++l2_to_l1_;
        return 1;
    }
    ++hits_l3_;
    ++l3_reads_;
    l2_insert(layer, expert); // fill L2 reserve; no-op when l2 slots = 0
    l1_insert(layer, expert);
    return 2;
}

void q4e_expert_pool::promote(int layer, int expert) {
    check_layer(layer);
    check_expert(expert);
    layer_state & ls = layers_[(std::size_t) layer];
    if (ls.l1_pos.count(expert)) {
        lru_move_to_end(ls.l1_order, ls.l1_pos, expert); // already resident
        return;
    }
    if (ls.l2_pos.count(expert)) {
        lru_move_to_end(ls.l2_order, ls.l2_pos, expert);
        l1_insert(layer, expert);
        ++l2_to_l1_; // H2D copy by A-15.3
        return;
    }
    l2_insert(layer, expert); // L3 -> L2 fill reserve; SSD read by A-15.3
}

void q4e_expert_pool::begin_round(int layer) {
    check_layer(layer);
    layer_state & ls = layers_[(std::size_t) layer];
    ls.round_ids.clear();
    ls.round_active = true;
}

void q4e_expert_pool::end_round(int layer) {
    check_layer(layer);
    layer_state & ls = layers_[(std::size_t) layer];
    if (!ls.round_active) {
        return;
    }
    ls.round_active = false;
    const int64_t m = (int64_t) ls.round_ids.size();
    ls.round_ids.clear();
    ++rounds_;
    m_sum_ += m;
    if (m > m_max_) {
        m_max_ = m;
    }
}

const char * q4e_expert_pool::state_of(int layer, int expert) const {
    if (layer < 0 || layer >= cfg_.n_layers) {
        return "l3";
    }
    const layer_state & ls = layers_[(std::size_t) layer];
    if (ls.l1_pos.count(expert)) {
        return "l1";
    }
    if (ls.l2_pos.count(expert)) {
        return "l2";
    }
    return "l3";
}

q4e_expert_pool_stats q4e_expert_pool::stats() const {
    q4e_expert_pool_stats s;
    s.accesses = accesses_;
    s.hits_l1 = hits_l1_;
    s.hits_l2 = hits_l2_;
    s.hits_l3 = hits_l3_;
    s.l1_evictions = l1_evictions_;
    s.l2_evictions = l2_evictions_;
    s.l3_reads = l3_reads_;
    s.l2_fills = l2_fills_;
    s.l2_to_l1 = l2_to_l1_;
    s.rounds = rounds_;
    s.m_sum = m_sum_;
    s.m_max = m_max_;
    return s;
}

void q4e_expert_pool::reset() {
    for (auto & ls : layers_) {
        ls.l1_order.clear();
        ls.l1_pos.clear();
        ls.l2_order.clear();
        ls.l2_pos.clear();
        ls.round_ids.clear();
        ls.round_active = false;
    }
    accesses_ = 0;
    hits_l1_ = 0;
    hits_l2_ = 0;
    hits_l3_ = 0;
    l1_evictions_ = 0;
    l2_evictions_ = 0;
    l3_reads_ = 0;
    l2_fills_ = 0;
    l2_to_l1_ = 0;
    rounds_ = 0;
    m_sum_ = 0;
    m_max_ = 0;
}

int q4e_expert_pool::per_layer_slots(int64_t n_experts, int n_layers, int n_experts_per_layer) {
    if (n_experts <= 0 || n_layers <= 0) {
        return 0;
    }
    const int64_t r = (n_experts + n_layers - 1) / n_layers;
    return (int) (r < (int64_t) n_experts_per_layer ? r : (int64_t) n_experts_per_layer);
}

//
// eval hook: touch() on the real router selection
//

int q4e_expert_layer_from_name(const char * name, int n_layers) {
    if (name == nullptr || std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    const char * p = name + 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int il = 0;
    while (*p >= '0' && *p <= '9') {
        il = il * 10 + (*p - '0');
        if (il >= 1000000) {
            return -1;
        }
        ++p;
    }
    if (*p != '.') {
        return -1;
    }
    if (il >= n_layers) {
        return -1;
    }
    return il;
}

bool q4e_expert_cache_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = static_cast<q4e_expert_cache *>(user_data);
    if (t == nullptr || self == nullptr) {
        return true;
    }

    // one match per MoE layer: the ffn_up_exps mul_mat_id (ids = t->src[2])
    const bool is_moe_up = t->op == GGML_OP_MUL_MAT_ID &&
                           t->src[0] != nullptr &&
                           std::strstr(t->src[0]->name, "up") != nullptr;
    if (ask) {
        return is_moe_up;
    }
    if (!is_moe_up) {
        return true;
    }

    const int il = q4e_expert_layer_from_name(t->src[0]->name, self->cfg().n_layers);
    if (il < 0) {
        return true;
    }

    const struct ggml_tensor * ids = t->src[2]; // [n_expert_used, n_tokens]
    if (ids == nullptr || ids->ne[0] <= 0 || ids->ne[1] <= 0) {
        return true;
    }
    // A-9: no sync D2H in the eval hot path; bookkeeping only for host-resident ids (CPU layers)
    // GPU layers keep all experts resident, so tiering does not apply to them
    if (ids->buffer == nullptr || !ggml_backend_buffer_is_host(ids->buffer)) {
        self->note_hook_skip();
        return true;
    }
    const int64_t n_ids = ids->ne[0];
    const int64_t n_tok = ids->ne[1];
    const int32_t * ids_data = (const int32_t *) ids->data;

    // ids is column-major: expert k of token t is ids_data[k * n_tok + t]
    std::vector<int> exps;
    exps.reserve((std::size_t) n_ids);
    for (int64_t t_i = 0; t_i < n_tok; ++t_i) {
        for (int64_t k = 0; k < n_ids; ++k) {
            exps[k] = (int) ids_data[k * n_tok + t_i];
        }
        try {
            self->touch(il, exps.data(), n_ids);
        } catch (...) {
            // A-8c: skip bad expert id; do not break the graph
            return true;
        }
    }
    return true;
}
