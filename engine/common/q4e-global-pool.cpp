#include "q4e-global-pool.h"

#include <algorithm>

q4e_global_pool::q4e_global_pool(const q4e_global_pool_config & cfg) :
    cfg_(cfg),
    layers_((std::size_t) std::max(cfg.n_layers, 0)),
    layer_count_((std::size_t) std::max(cfg.n_layers, 0), 0),
    slot_owner_((std::size_t) std::max(cfg.n_slots, 0), std::make_pair(-1, -1)),
    hits_layer_((std::size_t) std::max(cfg.n_layers, 0), 0),
    misses_layer_((std::size_t) std::max(cfg.n_layers, 0), 0),
    evictions_layer_((std::size_t) std::max(cfg.n_layers, 0), 0) {
    if (cfg_.n_layers <= 0) {
        throw std::invalid_argument("q4e_global_pool: n_layers must be > 0");
    }
    if (cfg_.n_experts_per_layer <= 0) {
        throw std::invalid_argument("q4e_global_pool: n_experts_per_layer must be > 0");
    }
    if (cfg_.n_slots < 0) {
        throw std::invalid_argument("q4e_global_pool: n_slots must be >= 0 (0 = disabled)");
    }
    if (cfg_.floor < 0) {
        throw std::invalid_argument("q4e_global_pool: floor must be >= 0");
    }
    if (cfg_.floor > cfg_.n_experts_per_layer) {
        cfg_.floor = cfg_.n_experts_per_layer; // load-time clamp (A-16.0 design)
    }
    if (cfg_.n_slots != 0 && cfg_.n_slots < cfg_.n_layers * cfg_.floor) {
        throw std::invalid_argument("q4e_global_pool: n_slots below n_layers * floor");
    }
}

void q4e_global_pool::check_layer(int layer) const {
    if (layer < 0 || layer >= cfg_.n_layers) {
        throw std::invalid_argument("q4e_global_pool: layer out of range");
    }
}

void q4e_global_pool::check_expert(int expert) const {
    if (expert < 0 || expert >= cfg_.n_experts_per_layer) {
        throw std::invalid_argument("q4e_global_pool: expert id out of range");
    }
}

void q4e_global_pool::note_round(int layer, int expert) {
    layer_state & ls = layers_[(std::size_t) layer];
    if (ls.round_active) {
        ls.round_ids.insert(expert);
    }
}

void q4e_global_pool::refresh_lru(uint64_t key) {
    lru_.splice(lru_.end(), lru_, lru_pos_[key]);
}

std::list<uint64_t>::iterator q4e_global_pool::find_victim() {
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        const int layer = (int) (*it >> 32);
        if (layer_count_[(std::size_t) layer] > cfg_.floor) {
            return it;
        }
        ++floor_hits_; // layer pinned at the floor, skip to next candidate
    }
    return lru_.end();
}

int q4e_global_pool::insert(int layer, int expert) {
    check_layer(layer);
    check_expert(expert);
    if (cfg_.n_slots <= 0) {
        return -1; // pool disabled
    }
    const uint64_t key = make_key(layer, expert);
    note_round(layer, expert);
    if (resident_.count(key)) {
        refresh_lru(key); // already resident; nothing to evict
        return -1;
    }
    int slot = -1;
    int evicted_expert = -1;
    if (lru_.size() < (std::size_t) cfg_.n_slots) {
        for (slot = 0; slot < cfg_.n_slots; ++slot) {
            if (slot_owner_[(std::size_t) slot].first == -1) {
                break;
            }
        }
    } else {
        auto victim = find_victim();
        if (victim == lru_.end()) {
            throw std::runtime_error("q4e_global_pool: pool full and all layers at floor; promotion path required");
        }
        const uint64_t vkey = *victim;
        slot = resident_[vkey];
        const int vlayer = (int) (vkey >> 32);
        evicted_expert = (int) (vkey & 0xffffffffull);
        slot_owner_[(std::size_t) slot] = std::make_pair(-1, -1);
        resident_.erase(vkey);
        lru_pos_.erase(vkey);
        lru_.erase(victim);
        --layer_count_[(std::size_t) vlayer];
        ++evictions_;
        ++evictions_layer_[(std::size_t) vlayer];
    }
    slot_owner_[(std::size_t) slot] = std::make_pair(layer, expert);
    resident_[key] = slot;
    ++layer_count_[(std::size_t) layer];
    lru_.push_back(key);
    lru_pos_[key] = std::prev(lru_.end());
    return evicted_expert;
}

bool q4e_global_pool::touch(int layer, int expert) {
    check_layer(layer);
    check_expert(expert);
    note_round(layer, expert);
    const uint64_t key = make_key(layer, expert);
    auto it = resident_.find(key);
    if (it != resident_.end()) {
        refresh_lru(key);
        ++hits_;
        ++hits_layer_[(std::size_t) layer];
        return true;
    }
    ++misses_;
    ++misses_layer_[(std::size_t) layer];
    return false;
}

const char * q4e_global_pool::state_of(int layer, int expert) const {
    check_layer(layer);
    check_expert(expert);
    return resident_.count(make_key(layer, expert)) ? "l1" : "l3";
}

int q4e_global_pool::occupancy_of(int layer) const {
    check_layer(layer);
    return layer_count_[(std::size_t) layer];
}

int q4e_global_pool::slot_of(int layer, int expert) const {
    check_layer(layer);
    check_expert(expert);
    const auto it = resident_.find(make_key(layer, expert));
    return it == resident_.end() ? -1 : it->second;
}

void q4e_global_pool::begin_round(int layer) {
    check_layer(layer);
    layer_state & ls = layers_[(std::size_t) layer];
    ls.round_ids.clear();
    ls.round_active = true;
}

void q4e_global_pool::end_round(int layer) {
    check_layer(layer);
    layer_state & ls = layers_[(std::size_t) layer];
    if (!ls.round_active) {
        return;
    }
    ls.round_active = false;
    const int m = (int) ls.round_ids.size();
    ls.round_ids.clear();
    ls.last_m = m;
    ++rounds_;
    m_sum_ += m;
    if (m > m_max_) {
        m_max_ = m;
    }
}

int q4e_global_pool::round_m_of(int layer) const {
    check_layer(layer);
    const layer_state & ls = layers_[(std::size_t) layer];
    return ls.round_active ? (int) ls.round_ids.size() : ls.last_m;
}

q4e_global_pool_stats q4e_global_pool::stats() const {
    q4e_global_pool_stats s;
    s.hits = hits_;
    s.misses = misses_;
    s.evictions = evictions_;
    s.floor_hits = floor_hits_;
    s.rounds = rounds_;
    s.m_sum = m_sum_;
    s.m_max = m_max_;
    s.hits_per_layer = hits_layer_;
    s.misses_per_layer = misses_layer_;
    s.evictions_per_layer = evictions_layer_;
    return s;
}

void q4e_global_pool::reset() {
    for (auto & ls : layers_) {
        ls.round_ids.clear();
        ls.round_active = false;
        ls.last_m = 0;
    }
    std::fill(layer_count_.begin(), layer_count_.end(), 0);
    std::fill(slot_owner_.begin(), slot_owner_.end(), std::make_pair(-1, -1));
    resident_.clear();
    lru_.clear();
    lru_pos_.clear();
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
    floor_hits_ = 0;
    rounds_ = 0;
    m_sum_ = 0;
    m_max_ = 0;
    std::fill(hits_layer_.begin(), hits_layer_.end(), 0);
    std::fill(misses_layer_.begin(), misses_layer_.end(), 0);
    std::fill(evictions_layer_.begin(), evictions_layer_.end(), 0);
}
