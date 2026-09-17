#include "q4e-pool-backend.h"

#include <cstring>

static std::size_t part_elems(const q4e_pool_backend_config & cfg) {
    return (std::size_t) cfg.n_ff * (std::size_t) cfg.n_embd;
}

q4e_pool_backend::q4e_pool_backend(const q4e_pool_backend_config & cfg) :
    cfg_(cfg),
    pool_(cfg.pool),
    up_(cfg.external_l1 ? 0 : (std::size_t) cfg.n_l1_slots * part_elems(cfg)),
    gate_(cfg.external_l1 ? 0 : (std::size_t) cfg.n_l1_slots * part_elems(cfg)),
    down_(cfg.external_l1 ? 0 : (std::size_t) cfg.n_l1_slots * part_elems(cfg)),
    l2_((std::size_t) cfg.n_l2_slots * 3 * part_elems(cfg)),
    sources_((std::size_t) cfg.pool.n_layers),
    tables_((std::size_t) cfg.pool.n_layers, std::vector<int32_t>((std::size_t) cfg.pool.n_experts_per_layer, 0)) {
    if (cfg_.n_ff <= 0) {
        throw std::invalid_argument("q4e_pool_backend: n_ff must be > 0");
    }
    if (cfg_.n_embd <= 0) {
        throw std::invalid_argument("q4e_pool_backend: n_embd must be > 0");
    }
    if (cfg_.n_l1_slots < 0) {
        throw std::invalid_argument("q4e_pool_backend: n_l1_slots must be >= 0");
    }
    if (cfg_.n_l2_slots < 0) {
        throw std::invalid_argument("q4e_pool_backend: n_l2_slots must be >= 0");
    }
    if (cfg_.pool.n_slots != cfg_.n_l1_slots) {
        throw std::invalid_argument("q4e_pool_backend: pool.n_slots must equal n_l1_slots");
    }
}

float * q4e_pool_backend::up_slot(int s) {
    if (s < 0 || s >= cfg_.n_l1_slots) {
        throw std::out_of_range("q4e_pool_backend: L1 slot out of range");
    }
    return l1_base(0) + (std::size_t) s * part_elems(cfg_);
}

float * q4e_pool_backend::gate_slot(int s) {
    if (s < 0 || s >= cfg_.n_l1_slots) {
        throw std::out_of_range("q4e_pool_backend: L1 slot out of range");
    }
    return l1_base(1) + (std::size_t) s * part_elems(cfg_);
}

float * q4e_pool_backend::down_slot(int s) {
    if (s < 0 || s >= cfg_.n_l1_slots) {
        throw std::out_of_range("q4e_pool_backend: L1 slot out of range");
    }
    return l1_base(2) + (std::size_t) s * part_elems(cfg_);
}

const float * q4e_pool_backend::up_slot(int s) const {
    if (s < 0 || s >= cfg_.n_l1_slots) {
        throw std::out_of_range("q4e_pool_backend: L1 slot out of range");
    }
    return l1_base(0) + (std::size_t) s * part_elems(cfg_);
}

const float * q4e_pool_backend::gate_slot(int s) const {
    if (s < 0 || s >= cfg_.n_l1_slots) {
        throw std::out_of_range("q4e_pool_backend: L1 slot out of range");
    }
    return l1_base(1) + (std::size_t) s * part_elems(cfg_);
}

const float * q4e_pool_backend::down_slot(int s) const {
    if (s < 0 || s >= cfg_.n_l1_slots) {
        throw std::out_of_range("q4e_pool_backend: L1 slot out of range");
    }
    return l1_base(2) + (std::size_t) s * part_elems(cfg_);
}

float * q4e_pool_backend::l1_base(int part) {
    if (cfg_.external_l1) {
        if (l1_ext_[part] == nullptr) {
            throw std::logic_error("q4e_pool_backend: external L1 not attached");
        }
        return l1_ext_[part];
    }
    return part == 0 ? up_.data() : (part == 1 ? gate_.data() : down_.data());
}

const float * q4e_pool_backend::l1_base(int part) const {
    if (cfg_.external_l1) {
        if (l1_ext_[part] == nullptr) {
            throw std::logic_error("q4e_pool_backend: external L1 not attached");
        }
        return l1_ext_[part];
    }
    return part == 0 ? up_.data() : (part == 1 ? gate_.data() : down_.data());
}

float * q4e_pool_backend::l2_slot(int s) {
    if (s < 0 || s >= cfg_.n_l2_slots) {
        throw std::out_of_range("q4e_pool_backend: L2 slot out of range");
    }
    return l2_.data() + (std::size_t) s * 3 * part_elems(cfg_);
}

const float * q4e_pool_backend::l2_slot(int s) const {
    if (s < 0 || s >= cfg_.n_l2_slots) {
        throw std::out_of_range("q4e_pool_backend: L2 slot out of range");
    }
    return l2_.data() + (std::size_t) s * 3 * part_elems(cfg_);
}

void q4e_pool_backend::set_layer_source(int layer, const float * up, const float * gate, const float * down) {
    if (layer < 0 || layer >= cfg_.pool.n_layers) {
        throw std::invalid_argument("q4e_pool_backend: layer out of range");
    }
    if (up == nullptr || gate == nullptr || down == nullptr) {
        throw std::invalid_argument("q4e_pool_backend: source pointers must not be null");
    }
    sources_[(std::size_t) layer] = source { up, gate, down, true };
}

int q4e_pool_backend::insert(int layer, int expert) {
    const int ev = pool_.insert(layer, expert);
    refresh_tables();
    return ev;
}

bool q4e_pool_backend::touch(int layer, int expert) {
    return pool_.touch(layer, expert);
}

int q4e_pool_backend::slot_of(int layer, int expert) const {
    return pool_.slot_of(layer, expert);
}

int q4e_pool_backend::occupancy_of(int layer) const {
    return pool_.occupancy_of(layer);
}

const char * q4e_pool_backend::state_of(int layer, int expert) const {
    return pool_.state_of(layer, expert);
}

void q4e_pool_backend::begin_round(int layer) {
    pool_.begin_round(layer);
}

void q4e_pool_backend::end_round(int layer) {
    pool_.end_round(layer);
}

int q4e_pool_backend::round_m_of(int layer) const {
    return pool_.round_m_of(layer);
}

q4e_global_pool_stats q4e_pool_backend::stats() const {
    return pool_.stats();
}

void q4e_pool_backend::reset() {
    pool_.reset();
    refresh_tables();
}

const int32_t * q4e_pool_backend::slot_table(int layer) const {
    if (layer < 0 || layer >= cfg_.pool.n_layers) {
        return nullptr;
    }
    return tables_[(std::size_t) layer].data();
}

void q4e_pool_backend::refresh_tables() {
    for (int l = 0; l < cfg_.pool.n_layers; ++l) {
        for (int e = 0; e < cfg_.pool.n_experts_per_layer; ++e) {
            const int s = pool_.slot_of(l, e);
            tables_[(std::size_t) l][(std::size_t) e] = (s >= 0) ? s : 0; // 0 = out-of-pool marker
        }
    }
}

void q4e_pool_backend::wire(q4e_h2d_pipeline & pipe) {
    pipe.set_fill_backend([this](int layer, int expert, void * dst, int64_t bytes) {
        return fill(layer, expert, dst, bytes);
    });
    pipe.set_h2d_backend([this](int64_t job_id, void * src, void * dst, int64_t bytes) {
        return h2d(job_id, src, dst, bytes);
    });
}

void q4e_pool_backend::attach_l1(float * up, float * gate, float * down) {
    if (!cfg_.external_l1) {
        throw std::logic_error("q4e_pool_backend: external_l1 not set in config");
    }
    if (up == nullptr || gate == nullptr || down == nullptr) {
        throw std::invalid_argument("q4e_pool_backend: L1 pointers must not be null");
    }
    if (l1_ext_[0] != nullptr) {
        throw std::logic_error("q4e_pool_backend: L1 already attached");
    }
    l1_ext_[0] = up;
    l1_ext_[1] = gate;
    l1_ext_[2] = down;
}

int q4e_pool_backend::graph_slot(int layer, int expert) const {
    const int s = pool_.slot_of(layer, expert); // throws on bad ids
    return s < 0 ? cfg_.n_l1_slots : s;
}

void q4e_pool_backend::sync_round(q4e_h2d_pipeline & pipe, const std::vector<std::vector<int>> & topk) {
    if (topk.size() > (std::size_t) cfg_.pool.n_layers) {
        throw std::invalid_argument("q4e_pool_backend: topk covers more layers than configured");
    }
    // missing experts need fill + h2d; resident ones already hold their L1 data
    std::vector<std::pair<int, int>> missing;
    for (std::size_t l = 0; l < topk.size(); ++l) {
        pool_.begin_round((int) l);
        for (int e : topk[l]) {
            if (pool_.slot_of((int) l, e) < 0) { // throws on bad id
                missing.push_back({ (int) l, e });
            }
        }
    }
    // reserve L1 for the whole round before any data moves: insert refreshes the
    // resident LRU and reserves the missing, so no later insert evicts a round member
    for (std::size_t l = 0; l < topk.size(); ++l) {
        for (int e : topk[l]) {
            pool_.insert((int) l, e);
        }
    }
    refresh_tables();
    // promote the missing; one poll drains every in-flight job
    const int64_t failed0 = pipe.stats().failed;
    for (const auto & m : missing) {
        while (pipe.submit(m.first, m.second) < 0) {
            pipe.poll();
        }
    }
    while (pipe.stats().in_flight > 0) {
        pipe.poll();
    }
    if (pipe.stats().failed > failed0) {
        throw std::runtime_error("q4e_pool_backend: sync_round h2d copy failed");
    }
    // every top-k expert must stay resident, else the pool path would zero-weight it
    for (std::size_t l = 0; l < topk.size(); ++l) {
        pool_.end_round((int) l);
        for (int e : topk[l]) {
            if (pool_.slot_of((int) l, e) < 0) {
                throw std::runtime_error("q4e_pool_backend: sync_round lost a top-k expert; pool too small for this round");
            }
        }
    }
    refresh_tables();
}

int64_t q4e_pool_backend::fill(int layer, int expert, void * dst, int64_t bytes) {
    if (bytes != expert_bytes()) {
        return -1;
    }
    if (layer < 0 || layer >= cfg_.pool.n_layers) {
        return -1;
    }
    if (expert < 0 || expert >= cfg_.pool.n_experts_per_layer) {
        return -1;
    }
    const int slot = (int) (uintptr_t) dst;
    if (slot < 0 || slot >= cfg_.n_l2_slots) {
        return -1;
    }
    const source & src = sources_[(std::size_t) layer];
    if (!src.set) {
        return -1;
    }
    const std::size_t elems = part_elems(cfg_);
    const std::size_t off = (std::size_t) expert * elems;
    float * out = l2_.data() + (std::size_t) slot * 3 * elems;
    std::memcpy(out, src.up + off, elems * sizeof(float));
    std::memcpy(out + elems, src.gate + off, elems * sizeof(float));
    std::memcpy(out + 2 * elems, src.down + off, elems * sizeof(float));
    l2_owner_[slot] = std::make_pair(layer, expert);
    return bytes;
}

bool q4e_pool_backend::h2d(int64_t job_id, void * src, void * dst, int64_t bytes) {
    (void) job_id;
    (void) dst; // the pipeline passes nullptr; the L1 target is resolved from the slot owner
    if (bytes != expert_bytes()) {
        return false;
    }
    const int slot = (int) (uintptr_t) src;
    if (slot < 0 || slot >= cfg_.n_l2_slots) {
        return false;
    }
    const auto it = l2_owner_.find(slot);
    if (it == l2_owner_.end()) {
        return false;
    }
    const int layer = it->second.first;
    const int expert = it->second.second;
    const int p = pool_.slot_of(layer, expert);
    if (p < 0) {
        return false; // not reserved in L1; the caller must run the pool insert first
    }
    const std::size_t elems = part_elems(cfg_);
    const std::size_t off = (std::size_t) p * elems;
    const float * in = l2_.data() + (std::size_t) slot * 3 * elems;
    std::memcpy(l1_base(0) + off, in, elems * sizeof(float));
    std::memcpy(l1_base(1) + off, in + elems, elems * sizeof(float));
    std::memcpy(l1_base(2) + off, in + 2 * elems, elems * sizeof(float));
    return true;
}
