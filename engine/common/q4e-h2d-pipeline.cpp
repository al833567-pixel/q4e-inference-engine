#include "q4e-h2d-pipeline.h"

q4e_h2d_pipeline::q4e_h2d_pipeline(const q4e_h2d_pipeline_config & cfg) :
    cfg_(cfg) {
    if (cfg_.l2_ram_mb < 4) {
        throw std::invalid_argument("q4e_h2d_pipeline: l2_ram_mb must be >= 4");
    }
    if (cfg_.expert_bytes == 0) {
        throw std::invalid_argument("q4e_h2d_pipeline: expert_bytes must be > 0");
    }
    if (cfg_.max_inflight < 1) {
        throw std::invalid_argument("q4e_h2d_pipeline: max_inflight must be >= 1");
    }
    n_slots_ = (uint32_t) ((uint64_t) cfg_.l2_ram_mb * 1024u * 1024u / cfg_.expert_bytes);
    if (n_slots_ == 0) {
        throw std::invalid_argument("q4e_h2d_pipeline: L2 pool holds zero expert slots");
    }
    free_slots_.resize(n_slots_);
    for (uint32_t i = 0; i < n_slots_; ++i) {
        free_slots_[i] = i;
    }
}

void q4e_h2d_pipeline::set_fill_backend(q4e_h2d_fill_fn fn) {
    fill_ = std::move(fn);
}

void q4e_h2d_pipeline::set_h2d_backend(q4e_h2d_copy_fn fn) {
    h2d_ = std::move(fn);
}

int64_t q4e_h2d_pipeline::make_key(int layer, int expert) {
    return ((int64_t) layer << 32) | (uint32_t) expert;
}

void q4e_h2d_pipeline::check_id(int layer, int expert) const {
    if (layer < 0 || expert < 0) {
        throw std::invalid_argument("q4e_h2d_pipeline: layer/expert must be >= 0");
    }
}

bool q4e_h2d_pipeline::in_flight_of(int64_t key) const {
    for (const job & j : jobs_) {
        if (j.alive && make_key(j.layer, j.expert) == key) {
            return true;
        }
    }
    return false;
}

int64_t q4e_h2d_pipeline::submit(int layer, int expert) {
    check_id(layer, expert);
    if (stats_.in_flight >= cfg_.max_inflight) {
        ++stats_.busy;
        return -1;
    }
    const int64_t key = make_key(layer, expert);
    uint32_t slot;
    auto it = lru_pos_.find(key);
    if (it != lru_pos_.end()) {
        slot = it->second->slot;
        lru_.splice(lru_.begin(), lru_, it->second); // touch MRU
    } else {
        if (!fill_) {
            throw std::logic_error("q4e_h2d_pipeline: fill backend not set");
        }
        if (free_slots_.empty()) {
            // evict the LRU-most expert that is not in flight
            std::list<l2_entry>::iterator victim = lru_.end();
            for (auto it2 = lru_.begin(); it2 != lru_.end(); ++it2) {
                if (!in_flight_of(make_key(it2->layer, it2->expert))) {
                    victim = it2;
                }
            }
            if (victim == lru_.end()) {
                throw std::runtime_error("q4e_h2d_pipeline: L2 full and every resident expert is in flight");
            }
            lru_pos_.erase(make_key(victim->layer, victim->expert));
            free_slots_.push_back(victim->slot);
            lru_.erase(victim);
            ++stats_.evictions;
        }
        slot = free_slots_.back();
        free_slots_.pop_back();
        void * dst = reinterpret_cast<void *>(static_cast<uintptr_t>(slot));
        const int64_t n = fill_(layer, expert, dst, (int64_t) cfg_.expert_bytes);
        if (n <= 0) {
            free_slots_.push_back(slot);
            throw std::runtime_error("q4e_h2d_pipeline: fill backend failed");
        }
        lru_.push_front({ layer, expert, slot });
        lru_pos_[key] = lru_.begin();
        ++stats_.fills;
    }
    int64_t id;
    if (free_jobs_.empty()) {
        id = (int64_t) jobs_.size();
        jobs_.push_back({ layer, expert, slot, true });
    } else {
        id = (int64_t) free_jobs_.back();
        free_jobs_.pop_back();
        jobs_[id] = { layer, expert, slot, true };
    }
    ++stats_.submits;
    ++stats_.in_flight;
    return id;
}

std::vector<int64_t> q4e_h2d_pipeline::poll() {
    if (!h2d_) {
        throw std::logic_error("q4e_h2d_pipeline: h2d backend not set");
    }
    std::vector<int64_t> done;
    for (size_t i = 0; i < jobs_.size(); ++i) {
        job & j = jobs_[i];
        if (!j.alive) {
            continue;
        }
        void * src = reinterpret_cast<void *>(static_cast<uintptr_t>(j.slot));
        const bool ok = h2d_((int64_t) i, src, nullptr, (int64_t) cfg_.expert_bytes);
        j.alive = false;
        free_jobs_.push_back(i);
        --stats_.in_flight;
        if (ok) {
            ++stats_.completed;
            done.push_back((int64_t) i);
        } else {
            ++stats_.failed;
        }
    }
    return done;
}

const char * q4e_h2d_pipeline::l2_state(int layer, int expert) const {
    check_id(layer, expert);
    return lru_pos_.count(make_key(layer, expert)) ? "l2" : "l3";
}

uint32_t q4e_h2d_pipeline::l2_occupancy() const {
    return (uint32_t) lru_.size();
}

int64_t q4e_h2d_pipeline::l2_slot_of(int layer, int expert) const {
    check_id(layer, expert);
    auto it = lru_pos_.find(make_key(layer, expert));
    return it == lru_pos_.end() ? -1 : (int64_t) it->second->slot;
}

uint32_t q4e_h2d_pipeline::l2_slots() const {
    return n_slots_;
}

q4e_h2d_stats q4e_h2d_pipeline::stats() const {
    return stats_;
}

void q4e_h2d_pipeline::reset() {
    free_jobs_.clear();
    jobs_.clear();
    free_slots_.resize(n_slots_);
    for (uint32_t i = 0; i < n_slots_; ++i) {
        free_slots_[i] = i;
    }
    lru_.clear();
    lru_pos_.clear();
    stats_ = q4e_h2d_stats();
}
