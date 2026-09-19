#include "q4e-v6-runtime.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "ggml.h"
#include "ggml-backend.h"

q4e_v6_runtime::q4e_v6_runtime(const q4e_v6_runtime_cfg & cfg) :
        cfg_(cfg),
        pool_([this]() {
            q4e_global_pool_config pc;
            pc.n_experts_per_layer = cfg_.n_expert;
            pc.n_layers            = cfg_.n_layers;
            pc.n_slots             = cfg_.n_slots;
            pc.floor               = cfg_.floor;
            return pc;
        }()) {
    if (cfg_.n_layers <= 0 || cfg_.n_expert <= 0 || cfg_.n_used <= 0) {
        throw std::invalid_argument("q4e_v6_runtime: n_layers/n_expert/n_used must be positive");
    }
    if (cfg_.n_slots < 0 || cfg_.floor < 0) {
        throw std::invalid_argument("q4e_v6_runtime: n_slots/floor must be non-negative");
    }
    if (cfg_.n_slots != 0 && cfg_.n_slots < cfg_.n_layers * cfg_.floor) {
        throw std::invalid_argument("q4e_v6_runtime: n_slots below n_layers * floor");
    }
    layers_.resize((size_t) cfg_.n_layers);
}

void q4e_v6_runtime::bind_layer(int il, const q4e_v6_layer_io & io) {
    if (il < 0 || il >= cfg_.n_layers) {
        throw std::invalid_argument("q4e_v6_runtime::bind_layer: layer out of range");
    }
    layer_state & ls = layers_[(size_t) il];
    if (ls.bound) {
        throw std::invalid_argument("q4e_v6_runtime::bind_layer: layer already bound");
    }
    ls.io  = io;
    ls.bound = true;
}

void q4e_v6_runtime::observe(int il, const int32_t * sel, int64_t count) {
    if (il < 0 || il >= cfg_.n_layers || !layers_[(size_t) il].bound) {
        throw std::invalid_argument("q4e_v6_runtime::observe: unbound layer");
    }
    if (count < 0 || (count > 0 && sel == nullptr)) {
        throw std::invalid_argument("q4e_v6_runtime::observe: bad selections");
    }
    for (int64_t i = 0; i < count; ++i) {
        const int e = sel[i];
        if (e < 0 || e >= cfg_.n_expert) {
            continue; // defensive: the router never emits out-of-range ids
        }
        if (pool_.touch(il, e)) {
            continue; // resident (or staged earlier this round): LRU refreshed
        }
        try {
            pool_.insert(il, e);
        } catch (const std::runtime_error &) {
            ++insert_fails_; // pool full at floor: the expert stays a miss this round
            continue;
        }
        pending_.emplace_back(il, e);
    }
}

void q4e_v6_runtime::commit() {
    for (const auto & [il, e] : pending_) {
        const int slot = pool_.slot_of(il, e);
        if (slot < 0) {
            continue; // defensive: insert() guarantees a slot
        }
        fill_slot(slot, il, e);
    }
    pending_.clear();
    if (!enabled()) {
        return;
    }
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (layers_[(size_t) il].bound) {
            write_table(il);
        }
    }
}

void q4e_v6_runtime::fill_slot(int slot, int il, int expert) {
    const layer_state & ls = layers_[(size_t) il];
    copy_slice(ls.io.up_pool,
            (const uint8_t *) ls.io.up_src + (int64_t) expert * ls.io.up_stride,
            (int64_t) slot * ls.io.up_pool->nb[2], ls.io.up_pool->nb[2]);
    copy_slice(ls.io.gate_pool,
            (const uint8_t *) ls.io.gate_src + (int64_t) expert * ls.io.gate_stride,
            (int64_t) slot * ls.io.gate_pool->nb[2], ls.io.gate_pool->nb[2]);
    copy_slice(ls.io.down_pool,
            (const uint8_t *) ls.io.down_src + (int64_t) expert * ls.io.down_stride,
            (int64_t) slot * ls.io.down_pool->nb[2], ls.io.down_pool->nb[2]);
}

void q4e_v6_runtime::write_table(int il) {
    const layer_state & ls = layers_[(size_t) il];
    std::vector<float> tp((size_t) cfg_.n_expert);
    for (int e = 0; e < cfg_.n_expert; ++e) {
        const int slot = pool_.slot_of(il, e);
        tp[(size_t) e] = (slot >= 0) ? (float) slot : (float) cfg_.n_slots;
    }
    copy_slice(ls.io.table, tp.data(), 0, tp.size() * sizeof(float));
}

void q4e_v6_runtime::copy_slice(ggml_tensor * dst, const void * src, int64_t dst_offset, int64_t bytes) {
    if (dst->buffer == nullptr) {
        std::memcpy((uint8_t *) ggml_get_data(dst) + (size_t) dst_offset, src, (size_t) bytes);
        return;
    }
    ggml_backend_tensor_set(dst, src, (size_t) dst_offset, (size_t) bytes);
}

q4e_global_pool_stats q4e_v6_runtime::stats() const {
    return pool_.stats();
}
