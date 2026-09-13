// v6 VRAM gather (A-16.2b): CPU-side contract for the CUDA gather kernel.
// Kernel contract:
// - input : one published staging buffer, n chunks at chunk-aligned slots (offsets[])
// - output: dst holds n consecutive chunks, each chunk_bytes, in expert order
// - reads : stay inside [0, per_buf_bytes) of that one buffer, never cross buffers
// build/validate are pure C++ (no CUDA); the launch is implemented on the CUDA side.
#pragma once

#include "q4e-staging.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct q4e_gather_args_t {
    const void * staging_base = nullptr; // staging buffer base, set by caller (staging owns no memory)
    int buf = 0;                         // staging buffer id (0/1)
    uint32_t n_experts = 0;              // number of gathered expert chunks
    uint64_t chunk_bytes = 0;            // staging slot size (staging config chunk_bytes)
    uint64_t per_buf_bytes = 0;          // staging buffer capacity, validate bound
    std::vector<uint64_t> offsets;       // chunk slot offsets in staging, expert order
    std::vector<uint64_t> bytes;         // actual placed bytes per chunk (consume view)
    std::vector<int> expert_ids;         // router-selected expert ids, same order as offsets
};

// Fill args from the consume view of a published staging buffer.
// throws std::invalid_argument: bad buf id, n <= 0, null expert_ids, n > placed
// throws std::logic_error: buffer not published, chunk out of capacity
inline void q4e_gather_build_args(const q4e_staging & st, int buf, const int * expert_ids, int n, q4e_gather_args_t & out) {
    if (buf < 0 || buf > 1) {
        throw std::invalid_argument("q4e_gather: buf id must be 0 or 1");
    }
    if (n <= 0) {
        throw std::invalid_argument("q4e_gather: n must be > 0");
    }
    if (expert_ids == nullptr) {
        throw std::invalid_argument("q4e_gather: expert_ids must not be null");
    }
    if (st.state_of(buf) != q4e_staging::state::PUBLISHED) {
        throw std::logic_error("q4e_gather: buffer is not published");
    }
    const uint32_t placed = st.placed_of(buf);
    if ((uint32_t) n > placed) {
        throw std::invalid_argument("q4e_gather: n exceeds the number of placed chunks");
    }
    q4e_gather_args_t a;
    a.buf = buf;
    a.n_experts = (uint32_t) n;
    a.chunk_bytes = st.chunk_bytes();
    a.per_buf_bytes = st.per_buf_bytes();
    a.offsets.resize((std::size_t) n);
    a.bytes.resize((std::size_t) n);
    a.expert_ids.resize((std::size_t) n);
    for (int i = 0; i < n; ++i) {
        const q4e_staging_entry e = st.consume_at(buf, (uint32_t) i);
        if (e.bytes == 0 || e.bytes > a.chunk_bytes || e.offset + e.bytes > a.per_buf_bytes) {
            throw std::logic_error("q4e_gather: chunk is out of capacity");
        }
        a.offsets[(std::size_t) i] = e.offset;
        a.bytes[(std::size_t) i] = e.bytes;
        a.expert_ids[(std::size_t) i] = expert_ids[i];
    }
    out = a;
}

// throws std::invalid_argument: null base, bad buf id, n_experts == 0, size mismatch,
// offset out of [0, per_buf_bytes), offset not chunk aligned, bytes out of slot
inline void q4e_gather_validate(const q4e_gather_args_t & a) {
    if (a.staging_base == nullptr) {
        throw std::invalid_argument("q4e_gather: staging_base must not be null");
    }
    if (a.buf < 0 || a.buf > 1) {
        throw std::invalid_argument("q4e_gather: buf id must be 0 or 1");
    }
    if (a.n_experts == 0) {
        throw std::invalid_argument("q4e_gather: n_experts must be > 0");
    }
    if (a.chunk_bytes == 0 || a.per_buf_bytes == 0) {
        throw std::invalid_argument("q4e_gather: chunk_bytes and per_buf_bytes must be > 0");
    }
    if (a.offsets.size() != a.n_experts || a.bytes.size() != a.n_experts || a.expert_ids.size() != a.n_experts) {
        throw std::invalid_argument("q4e_gather: arg array size mismatch");
    }
    for (uint32_t i = 0; i < a.n_experts; ++i) {
        if (a.offsets[i] >= a.per_buf_bytes) {
            throw std::invalid_argument("q4e_gather: chunk offset out of capacity");
        }
        if (a.offsets[i] % a.chunk_bytes != 0) {
            throw std::invalid_argument("q4e_gather: chunk offset not aligned to chunk boundary");
        }
        if (a.bytes[i] == 0 || a.bytes[i] > a.chunk_bytes) {
            throw std::invalid_argument("q4e_gather: chunk bytes out of slot");
        }
        if (a.offsets[i] + a.bytes[i] > a.per_buf_bytes) {
            throw std::invalid_argument("q4e_gather: chunk exceeds capacity");
        }
    }
}

#ifdef GGML_CUDA
// CUDA side (B-16.x): copy the staged chunks into dst in expert order.
// Returns false when the launch is not possible (e.g. no CUDA context).
bool q4e_gather_launch_cuda(const q4e_gather_args_t & args, void * dst);
#else
inline bool q4e_gather_launch_stub(const q4e_gather_args_t & args, void * dst) {
    (void) args;
    (void) dst;
    return false; // CPU build: no CUDA kernel, stub for link
}
inline bool q4e_gather_launch_cuda(const q4e_gather_args_t & args, void * dst) {
    return q4e_gather_launch_stub(args, dst);
}
#endif
