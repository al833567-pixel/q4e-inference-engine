// v6 VRAM gather staging, A/B double buffer manager (A-16.2a)
// Two fixed-size buffers (0/1) alternate: one is filled with expert chunks while the
// other stays published and is consumed. publish makes a buffer consumable.
// A place into a full buffer is an overflow event: returns -1, caller must batch.
// Pure C++ bookkeeping, no ggml/CUDA dependency, owns no buffer memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct q4e_staging_config {
    uint32_t total_mb    = 128;     // A/B combined (--q4e-staging-mb); each buffer gets half
    uint64_t chunk_bytes = 3216896; // expert chunk slot size (~3.07 MB); each place must fit one slot
};

struct q4e_staging_entry {
    uint64_t offset = 0; // slot start within the buffer
    uint64_t bytes  = 0; // actual bytes of the placed chunk
};

struct q4e_staging_stats {
    uint64_t fills     = 0;
    uint64_t publishes = 0;
    uint64_t consumes  = 0;
    uint64_t overflows = 0; // places rejected on a full buffer
};

class q4e_staging {
public:
    enum class state { IDLE, FILLING, PUBLISHED };

    explicit q4e_staging(const q4e_staging_config & cfg = q4e_staging_config());
    // throws std::invalid_argument: bad buf id, bytes <= 0, bytes > chunk_bytes, total_mb < 4, chunk_bytes == 0
    // throws std::logic_error: protocol violation (operation called in the wrong buffer state)

    void begin_fill(int buf); // IDLE -> FILLING; throws if the buffer is busy or another buffer is filling
    int64_t place(int buf, uint64_t bytes); // FILLING only; slot offset, or -1 on a full buffer (overflow event)
    uint32_t publish(int buf); // FILLING -> PUBLISHED; returns placed count
    uint32_t consume_begin(int buf); // PUBLISHED only; returns placed count
    q4e_staging_entry consume_at(int buf, uint32_t i) const; // PUBLISHED only; i out of range throws
    void release(int buf); // PUBLISHED -> IDLE, drops the content

    int filling_buf() const; // -1 = none
    state state_of(int buf) const;
    uint32_t placed_of(int buf) const;

    q4e_staging_stats stats() const;
    uint64_t per_buf_bytes() const;
    uint64_t chunk_bytes() const;
    uint32_t max_chunks_per_buf() const; // per_buf_bytes / chunk_bytes
    void reset();

private:
    struct buffer {
        state st = state::IDLE;
        uint32_t placed = 0;
        std::vector<q4e_staging_entry> entries;
    };

    void check_buf(int buf) const;

    q4e_staging_config cfg_;
    std::vector<buffer> bufs_;
    uint64_t per_buf_bytes_ = 0;
    uint32_t max_chunks_ = 0;
    int filling_ = -1;
    q4e_staging_stats stats_;
};
