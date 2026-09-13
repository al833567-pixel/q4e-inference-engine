#include "q4e-staging.h"

q4e_staging::q4e_staging(const q4e_staging_config & cfg) :
    cfg_(cfg),
    bufs_(2) {
    if (cfg_.total_mb < 4) {
        throw std::invalid_argument("q4e_staging: total_mb must be >= 4");
    }
    if (cfg_.chunk_bytes == 0) {
        throw std::invalid_argument("q4e_staging: chunk_bytes must be > 0");
    }
    per_buf_bytes_ = (uint64_t) cfg_.total_mb * 1024u * 1024u / 2u;
    max_chunks_ = (uint32_t) (per_buf_bytes_ / cfg_.chunk_bytes);
}

void q4e_staging::check_buf(int buf) const {
    if (buf < 0 || buf > 1) {
        throw std::invalid_argument("q4e_staging: buf id must be 0 or 1");
    }
}

void q4e_staging::begin_fill(int buf) {
    check_buf(buf);
    if (filling_ != -1) {
        throw std::logic_error("q4e_staging: another buffer is being filled");
    }
    buffer & b = bufs_[(std::size_t) buf];
    if (b.st != state::IDLE) {
        throw std::logic_error("q4e_staging: buffer is not idle (release it first)");
    }
    b.st = state::FILLING;
    b.placed = 0;
    b.entries.clear();
    filling_ = buf;
    ++stats_.fills;
}

int64_t q4e_staging::place(int buf, uint64_t bytes) {
    check_buf(buf);
    if (bytes == 0) {
        throw std::invalid_argument("q4e_staging: bytes must be > 0");
    }
    if (bytes > cfg_.chunk_bytes) {
        throw std::invalid_argument("q4e_staging: chunk does not fit one slot");
    }
    buffer & b = bufs_[(std::size_t) buf];
    if (b.st != state::FILLING) {
        throw std::logic_error("q4e_staging: buffer is not being filled");
    }
    if (b.placed >= max_chunks_) {
        ++stats_.overflows;
        return -1; // buffer full; caller must batch the gather
    }
    const uint32_t i = b.placed++;
    b.entries.push_back(q4e_staging_entry { (uint64_t) i * cfg_.chunk_bytes, bytes });
    return (int64_t) b.entries.back().offset;
}

uint32_t q4e_staging::publish(int buf) {
    check_buf(buf);
    buffer & b = bufs_[(std::size_t) buf];
    if (b.st != state::FILLING) {
        throw std::logic_error("q4e_staging: buffer is not being filled");
    }
    b.st = state::PUBLISHED;
    filling_ = -1;
    ++stats_.publishes;
    return b.placed;
}

uint32_t q4e_staging::consume_begin(int buf) {
    check_buf(buf);
    buffer & b = bufs_[(std::size_t) buf];
    if (b.st != state::PUBLISHED) {
        throw std::logic_error("q4e_staging: buffer is not published");
    }
    ++stats_.consumes;
    return b.placed;
}

q4e_staging_entry q4e_staging::consume_at(int buf, uint32_t i) const {
    check_buf(buf);
    const buffer & b = bufs_[(std::size_t) buf];
    if (b.st != state::PUBLISHED) {
        throw std::logic_error("q4e_staging: buffer is not published");
    }
    if (i >= b.placed) {
        throw std::out_of_range("q4e_staging: chunk index out of range");
    }
    return b.entries[i];
}

void q4e_staging::release(int buf) {
    check_buf(buf);
    buffer & b = bufs_[(std::size_t) buf];
    if (b.st != state::PUBLISHED) {
        throw std::logic_error("q4e_staging: buffer is not published");
    }
    b.st = state::IDLE;
    b.placed = 0;
    b.entries.clear();
}

int q4e_staging::filling_buf() const {
    return filling_;
}

q4e_staging::state q4e_staging::state_of(int buf) const {
    check_buf(buf);
    return bufs_[(std::size_t) buf].st;
}

uint32_t q4e_staging::placed_of(int buf) const {
    check_buf(buf);
    return bufs_[(std::size_t) buf].placed;
}

q4e_staging_stats q4e_staging::stats() const {
    return stats_;
}

uint64_t q4e_staging::per_buf_bytes() const {
    return per_buf_bytes_;
}

uint64_t q4e_staging::chunk_bytes() const {
    return cfg_.chunk_bytes;
}

uint32_t q4e_staging::max_chunks_per_buf() const {
    return max_chunks_;
}

void q4e_staging::reset() {
    for (buffer & b : bufs_) {
        b.st = state::IDLE;
        b.placed = 0;
        b.entries.clear();
    }
    filling_ = -1;
    stats_ = q4e_staging_stats();
}
