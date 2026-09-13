#include "q4e-promotion.h"

#include <algorithm>

q4e_promotion_op::q4e_promotion_op(q4e_graph_map & map, q4e_h2d_pipeline & pipe, q4e_staging & st, int layer, int64_t expert_bytes) :
    map_(map), pipe_(pipe), st_(st), layer_(layer), expert_bytes_(expert_bytes) {
    if (layer_ < 0) {
        throw std::invalid_argument("q4e_promotion_op: layer must be >= 0");
    }
    if (expert_bytes_ <= 0) {
        throw std::invalid_argument("q4e_promotion_op: expert_bytes must be > 0");
    }
    if (expert_bytes_ > (int64_t) st_.chunk_bytes()) {
        throw std::invalid_argument("q4e_promotion_op: expert_bytes exceeds the staging chunk size");
    }
}

bool q4e_promotion_op::resident(int expert) const {
    const auto it = resident_.find(expert);
    if (it == resident_.end()) {
        return false;
    }
    const auto stt = st_.state_of(it->second.first);
    return stt == q4e_staging::state::FILLING || stt == q4e_staging::state::PUBLISHED;
}

int q4e_promotion_op::next_fill_buf() const {
    const int filling = st_.filling_buf();
    if (filling >= 0) {
        return filling;
    }
    if (st_.state_of(0) == q4e_staging::state::IDLE) {
        return 0;
    }
    if (st_.state_of(1) == q4e_staging::state::IDLE) {
        return 1;
    }
    return -1;
}

q4e_promotion_run q4e_promotion_op::run() {
    ++stats_.runs;
    deferred_.clear();
    q4e_promotion_run r;
    r.n_failed = (int) unreported_failed_;
    unreported_failed_ = 0;

    std::vector<int> q;
    map_.promotion_queue(q);
    for (const int e : q) {
        if (resident(e)) {
            ++stats_.skipped;
            continue;
        }
        const int64_t id = pipe_.submit(layer_, e);
        if (id < 0) {
            ++stats_.deferred;
            ++r.n_busy_deferred;
            deferred_.push_back(e);
            continue;
        }
        ++stats_.submitted;
        ++r.n_submitted;
        job_expert_[id] = e;
    }
    return r;
}

std::vector<int> q4e_promotion_op::reap() {
    ++stats_.reaps;
    std::vector<int> promoted;

    const uint64_t failed_before = pipe_.stats().failed;
    const std::vector<int64_t> done = pipe_.poll();
    const uint64_t failed_now = pipe_.stats().failed - failed_before;
    stats_.failed += failed_now;
    unreported_failed_ += failed_now;

    // pipeline job ids are reused LIFO, so poll() order is not submit order;
    // place in the map's first-seen order instead
    std::vector<int> q;
    map_.promotion_queue(q);
    std::unordered_map<int, int> pos;
    for (int i = 0; i < (int) q.size(); ++i) {
        pos[q[i]] = i;
    }
    std::vector<int> ready;
    for (const int64_t id : done) {
        const auto it = job_expert_.find(id);
        if (it == job_expert_.end()) {
            continue; // job not submitted by this op
        }
        const int e = it->second;
        job_expert_.erase(it);
        if (resident(e)) {
            ++stats_.skipped;
            continue;
        }
        ready.push_back(e);
    }
    const int far = (int) (1 << 30);
    std::sort(ready.begin(), ready.end(), [&](int a, int b) {
        const auto pa = pos.find(a);
        const auto pb = pos.find(b);
        const int ia = pa == pos.end() ? far : pa->second;
        const int ib = pb == pos.end() ? far : pb->second;
        if (ia != ib) {
            return ia < ib;
        }
        return a < b;
    });

    int buf = -1;
    for (const int e : ready) {
        if (buf < 0) {
            buf = next_fill_buf();
            if (buf >= 0 && st_.filling_buf() != buf) {
                st_.begin_fill(buf);
            }
        }
        if (buf < 0) {
            ++stats_.staging_full; // both buffers busy: expert stays in L2
            continue;
        }
        const uint32_t slot = st_.placed_of(buf);
        if (st_.place(buf, (uint64_t) expert_bytes_) < 0) {
            ++stats_.staging_full; // buffer full: expert stays in L2
            continue;
        }
        resident_[e] = { buf, slot };
        ++stats_.promoted;
        promoted.push_back(e);
    }
    return promoted;
}

void q4e_promotion_op::deferred_retry(std::vector<int> & out) const {
    out.clear();
    for (const int e : deferred_) {
        if (!resident(e)) {
            out.push_back(e);
        }
    }
}

q4e_promotion_stats q4e_promotion_op::stats() const {
    return stats_;
}

void q4e_promotion_op::reset() {
    resident_.clear();
    job_expert_.clear();
    deferred_.clear();
    unreported_failed_ = 0;
    stats_ = q4e_promotion_stats();
}
