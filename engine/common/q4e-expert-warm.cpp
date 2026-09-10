#ifndef NOMINMAX
#   define NOMINMAX
#endif

#include "q4e-expert-warm.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {

bool q4e_warm_parse_moe(const char * name, int * layer, int * which) {
    static const char * suf[4] = {
        ".ffn_gate_exps.weight",
        ".ffn_up_exps.weight",
        ".ffn_down_exps.weight",
        ".ffn_shexp.weight",
    };
    if (name == nullptr || strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    const char * p = name + 4;
    int L = 0;
    bool digit = false;
    while (*p >= '0' && *p <= '9') {
        L = L * 10 + (*p - '0');
        p++;
        digit = true;
    }
    if (!digit) {
        return false;
    }
    for (int w = 0; w < 4; w++) {
        if (strcmp(p, suf[w]) == 0) {
            *layer = L;
            *which = w;
            return true;
        }
    }
    return false;
}

// one long-lived stream per device; must outlive in-flight async copies
std::mutex g_warm_stream_mu;
std::vector<std::pair<ggml_backend_dev_t, ggml_backend_t>> g_warm_streams;

ggml_backend_t warm_stream_for(ggml_backend_dev_t dev) {
    std::lock_guard lk(g_warm_stream_mu);
    for (const auto & p : g_warm_streams) {
        if (p.first == dev) {
            return p.second;
        }
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be != nullptr) {
        g_warm_streams.push_back({dev, be});
    }
    return be;
}

}

q4e_expert_warm::q4e_expert_warm(const std::string & model_path) : model_path_(model_path) {
    slots_.assign(32, std::vector<int32_t>(4096));
    slot_busy_.assign(slots_.size(), 0);
}

q4e_expert_warm::~q4e_expert_warm() {
    stop();
}

void q4e_expert_warm::bind(q4e_expert_cache & cache) {
    cache_ = &cache;
    prev_.resize((size_t) cache.cfg().n_layers);
}

int64_t q4e_expert_warm::file_size(const std::string & path) const {
#ifdef _WIN32
    const std::wstring wpath = std::filesystem::path(path).wstring();
    HANDLE fh = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("open failed: " + path);
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(fh, &sz)) {
        CloseHandle(fh);
        throw std::runtime_error("GetFileSizeEx failed: " + path);
    }
    CloseHandle(fh);
    return (int64_t) sz.QuadPart;
#else
    (void) path;
    throw std::runtime_error("Windows-only (A-14.1)");
#endif
}

bool q4e_expert_warm::init() {
    if (!index_.empty() || !files_.empty()) {
        throw std::runtime_error("already initialized");
    }
#ifndef _WIN32
    throw std::runtime_error("Windows-only (A-14.1)");
#else
    // resolve shard files: the dir itself, or the parent dir of the first shard
    std::vector<std::string> files;
    std::error_code ec;
    if (std::filesystem::is_directory(model_path_, ec)) {
        for (const auto & entry : std::filesystem::directory_iterator(model_path_, ec)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
    } else {
        const std::filesystem::path root(model_path_);
        if (std::filesystem::is_regular_file(root, ec) && root.has_parent_path()) {
            for (const auto & entry : std::filesystem::directory_iterator(root.parent_path(), ec)) {
                if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                    files.push_back(entry.path().string());
                }
            }
            std::sort(files.begin(), files.end());
        }
    }
    if (files.empty()) {
        throw std::runtime_error("no gguf files found");
    }
    files_ = std::move(files);

    // read-only mapping per shard (mirror q4e-ngram)
    for (const auto & fn : files_) {
        const std::wstring wpath = std::filesystem::path(fn).wstring();
        HANDLE fh = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (fh == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("open failed: " + fn);
        }
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(fh, &sz)) {
            CloseHandle(fh);
            throw std::runtime_error("GetFileSizeEx failed: " + fn);
        }
        HANDLE mh = CreateFileMappingW(fh, nullptr, PAGE_READONLY,
            (DWORD) (sz.QuadPart >> 32), (DWORD) (sz.QuadPart & 0xFFFFFFFF), nullptr);
        if (mh == nullptr) {
            CloseHandle(fh);
            throw std::runtime_error("CreateFileMapping failed: " + fn);
        }
        const uint8_t * view = (const uint8_t *) MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
        if (view == nullptr) {
            CloseHandle(mh);
            CloseHandle(fh);
            throw std::runtime_error("MapViewOfFile failed: " + fn);
        }
        file_h_.push_back(fh);
        map_h_.push_back(mh);
        view_.push_back((HANDLE) view);
    }

    // GGUF scan: collect matched expert tensors, then fill index_
    struct mten {
        int     layer;
        int     which;
        int     file;
        int64_t offset; // absolute file offset of expert 0 (shared: tensor start)
        int64_t len;    // per-expert stride (shared: full tensor size)
        int64_t n_e;
    };
    std::vector<mten> m;
    for (size_t fi = 0; fi < files_.size(); fi++) {
        struct gguf_init_params params;
        params.no_alloc = true;
        params.ctx = nullptr;
        struct gguf_context * ctx = gguf_init_from_file(files_[fi].c_str(), params);
        if (ctx == nullptr) {
            throw std::runtime_error(files_[fi] + ": not a GGUF file (bad magic)");
        }
        const int64_t nt = gguf_get_n_tensors(ctx);
        const int64_t data_start = (int64_t) gguf_get_data_offset(ctx);
        for (int64_t i = 0; i < nt; i++) {
            const char * name = gguf_get_tensor_name(ctx, i);
            int L = 0;
            int which = 0;
            if (!q4e_warm_parse_moe(name, &L, &which)) {
                continue;
            }
            const int64_t * ne = gguf_get_tensor_ne(ctx, i);
            // expert axis is the slowest dim (llama.cpp convention: [.., .., n_experts])
            const int64_t n_e = ne[2];
            const int64_t tsize = (int64_t) gguf_get_tensor_size(ctx, i);
            const int64_t toff = data_start + (int64_t) gguf_get_tensor_offset(ctx, i);
            if (which < 3) {
                if (n_e <= 0 || tsize % n_e != 0) {
                    throw std::runtime_error("bad expert tensor dims: " + std::string(name));
                }
                m.push_back({L, which, (int) fi, toff, tsize / n_e, n_e});
            } else {
                m.push_back({L, 3, (int) fi, toff, tsize, 1});
            }
        }
        gguf_free(ctx);
    }

    if (m.empty()) {
        return true; // dense model: empty index, mappings stay open
    }
    for (size_t a = 0; a < m.size(); a++) {
        for (size_t b = a + 1; b < m.size(); b++) {
            if (m[a].layer == m[b].layer && m[a].which == m[b].which) {
                throw std::runtime_error("duplicate expert tensor for layer " + std::to_string(m[a].layer));
            }
        }
    }

    int max_l = 0;
    int max_e = 0;
    for (const auto & t : m) {
        max_l = std::max(max_l, t.layer);
        if (t.which < 3) {
            max_e = std::max(max_e, (int) t.n_e);
        }
    }
    index_.resize((size_t) max_l + 1);
    for (auto & L : index_) {
        L.resize((size_t) max_e + 1);
        L[0].resize(1); // shared expert (-1)
        for (size_t e = 1; e < L.size(); e++) {
            L[e].resize(3); // gate/up/down
        }
    }
    for (const auto & t : m) {
        const uint8_t * base = (const uint8_t *) view_[(size_t) t.file];
        if (t.which < 3) {
            for (int64_t e = 0; e < t.n_e; e++) {
                index_[t.layer][e + 1][t.which] = q4e_warm_slice{base, t.offset + e * t.len, t.len};
            }
        } else {
            index_[t.layer][0][0] = q4e_warm_slice{base, t.offset, t.len};
        }
    }

    n_layers_ = (int) index_.size();
    n_experts_ = (index_[0].size() > 0) ? (int) index_[0].size() - 1 : 0;
    for (int L = 0; L < n_layers_; L++) {
        for (int e = 0; e < n_experts_; e++) {
            if ((size_t) (e + 1) >= index_[L].size() || index_[L][e + 1].size() != 3) {
                throw std::runtime_error("expert index incomplete for layer " + std::to_string(L));
            }
            // UD models quantize each expert tensor differently, so gate/up/down len may differ
            if (!(index_[L][e + 1][0].len > 0 && index_[L][e + 1][1].len > 0 && index_[L][e + 1][2].len > 0)) {
                throw std::runtime_error("expert slice empty for layer " + std::to_string(L));
            }
        }
    }
    max_slice_len_ = 0;
    for (const auto & L : index_) {
        for (const auto & E : L) {
            for (const auto & s : E) {
                max_slice_len_ = std::max(max_slice_len_, s.len);
            }
        }
    }
    staging_[0].assign((size_t) max_slice_len_, 0);
    staging_[1].assign((size_t) max_slice_len_, 0);
    prev_.resize((size_t) n_layers_);
    return true;
#endif
}

void q4e_expert_warm::start() {
    if (th_.joinable()) {
        throw std::runtime_error("already started");
    }
    th_ = std::thread(&q4e_expert_warm::worker_loop, this);
}

void q4e_expert_warm::worker_loop() {
    for (;;) {
        std::unique_lock<std::mutex> lk(qmu_);
        qcv_.wait(lk, [this] {
            return stop_.load() || !ids_q_.empty() || !warm_q_.empty() || !ngram_q_.empty();
        });
        if (stop_.load() && ids_q_.empty() && warm_q_.empty() && ngram_q_.empty()) {
            break;
        }
        lk.unlock();
        drain_once();
    }
}

void q4e_expert_warm::drain_once() {
    ids_event e;
    bool have_ids = false;
    bool last_share = false;
    bool last_slot = false;
    {
        std::lock_guard lk(qmu_);
        if (!ids_q_.empty()) {
            e = ids_q_.front();
            ids_q_.pop_front();
            have_ids = true;
            // the slot is released once no queued event references it anymore
            if (ids_q_.empty() || ids_q_.front().slot != e.slot) {
                last_slot = true;
            }
            // ggml events are not refcounted: the last queued event sharing it frees it
            if (e.have_event && (ids_q_.empty() || ids_q_.front().event != e.event)) {
                last_share = true;
            }
        }
    }
    if (have_ids) {
        if (e.have_event) {
            ggml_backend_event_synchronize((ggml_backend_event_t) e.event);
            if (last_share) {
                ggml_backend_event_free((ggml_backend_event_t) e.event);
            }
        }
        const int32_t * ids = slots_[e.slot].data() + e.off;
        for (int64_t i = 0; i < e.n; i++) {
            const int32_t x = ids[i];
            if (x >= 0 && warm_done_.count(key(e.layer, x)) > 0) {
                overlap_hits_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (cache_ != nullptr && e.n > 0) {
            try {
                cache_->touch(e.layer, (const int *) ids, e.n);
            } catch (const std::exception &) {
            }
        }
        if (e.layer < (int) prev_.size() && !prev_[e.layer].empty()) {
            if (cache_ != nullptr) {
                try {
                    cache_->prefetch(e.layer, prev_[e.layer].data(), (int64_t) prev_[e.layer].size());
                } catch (const std::exception &) {
                }
            }
            for (int32_t x : prev_[e.layer]) {
                if (x >= 0) {
                    warm_hint(e.layer, x);
                } else if (x == Q4E_EXPERT_SHARED_ID && e.layer < (int) index_.size() &&
                           e.layer < (int) prev_.size() && index_[e.layer].size() > 0 &&
                           !index_[e.layer][0].empty()) {
                    warm_hint(e.layer, x);
                }
            }
        }
        if (e.layer < (int) prev_.size()) {
            prev_[e.layer].assign(ids, ids + e.n);
        }
        touches_seen_.fetch_add(e.n, std::memory_order_relaxed);
        if (last_slot) {
            std::lock_guard lk(qmu_);
            slot_busy_[(size_t) e.slot] = 0;
        }
    }
    for (int k = 0; k < 8; k++) {
        std::pair<int, int> p;
        {
            std::lock_guard lk(qmu_);
            if (warm_q_.empty()) {
                break;
            }
            p = warm_q_.front();
            warm_q_.pop_front();
        }
        do_warm_one(p.first, p.second);
    }
    for (int k = 0; k < 16; k++) {
        int64_t row = 0;
        {
            std::lock_guard lk(qmu_);
            if (ngram_q_.empty()) {
                break;
            }
            row = ngram_q_.front();
            ngram_q_.pop_front();
        }
        do_warm_row(row);
    }
}

void q4e_expert_warm::do_warm_one(int layer, int expert) {
    if (layer < 0 || layer >= (int) index_.size()) {
        return;
    }
    const size_t idxi = (size_t) (expert + 1);
    if (idxi >= index_[layer].size() || index_[layer][idxi].empty()) {
        return;
    }
    const uint64_t k = key(layer, expert);
    if (warm_done_.count(k) > 0) {
        inflight_.erase(k);
        return;
    }
    io_issued_.fetch_add(1, std::memory_order_relaxed);
    uint8_t * buf = staging_[staging_buf_].data();
    staging_buf_ ^= 1;
    for (const auto & s : index_[layer][idxi]) {
        memcpy(buf, s.base + s.offset, (size_t) s.len);
        io_bytes_.fetch_add(s.len, std::memory_order_relaxed);
    }
    warm_done_.insert(k);
    inflight_.erase(k);
    io_completed_.fetch_add(1, std::memory_order_relaxed);
}

void q4e_expert_warm::do_warm_row(int64_t row) {
    if (ngram_view_ == nullptr || row < 0 || row >= ngram_rows_) {
        return;
    }
    const uint64_t k = ngram_key(row);
    if (warm_done_.count(k) > 0) {
        inflight_.erase(k);
        return;
    }
    if ((int64_t) staging_[staging_buf_].size() < ngram_row_bytes_) {
        return;
    }
    // same page-warm technique as do_warm_one: copy the row into staging to pull its pages in
    uint8_t * buf = staging_[staging_buf_].data();
    staging_buf_ ^= 1;
    memcpy(buf, ngram_view_ + (ngram_row0_ + row * ngram_row_bytes_), (size_t) ngram_row_bytes_);
    warm_done_.insert(k);
    inflight_.erase(k);
    ngram_rows_warmed_.fetch_add(1, std::memory_order_relaxed);
    ngram_bytes_warmed_.fetch_add(ngram_row_bytes_, std::memory_order_relaxed);
}

bool q4e_expert_warm::enqueue_host_ids(int layer, const int32_t * ids, int64_t n_ids) {
    if (!ids || n_ids <= 0 || n_ids > 64 || layer < 0 || layer >= (int) index_.size()) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::lock_guard lk(qmu_);
    if ((int) ids_q_.size() >= ids_cap_) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // acquire a free slot; the worker releases it once its queued event is drained
    int slot = -1;
    for (int i = 0; i < (int) slots_.size(); i++) {
        const int s = (slot_next_ + i) % (int) slots_.size();
        if (!slot_busy_[(size_t) s]) {
            slot = s;
            break;
        }
    }
    if (slot < 0) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    slot_next_ = (slot + 1) % (int) slots_.size();
    slot_busy_[(size_t) slot] = 1;
    memcpy(slots_[slot].data(), ids, (size_t) n_ids * sizeof(int32_t));
    ids_q_.push_back(ids_event{layer, n_ids, slot, false, nullptr, 0});
    qcv_.notify_one();
    return true;
}

bool q4e_expert_warm::enqueue_device_ids(int layer, const ggml_tensor * ids) {
    if (!ids || layer < 0) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const int64_t n_ids = ids->ne[0];
    const int64_t n_tok = (ggml_n_dims(ids) >= 2) ? ids->ne[1] : 1;
    if (n_ids <= 0 || n_ids > 64 || n_tok <= 0 || n_tok > 32) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const ggml_backend_buffer_t buf = (ids->view_src != nullptr) ? ids->view_src->buffer : ids->buffer;
    const ggml_backend_buffer_type_t buft = (buf != nullptr) ? ggml_backend_buffer_get_type(buf) : nullptr;
    const ggml_backend_dev_t dev = (buft != nullptr) ? ggml_backend_buft_get_device(buft) : nullptr;
    if (dev == nullptr) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::lock_guard lk(qmu_);
    if ((int) ids_q_.size() >= ids_cap_) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // acquire a free slot; the worker releases it once its queued event is drained
    int slot = -1;
    for (int i = 0; i < (int) slots_.size(); i++) {
        const int s = (slot_next_ + i) % (int) slots_.size();
        if (!slot_busy_[(size_t) s]) {
            slot = s;
            break;
        }
    }
    if (slot < 0) {
        events_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    slot_next_ = (slot + 1) % (int) slots_.size();
    slot_busy_[(size_t) slot] = 1;
    const size_t nbytes = (size_t) n_ids * n_tok * sizeof(int32_t);

    bool have_event = false;
    ggml_backend_event_t ev = nullptr;
    struct ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    if (props.caps.events) {
        ggml_backend_t backend = warm_stream_for(dev);
        if (backend != nullptr) {
            ev = ggml_backend_event_new(dev);
            if (ev != nullptr) {
                ggml_backend_tensor_get_async(backend, ids, slots_[slot].data(), 0, nbytes);
                ggml_backend_event_record(ev, backend);
                have_event = true;
            }
        }
    }
    if (!have_event) {
        // no async/event support on this backend: blocking copy, eventless queue entry
        ggml_backend_tensor_get(ids, slots_[slot].data(), 0, nbytes);
    }
    for (int64_t t = 0; t < n_tok; t++) {
        ids_q_.push_back(ids_event{layer, n_ids, slot, have_event, (void *) ev, t * n_ids});
    }
    qcv_.notify_one();
    return true;
}

void q4e_expert_warm::warm_hint(int layer, int expert) {
    if (layer < 0 || layer >= (int) index_.size()) {
        return;
    }
    const size_t idxi = (size_t) (expert + 1);
    if (idxi >= index_[layer].size() || index_[layer][idxi].empty()) {
        return;
    }
    const uint64_t k = key(layer, expert);
    std::lock_guard lk(qmu_);
    if (inflight_.count(k) > 0) {
        return;
    }
    if ((int) warm_q_.size() >= warm_cap_) {
        io_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    inflight_.insert(k);
    warm_q_.push_back({layer, expert});
    qcv_.notify_one();
}

bool q4e_expert_warm::init_ngram(const std::string & model_path) {
#ifndef _WIN32
    (void) model_path;
    throw std::runtime_error("Windows-only (A-14.4)");
#else
    if (files_.empty()) {
        throw std::runtime_error("call init() first");
    }
    // parse over the resolved shard dir, so the tensor is found regardless of which shard holds it
    const std::string dir = std::filesystem::path(files_.front()).parent_path().string();
    try {
        ngram_tbl_.parse(dir);
    } catch (const std::exception & e) {
        LOG_WRN("q4e: ngram tensor absent, ngram prefetch off (%s)\n", e.what());
        return false;
    }
    int fi = -1;
    for (size_t i = 0; i < files_.size(); i++) {
        if (files_[i] == ngram_tbl_.path()) {
            fi = (int) i;
            break;
        }
    }
    if (fi < 0) {
        LOG_WRN("q4e: ngram tensor file %s not among mapped shards, ngram prefetch off\n", ngram_tbl_.path().c_str());
        return false;
    }
    ngram_row0_ = ngram_tbl_.data_offset(); // absolute in-file offset of the tensor data
    ngram_rows_ = ngram_tbl_.n_rows();
    ngram_row_bytes_ = ngram_tbl_.row_bytes();
    const int64_t fsz = file_size(files_[fi]);
    if (ngram_row0_ <= 0 || ngram_rows_ <= 0 || ngram_row_bytes_ <= 0 ||
        ngram_row0_ + ngram_rows_ * ngram_row_bytes_ > fsz) {
        LOG_WRN("q4e: ngram tensor range out of file, ngram prefetch off\n");
        return false;
    }
    ngram_view_ = (const uint8_t *) view_[(size_t) fi];
    if ((int64_t) staging_[0].size() < ngram_row_bytes_) {
        staging_[0].resize((size_t) ngram_row_bytes_);
        staging_[1].resize((size_t) ngram_row_bytes_);
    }
    ngram_ready_ = true;
    ngram_mode_ = 2; // full = default on, like q4e_prefetch
    LOG_INF("q4e: ngram prefetch ready (file=%s, rows=%lld, row=%lld B)\n",
            files_[fi].c_str(), (long long) ngram_rows_, (long long) ngram_row_bytes_);
    return true;
#endif
}

void q4e_expert_warm::ngram_hint(int64_t t, int64_t t_prev1, int64_t t_prev2) {
    if (!ngram_ready_) {
        return;
    }
    const int m = ngram_mode_.load(std::memory_order_relaxed);
    if (m == 0) {
        return;
    }
    ngram_hints_.fetch_add(1, std::memory_order_relaxed);
    if (m == 1) {
        return; // shim: enqueue-cost accounting only, no I/O
    }
    bool pushed = false;
    for (int64_t row : ngram_tbl_.row_ids_for_token(t, t_prev1, t_prev2)) {
        if (row < 0 || row >= ngram_rows_) {
            continue;
        }
        const uint64_t k = ngram_key(row);
        std::lock_guard lk(qmu_);
        if (inflight_.count(k) > 0) {
            continue;
        }
        if ((int) ngram_q_.size() >= ngram_cap_) {
            ngram_rows_dropped_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        inflight_.insert(k);
        ngram_q_.push_back(row);
        pushed = true;
    }
    if (pushed) {
        qcv_.notify_one();
    }
}

bool q4e_expert_warm::ngram_ready() const {
    return ngram_ready_;
}

int q4e_expert_warm::ngram_mode() const {
    return ngram_mode_.load(std::memory_order_relaxed);
}

void q4e_expert_warm::set_ngram_mode(int m) {
    ngram_mode_ = (m == 1 || m == 2) ? m : 0;
}

int64_t q4e_expert_warm::ngram_hints() const {
    return ngram_hints_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::ngram_rows_warmed() const {
    return ngram_rows_warmed_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::ngram_rows_dropped() const {
    return ngram_rows_dropped_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::ngram_bytes_warmed() const {
    return ngram_bytes_warmed_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::io_issued() const {
    return io_issued_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::io_bytes() const {
    return io_bytes_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::io_completed() const {
    return io_completed_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::io_dropped() const {
    return io_dropped_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::overlap_hits() const {
    return overlap_hits_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::events_dropped() const {
    return events_dropped_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::touches_seen() const {
    return touches_seen_.load(std::memory_order_relaxed);
}

int64_t q4e_expert_warm::hook_skips_weights_device() const {
    return hook_skips_weights_device_.load(std::memory_order_relaxed);
}

void q4e_expert_warm::set_index_for_test(std::vector<std::vector<std::vector<q4e_warm_slice>>> idx) {
    index_ = std::move(idx);
    n_layers_ = (int) index_.size();
    n_experts_ = n_layers_ > 0 ? (int) index_[0].size() - 1 : 0;
    max_slice_len_ = 0;
    for (const auto & L : index_) {
        for (const auto & E : L) {
            for (const auto & s : E) {
                max_slice_len_ = std::max(max_slice_len_, s.len);
            }
        }
    }
    staging_[0].assign((size_t) max_slice_len_, 0);
    staging_[1].assign((size_t) max_slice_len_, 0);
    prev_.resize((size_t) n_layers_);
}

void q4e_expert_warm::set_queue_caps_for_test(int ids_cap, int warm_cap) {
    std::lock_guard lk(qmu_);
    ids_cap_ = ids_cap;
    warm_cap_ = warm_cap;
}

void q4e_expert_warm::set_ngram_for_test(const uint8_t * base, int64_t row0, int64_t rows, int64_t row_bytes, bool ready) {
    ngram_view_ = base;
    ngram_row0_ = row0;
    ngram_rows_ = rows;
    ngram_row_bytes_ = row_bytes;
    ngram_ready_ = ready;
    if (rows >= 16) {
        ngram_tbl_.set_entries_for_test(rows / 16);
    }
    if ((int64_t) staging_[0].size() < row_bytes) {
        staging_[0].resize((size_t) row_bytes);
        staging_[1].resize((size_t) row_bytes);
    }
}

bool q4e_expert_warm::has_pending() const {
    std::lock_guard lk(const_cast<std::mutex &>(qmu_)); // mutex is non-mutable in the header
    return !ids_q_.empty() || !warm_q_.empty() || !ngram_q_.empty();
}

void q4e_expert_warm::mark_all_slots_busy_for_test() {
    std::lock_guard lk(qmu_);
    std::fill(slot_busy_.begin(), slot_busy_.end(), 1);
}

int q4e_expert_warm::free_slots_for_test() const {
    std::lock_guard lk(const_cast<std::mutex &>(qmu_)); // mutex is non-mutable in the header
    int n = 0;
    for (char b : slot_busy_) {
        if (b == 0) {
            n++;
        }
    }
    return n;
}

uint64_t q4e_expert_warm::key(int layer, int expert) {
    return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) (expert + 1);
}

// bit 62 namespaces ngram row ids so they never collide with (layer, expert) keys
uint64_t q4e_expert_warm::ngram_key(int64_t row) {
    return (1ULL << 62) | (uint64_t) (row & ((1LL << 62) - 1));
}

void q4e_expert_warm::stop() {
    stop_ = true;
    qcv_.notify_all();
    if (th_.joinable()) {
        th_.join();
    }
#ifdef _WIN32
    for (HANDLE v : view_) {
        if (v != nullptr) {
            UnmapViewOfFile(v);
        }
    }
    for (HANDLE h : map_h_) {
        if (h != nullptr) {
            CloseHandle(h);
        }
    }
    for (HANDLE h : file_h_) {
        if (h != nullptr) {
            CloseHandle(h);
        }
    }
    view_.clear();
    map_h_.clear();
    file_h_.clear();
#endif
}

bool q4e_expert_warm_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = static_cast<q4e_expert_warm *>(user_data);
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

    const int il = q4e_expert_layer_from_name(t->src[0]->name, self->n_layers());
    if (il < 0) {
        return true;
    }

    // only prefetch host-resident expert weights; GPU layers stay resident (zero I/O)
    if (t->src[0]->buffer == nullptr || !ggml_backend_buffer_is_host(t->src[0]->buffer)) {
        self->note_hook_skip_weights_device();
        return true;
    }

    const struct ggml_tensor * ids = t->src[2];
    if (ids == nullptr || ids->ne[0] <= 0 || ids->ne[1] <= 0) {
        return true;
    }
    const int64_t n_ids = ids->ne[0];
    const int64_t n_tok = ids->ne[1];

    if (ids->buffer != nullptr && ggml_backend_buffer_is_host(ids->buffer)) {
        // ids is column-major: expert k of token t is ids_data[k * n_tok + t]
        std::vector<int32_t> token_ids((size_t) n_ids);
        const int32_t * ids_data = (const int32_t *) ids->data;
        for (int64_t t_i = 0; t_i < n_tok; t_i++) {
            for (int64_t k = 0; k < n_ids; k++) {
                token_ids[(size_t) k] = ids_data[k * n_tok + t_i];
            }
            self->enqueue_host_ids(il, token_ids.data(), n_ids); // > 64 ids: drops + counts
        }
    } else {
        self->enqueue_device_ids(il, ids);
    }
    return true;
}

//
// v5 H2D promotion pipeline (A-15.3); deterministic tick clock, pure CPU
//

q4e_expert_h2d::q4e_expert_h2d(q4e_expert_h2d_backend * backend, const q4e_h2d_config & cfg) :
    backend_(backend), cfg_(cfg) {
}

void q4e_expert_h2d::tick() {
    ++tick_;
}

uint64_t q4e_expert_h2d::key(int layer, int expert) {
    return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) (expert + 1);
}

bool q4e_expert_h2d::in_range(int layer, int expert) const {
    const int nl = pool_ != nullptr ? pool_->cfg().n_layers : cfg_.n_layers;
    const int ne = pool_ != nullptr ? pool_->cfg().n_experts_per_layer : cfg_.n_experts_per_layer;
    return layer >= 0 && layer < nl && expert >= 0 && expert < ne;
}

// L1 = resident: pool state when bound, else the done set (standalone mode)
bool q4e_expert_h2d::resident(int layer, int expert) const {
    if (pool_ != nullptr) {
        return std::strcmp(pool_->state_of(layer, expert), "l1") == 0;
    }
    return done_.count(key(layer, expert)) > 0;
}

bool q4e_expert_h2d::issue(int layer, int expert, int64_t bytes) {
    if ((int) queued_.size() >= cfg_.queue_cap) {
        ++h2d_dropped_;
        return false;
    }
    job j;
    j.layer      = layer;
    j.expert     = expert;
    j.bytes      = bytes;
    j.issue_tick = tick_;
    queued_.push_back(j);
    queued_keys_.insert(key(layer, expert));
    return true;
}

bool q4e_expert_h2d::hint(int layer, int expert, int64_t bytes) {
    if (!in_range(layer, expert)) {
        return false;
    }
    const uint64_t k = key(layer, expert);
    if (resident(layer, expert)) {
        ++dedup_skips_;
        return false;
    }
    if (queued_keys_.count(k) > 0 || inflight_keys_.count(k) > 0) {
        ++dedup_skips_;
        return false;
    }
    return issue(layer, expert, bytes >= 0 ? bytes : cfg_.expert_bytes);
}

void q4e_expert_h2d::access(int layer, int expert) {
    if (!in_range(layer, expert)) {
        return;
    }
    ++accesses_;
    const uint64_t k = key(layer, expert);
    if (resident(layer, expert)) {
        return;
    }
    if (inflight_keys_.count(k) > 0) {
        ++overlap_hits_; // this access overlapped the in-flight H2D copy
        return;
    }
    if (queued_keys_.count(k) > 0) {
        return; // copy not started yet: no overlap
    }
    issue(layer, expert, cfg_.expert_bytes); // L1 miss: lazy promotion
}

void q4e_expert_h2d::drain_once() {
    // issue queued jobs in FIFO order; the copy starts now
    while (!queued_.empty()) {
        job j = queued_.front();
        queued_.pop_front();
        queued_keys_.erase(key(j.layer, j.expert));
        if (backend_ != nullptr) {
            backend_->copy_expert(j.layer, j.expert, j.bytes);
        }
        j.ready_tick = tick_ + cfg_.copy_latency;
        inflight_.push_back(j);
        inflight_keys_.insert(key(j.layer, j.expert));
        ++h2d_issued_;
        h2d_bytes_ += j.bytes;
    }
    // complete ready inflight jobs (FIFO); move the pool state (L2 -> L1, or L3 -> L2 reserve)
    for (auto it = inflight_.begin(); it != inflight_.end();) {
        if (tick_ >= it->second.ready_tick) {
            const job j = it->second;
            it = inflight_.erase(it);
            inflight_keys_.erase(key(j.layer, j.expert));
            done_.insert(key(j.layer, j.expert));
            ++h2d_completed_;
            if (pool_ != nullptr) {
                pool_->promote(j.layer, j.expert);
            }
        } else {
            ++it;
        }
    }
}

bool q4e_expert_h2d::has_pending() const {
    return !queued_.empty() || !inflight_.empty();
}

int q4e_expert_h2d::pending_count() const {
    return (int) queued_.size() + (int) inflight_.size();
}

int64_t q4e_expert_h2d::h2d_issued() const {
    return h2d_issued_;
}

int64_t q4e_expert_h2d::h2d_bytes() const {
    return h2d_bytes_;
}

int64_t q4e_expert_h2d::h2d_completed() const {
    return h2d_completed_;
}

int64_t q4e_expert_h2d::h2d_dropped() const {
    return h2d_dropped_;
}

int64_t q4e_expert_h2d::dedup_skips() const {
    return dedup_skips_;
}

int64_t q4e_expert_h2d::overlap_hits() const {
    return overlap_hits_;
}

int64_t q4e_expert_h2d::accesses() const {
    return accesses_;
}

double q4e_expert_h2d::overlap_ratio() const {
    return (double) overlap_hits_ / (double) (accesses_ > 0 ? accesses_ : 1);
}
