// q4e-expert-warm.h
// live expert page-warm backend (A-14.1)
#pragma once
#include "q4e-expert-cache.h"
#include "q4e-ngram.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

struct ggml_tensor;

// slice of one expert tensor inside one GGUF shard file (read-only mapping)
struct q4e_warm_slice {
    const uint8_t * base   = nullptr; // mapped view of the shard file
    int64_t         offset = 0;       // absolute offset within the file
    int64_t         len    = 0;       // bytes
};

// warms expert weight pages in the OS page cache ahead of CPU compute;
// shares the engine's mmap page cache. place/evict are no-ops (static placement).
class q4e_expert_warm final : public q4e_expert_cache_backend {
public:
    // model_path: first GGUF shard file, or the directory containing the shards
    explicit q4e_expert_warm(const std::string & model_path);
    ~q4e_expert_warm() override; // stops worker, closes mappings
    q4e_expert_warm(const q4e_expert_warm &) = delete;
    q4e_expert_warm & operator=(const q4e_expert_warm &) = delete;

    void bind(q4e_expert_cache & cache); // set owning cache (after cache construction)
    bool init();                         // build expert index + open read-only maps; dense model -> empty index, true
    void start();                        // start worker thread (after bind + init)
    void stop();                         // join worker, close mappings; idempotent

    int n_layers() const { return n_layers_; }

    // hot path (cb_eval): copy ids into an internal slot, enqueue an ids-event
    // (bounded FIFO cap 256; full -> events_dropped++, returns false)
    bool enqueue_host_ids(int layer, const int32_t * ids, int64_t n_ids);
    // hot path, device-resident ids (shape [n_ids, n_tok]): async get into a slot + event
    bool enqueue_device_ids(int layer, const ggml_tensor * ids);

    // real I/O request for one (layer, expert); dedup + bounded FIFO cap 512; full -> io_dropped++
    void warm_hint(int layer, int expert);

    // ngram (PLE table) row prefetch (A-14.4); modes: 0 = off, 1 = shim (accounting only), 2 = full
    bool init_ngram(const std::string & model_path); // parse the PLE table; absent tensor -> false, never fatal
    // hot path: enqueue up to 16 row-warm requests for the token triple (t, t_prev1, t_prev2)
    void ngram_hint(int64_t t, int64_t t_prev1, int64_t t_prev2);
    bool ngram_ready() const;
    int ngram_mode() const;
    void set_ngram_mode(int m); // 0/1/2; invalid clamps to 0 (off)
    int64_t ngram_hints() const;
    int64_t ngram_rows_warmed() const;
    int64_t ngram_rows_dropped() const;
    int64_t ngram_bytes_warmed() const;

    void place(int, int, const char *) override {}
    void evict(int, int, const char *) override {}

    // stats (thread-safe)
    int64_t io_issued() const;
    int64_t io_bytes() const;
    int64_t io_completed() const;
    int64_t io_dropped() const;
    int64_t overlap_hits() const;
    int64_t events_dropped() const;
    int64_t touches_seen() const;
    int64_t hook_skips_weights_device() const;
    void note_hook_skip_weights_device() { hook_skips_weights_device_.fetch_add(1, std::memory_order_relaxed); }

    // test-only seams (deterministic, no thread)
    void set_index_for_test(std::vector<std::vector<std::vector<q4e_warm_slice>>> idx);
    void set_queue_caps_for_test(int ids_cap, int warm_cap);
    void set_ngram_for_test(const uint8_t * base, int64_t row0, int64_t rows, int64_t row_bytes, bool ready);
    void drain_once();    // one worker-loop iteration: <=1 ids event, <=8 warm requests, <=16 ngram rows
    bool has_pending() const;
    void mark_all_slots_busy_for_test();
    int free_slots_for_test() const;

private:
    // one ids event: host ids already in slot, or async get with device event
    struct ids_event {
        int     layer = 0;
        int64_t n     = 0;
        int     slot  = 0;
        bool    have_event = false;
        void *  event = nullptr;
        int64_t off = 0; // per-token element offset within the slot (t * n_ids)
    };

    void worker_loop();
    void do_warm_one(int layer, int expert);
    void do_warm_row(int64_t row);
    int64_t file_size(const std::string & path) const;
    static uint64_t key(int layer, int expert);
    static uint64_t ngram_key(int64_t row);

    q4e_expert_cache * cache_ = nullptr;
    std::string model_path_;
    std::vector<std::string> files_; // resolved shard files
    // [layer][expert+1] (idx 0 = shared expert -1)
    std::vector<std::vector<std::vector<q4e_warm_slice>>> index_;
    int n_layers_ = 0;
    int n_experts_ = 0;
    int64_t max_slice_len_ = 0;

#ifdef _WIN32
    std::vector<HANDLE> file_h_, map_h_, view_;
#endif

    std::mutex qmu_;
    std::condition_variable qcv_;
    std::deque<ids_event> ids_q_;
    std::deque<std::pair<int, int>> warm_q_;
    std::deque<int64_t> ngram_q_; // pending ngram row-warm requests (row ids)
    int ids_cap_ = 256;
    int warm_cap_ = 512;
    int ngram_cap_ = 256;

    std::vector<std::vector<int32_t>> slots_;
    int slot_next_ = 0;
    std::vector<char> slot_busy_; // 1 = slot still referenced by a queued ids event

    std::thread th_;
    std::atomic<bool> stop_ {false};

    // worker-owned (no lock)
    std::unordered_set<uint64_t> inflight_;
    std::unordered_set<uint64_t> warm_done_;
    std::vector<std::vector<int32_t>> prev_;
    std::vector<uint8_t> staging_[2];
    int staging_buf_ = 0;

    std::atomic<int64_t> io_issued_ {0};
    std::atomic<int64_t> io_bytes_ {0};
    std::atomic<int64_t> io_completed_ {0};
    std::atomic<int64_t> io_dropped_ {0};
    std::atomic<int64_t> overlap_hits_ {0};
    std::atomic<int64_t> events_dropped_ {0};
    std::atomic<int64_t> touches_seen_ {0};
    std::atomic<int64_t> hook_skips_weights_device_ {0};

    // ngram (PLE) prefetch state (A-14.4)
    q4e_ngram_table ngram_tbl_;
    bool ngram_ready_ = false;
    const uint8_t * ngram_view_ = nullptr; // mapped shard view holding the PLE tensor
    int64_t ngram_row0_ = 0;               // absolute file offset of row 0
    int64_t ngram_rows_ = 0;
    int64_t ngram_row_bytes_ = 0;
    std::atomic<int> ngram_mode_ {0};      // 0 = off, 1 = shim, 2 = full

    std::atomic<int64_t> ngram_hints_ {0};
    std::atomic<int64_t> ngram_rows_warmed_ {0};
    std::atomic<int64_t> ngram_rows_dropped_ {0};
    std::atomic<int64_t> ngram_bytes_warmed_ {0};
};

// llama_context_params.cb_eval hook: on the MoE expert matmul, warms the selected expert
// pages; fires only when the expert weights are host-resident (GPU layers stay zero-I/O)
// user_data must be a q4e_expert_warm * (or nullptr to no-op)
bool q4e_expert_warm_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
