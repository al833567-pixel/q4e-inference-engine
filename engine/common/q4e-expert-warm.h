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
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
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

    // min-freq gate for the prev-token reuse warm (B-8-4 fix):
    // warm only experts selected at least min_freq() times; 1 = legacy (warm all)
    // worker-owned state: set before start() (server) or before first drain_once (test)
    void set_min_freq(int n);
    int  min_freq() const { return min_freq_; }
    int64_t warm_gated() const;

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
    bool warm_gate(int layer, int expert); // true = skip warm (below min freq); min_freq_ == 1 -> false
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
    std::unordered_map<uint64_t, uint32_t> freq_; // (layer,expert) selection counts; min-freq gate
    int min_freq_ = 1;                            // 1 = legacy (warm all); 2 = gate one-shot experts
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
    std::atomic<int64_t> warm_gated_ {0};

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

// v5 H2D promotion pipeline (A-15.3); deterministic tick clock, pure CPU, no thread
// the real L2 -> L1 copy is the backend job (B-side CUDA); here only queue, state, stats
// caller serializes hint/access/drain (A-15.5 wiring); no lock by design
struct q4e_expert_h2d_backend {
    virtual ~q4e_expert_h2d_backend() = default;
    // copy one expert (layer, expert) of `bytes` into the L1 pool slot
    virtual void copy_expert(int layer, int expert, int64_t bytes) {}
};

struct q4e_h2d_config {
    int64_t expert_bytes = 0;        // bytes per expert copy (job size)
    int     queue_cap    = 64;       // bounded FIFO; full -> drop
    int64_t copy_latency = 1;        // ticks a copy stays inflight after issue
    int     n_layers         = 48;   // range bounds when no pool is bound
    int     n_experts_per_layer = 512;
};

// deterministic state machine: same tick/hint/access/drain sequence => same state (replayable)
class q4e_expert_h2d {
public:
    q4e_expert_h2d(q4e_expert_h2d_backend * backend, const q4e_h2d_config & cfg = q4e_h2d_config());

    void bind(q4e_expert_pool * pool) { pool_ = pool; } // state_of() read before issue, promote() on job done
    q4e_expert_pool * pool() const { return pool_; }

    void tick();  // advance the logical clock by 1 (caller driven; no wall clock)
    int64_t now() const { return tick_; }

    // ngram early promotion, before the actual access; false = dedup skipped or dropped
    bool hint(int layer, int expert, int64_t bytes = -1);
    // actual expert use: counts overlap while inflight, lazy-issues a job on miss
    void access(int layer, int expert);

    // test seam (no thread): issue queued jobs in FIFO, complete ready inflight jobs
    void drain_once();
    bool has_pending() const;
    int  pending_count() const;

    // stats
    int64_t h2d_issued() const;
    int64_t h2d_bytes() const;
    int64_t h2d_completed() const;
    int64_t h2d_dropped() const;
    int64_t dedup_skips() const;
    int64_t overlap_hits() const;
    int64_t accesses() const;
    // overlap_hits / max(1, accesses); window = whole pipeline life
    double overlap_ratio() const;

private:
    struct job {
        int     layer = 0;
        int     expert = 0;
        int64_t bytes = 0;
        int64_t issue_tick = 0;
        int64_t ready_tick = 0;
    };
    static uint64_t key(int layer, int expert);
    bool in_range(int layer, int expert) const;
    bool resident(int layer, int expert) const;
    bool issue(int layer, int expert, int64_t bytes); // enqueue; false = dropped

    q4e_expert_h2d_backend * backend_ = nullptr;
    q4e_h2d_config cfg_;
    q4e_expert_pool * pool_ = nullptr;
    int64_t tick_ = 0;
    std::deque<job> queued_;
    std::unordered_set<uint64_t> queued_keys_;
    std::deque<job> inflight_;
    std::unordered_set<uint64_t> inflight_keys_;
    std::unordered_set<uint64_t> done_; // residency set in standalone mode (no pool)

    int64_t h2d_issued_ = 0;
    int64_t h2d_bytes_ = 0;
    int64_t h2d_completed_ = 0;
    int64_t h2d_dropped_ = 0;
    int64_t dedup_skips_ = 0;
    int64_t overlap_hits_ = 0;
    int64_t accesses_ = 0;
};

// v5 dynamic expert pool (A-15.2b): per-layer slot -> global map over the R-slot buffer
// residency state: q4e_expert_pool (A-15.1, reused); issue/dedup/overlap: q4e_expert_h2d (A-15.3)
// this class owns slot assignment; the CPU row copy (A side) runs as the h2d backend job
class q4e_pool_dynamic : public q4e_expert_h2d_backend {
public:
    explicit q4e_pool_dynamic(q4e_expert_pool & pool);

    void set_h2d(q4e_expert_h2d * h2d); // promotion pipeline; this class is its backend

    // gate/up/down tensors of a layer (pool buffer + full expert); used by the row copy
    void set_layer_tensors(int layer,
                           ggml_tensor * up_pool, ggml_tensor * gate_pool, ggml_tensor * down_pool,
                           ggml_tensor * up_full, ggml_tensor * gate_full, ggml_tensor * down_full);

    // one selected expert at layer; tier at access time: 0 = L1 hit, 1 = L2, 2 = L3
    // miss: slot remap (LRU victim, round-safe) + h2d job (lazy issue; A-15.3 caliber)
    int access(int layer, int expert);

    // slot (0..R-1) of a mapped expert; -1 if not mapped
    int slot_of(int layer, int expert) const;

    void begin_round(int layer); // pool M count + round-safe victim selection
    void end_round(int layer);

    // run the pipeline until no pending job; then sync pending list (L1-completed leaves it)
    void drain();

    int  n_slots() const { return n_slots_; }
    int64_t promotions() const { return promotions_; } // slot remaps issued (first out-of-pool per episode)
    int64_t hits_l1() const { return hits_l1_; }
    int64_t misses_l2() const { return misses_l2_; }
    int64_t misses_l3() const { return misses_l3_; }

    // A-15.3 backend seam: A-side data move = CPU row copy of the expert into the mapped slot
    void copy_expert(int layer, int expert, int64_t bytes) override;

private:
    struct layer_dyn {
        std::vector<int> expert_of_slot; // size R; -1 = free
        std::unordered_map<int, int> slot_of;
        std::vector<int> pending_order;  // mapped, not yet L1; head = first promoted
        explicit layer_dyn(int n) : expert_of_slot((std::size_t) n, -1) {}
    };
    struct layer_tensors {
        ggml_tensor * up_pool = nullptr;
        ggml_tensor * gate_pool = nullptr;
        ggml_tensor * down_pool = nullptr;
        ggml_tensor * up_full = nullptr;
        ggml_tensor * gate_full = nullptr;
        ggml_tensor * down_full = nullptr;
    };
    struct round_state {
        std::set<int> ids;
        bool active = false;
    };

    int pick_victim(int layer) const; // LRU-oldest mapped expert, round set excluded
    int remap(int layer, int expert); // assign slot (victim slot or free); idempotent

    q4e_expert_pool & pool_;
    q4e_expert_h2d * h2d_ = nullptr;
    int n_slots_ = 0;
    std::vector<layer_dyn> layers_;
    std::vector<layer_tensors> tensors_;
    std::vector<round_state> rounds_;
    int64_t promotions_ = 0;
    int64_t hits_l1_ = 0;
    int64_t misses_l2_ = 0;
    int64_t misses_l3_ = 0;
};

// A-side data move: copy one expert gate/up/down rows from the full tensors into the pool slot
void q4e_pool_promote_into(
    ggml_tensor * up_pool, ggml_tensor * gate_pool, ggml_tensor * down_pool,
    ggml_tensor * up_full, ggml_tensor * gate_full, ggml_tensor * down_full,
    int slot, int expert);
