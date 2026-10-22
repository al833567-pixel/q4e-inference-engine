#include "llama-q4e.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#else
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace {

using q4e_clock = std::chrono::steady_clock;

static double q4e_ms(q4e_clock::time_point a, q4e_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static size_t q4e_page_size() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return (size_t) sysconf(_SC_PAGESIZE);
#endif
}

static size_t q4e_free_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    GlobalMemoryStatusEx(&st);
    return (size_t) st.ullAvailPhys;
#else
    return (size_t) sysconf(_SC_AVPHYS_PAGES)*q4e_page_size();
#endif
}

// the mapped model pages this process already touched count as used, but the RAM tier locks those same pages
static size_t q4e_resident_ram() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (size_t) pmc.WorkingSetSize;
    }
#endif
    return 0;
}

// locks whole pages; a page shared by two blocks stays locked until both are unlocked, so unlock only frees the
// pages that lie fully inside the block
struct q4e_locker {
    size_t page   = q4e_page_size();
    size_t locked = 0;
    bool   failed = false;

    void reserve(size_t bytes) {
#ifdef _WIN32
        SIZE_T min_ws = 0;
        SIZE_T max_ws = 0;
        if (GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws, &max_ws)) {
            min_ws += bytes;
            max_ws += bytes;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws, max_ws)) {
                LLAMA_LOG_WARN("q4e: SetProcessWorkingSetSize(%.1f GiB) failed, the RAM tier may not stay locked\n", min_ws/1073741824.0);
            }
        }
#else
        GGML_UNUSED(bytes);
#endif
    }

    bool lock(const void * p, size_t n) {
        if (failed || n == 0) {
            return false;
        }
        const uintptr_t a = (uintptr_t) p & ~(uintptr_t) (page - 1);
        const uintptr_t b = ((uintptr_t) p + n + page - 1) & ~(uintptr_t) (page - 1);
#ifdef _WIN32
        const bool ok = VirtualLock((void *) a, b - a);
#else
        const bool ok = mlock((void *) a, b - a) == 0;
#endif
        if (!ok) {
            failed = true;
            LLAMA_LOG_WARN("q4e: locking the RAM tier stopped at %.2f GiB, the rest stays in the page cache\n", locked/1073741824.0);
            return false;
        }
        locked += n;
        return true;
    }

    void unlock(const void * p, size_t n) {
        const uintptr_t a = ((uintptr_t) p + page - 1) & ~(uintptr_t) (page - 1);
        const uintptr_t b = ((uintptr_t) p + n) & ~(uintptr_t) (page - 1);
        if (b > a) {
#ifdef _WIN32
            VirtualUnlock((void *) a, b - a);
#else
            munlock((void *) a, b - a);
#endif
        }
        locked -= std::min(locked, n);
    }
};

struct q4e_pack_entry {
    int32_t  il;
    int32_t  type;
    int64_t  n_ff;
    int64_t  n_embd;
    int64_t  n_expert;
    int32_t  n_blocks;
    int32_t  pad;
    uint64_t offset;
    uint64_t bytes;
};

constexpr uint64_t Q4E_PACK_MAGIC   = 0x31504b4e57443451ull; // "Q4DWNKP1"
constexpr uint32_t Q4E_PROF_MAGIC   = 0x46503451u;           // "Q4PF"
constexpr size_t   Q4E_PACK_ALIGN   = 65536;
static float       Q4E_DECAY        = 0.995f; // per token, half-life ~140 tokens (env Q4E_DECAY)
static float       Q4E_HYSTERESIS   = 1.1f;   // a block replaces a VRAM block only if it scores this much higher (env Q4E_HYST)
static size_t      Q4E_SWAP_BYTES   = 32u*1048576; // copies per ubatch once the pool is full (env Q4E_SWAP_MB); more
                                                // competes with the per-layer copies for PCIe: 128 MiB was 21.9/22.7 t/s, 8 MiB 23.2/24.0
static size_t      Q4E_RAM_SWAP_BYTES = 0; // blocks read from the SSD into the RAM tier per ubatch (env Q4E_RAM_SWAP_MB), off:
                                                 // on GLM-5.3-Flash with 47 GB of RAM it competed with the CPU blocks for the SSD
constexpr int      Q4E_MAX_BACKENDS = 16;

} // namespace

struct llama_q4e::impl {
    enum { P_GATE = 0, P_UP = 1, P_DOWN = 2, N_PARTS = 3 };

    struct group {
        int32_t dev = -1;
        ggml_type type[N_PARTS];
        int64_t n_embd   = 0;
        int64_t n_ff     = 0;
        int64_t n_expert = 0;
        int32_t n_blocks = 1;
        int64_t f        = 0; // FFN rows of a block

        size_t blk_bytes[N_PARTS] = {};
        size_t slot_bytes = 0;
        size_t host_bytes = 0;

        std::vector<int32_t> layers;
        int32_t n_slots = 0; // usable slots, then n_zero slots of zeros and one of padding read by MMQ
        int32_t n_zero  = 0; // one zero slot per block row of a token: mul_mat_id needs distinct ids within a token
        ggml_tensor * bank[N_PARTS] = {};

        std::vector<int64_t> owner; // slot -> il*n_virt + v, -1 if free
        std::vector<int32_t> free_slots;
    };

    struct device {
        ggml_backend_t backend      = nullptr; // the context's backend, it computes the graphs
        ggml_backend_t side         = nullptr; // a second stream for the copies into the pool
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buf;
        ggml_backend_buffer_ptr staging_buf;
        uint8_t * staging  = nullptr;
        size_t staging_size = 0;
        size_t staging_used = 0;

        ggml_tensor * sel_all = nullptr; // I32 [n_used*max_batch*n_layer]: the selections of all layers, read at once

        // the slot tables of all layers of the device in one tensor, uploaded in one copy after each ubatch
        ggml_tensor *      tables = nullptr;
        std::vector<float> tables_host;
        int64_t            n_table = 0;
    };

    struct layer {
        int32_t ig = -1;
        layer_tensors t;
        const ggml_tensor * src[N_PARTS] = {}; // the model's expert tensors
        const uint8_t * host[N_PARTS] = {};    // block v of part p starts at host[p] + v*blk_bytes[p]
        int64_t n_virt = 0;

        std::vector<float>   table;   // device copy: slot or -1
        int64_t table_off = 0;        // of the layer in the device's tables
        std::vector<int32_t> slot_of; // slot, also while the copy is in flight, -1 if not in VRAM
        std::vector<uint8_t> in_ram;  // the block is locked in the RAM tier
        std::vector<float>    score;  // per expert, decayed use count
        std::vector<uint64_t> last;   // token of the last update of score
        std::vector<uint32_t> uses;   // per expert, uses since the start: the working set
        std::vector<int32_t>  sel;    // the selections of the last ubatch, written by the routing op
        bool dirty = false;
        ggml_tensor * t2[N_PARTS] = {}; // the iGPU tier's copies of the blocks

        mutable bool has_sel = false; // the current graph copies the selections of this layer to sel_all
    };

    struct fill {
        int32_t il;
        int32_t v;
        int32_t slot;
        size_t  off; // in the staging buffer of the device
    };

    struct blk_ref {
        float   s;
        int32_t il;
        int32_t v;
    };

    struct ram_move {
        int32_t il;
        int32_t v;
        bool    lock; // false: unlock
    };

    struct counters {
        uint64_t tokens   = 0;
        uint64_t ubatches = 0;
        double   ms_wall  = 0; // begin_ubatch to the end of the compute
        double   ms_promo = 0; // end_ubatch: selections, plan, staging, table uploads
        uint64_t blk_vram = 0;
        uint64_t blk_ram  = 0;
        uint64_t blk_ssd  = 0;
        uint64_t promoted = 0;
        uint64_t promoted_bytes = 0;
        int64_t  sched_copy_us[Q4E_MAX_BACKENDS]    = {};
        int64_t  sched_compute_us[Q4E_MAX_BACKENDS] = {};
    };

    llama_q4e_params params;
    const llama_model & model;
    ggml_backend_t backend_cpu;

    std::vector<device> devices;
    std::vector<group>  groups;
    std::vector<layer>  layers;

    // repacked down blocks
    std::unique_ptr<llama_file> pack_file;
    std::unique_ptr<llama_mmap> pack_map;
    ggml_backend_buffer_ptr     pack_buf;
    ggml_context_ptr            ctx_host;

    q4e_locker locker;
    size_t ram_budget = 0;

    // the iGPU tier (see the constructor): its copies of the blocks
    ggml_backend_t          tier2 = nullptr;
    ggml_context_ptr        ctx_t2;
    ggml_backend_buffer_ptr buf_t2;

    std::vector<fill> pending;
    std::vector<ram_move> ram_moves; // done by the worker after the copies, the in_ram flags are set when planned
    std::vector<int32_t> ids;

    // the copies into VRAM: a worker reads the blocks into the staging buffers and queues them on the second
    // stream while the next ubatch computes
    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv;
    bool job  = false;
    bool quit = false;
    bool busy = false;

    uint64_t token = 0; // tokens seen, the clock of the scores
    q4e_clock::time_point t_begin;
    q4e_clock::time_point t_saved = q4e_clock::now();
    counters all;    // since the start, decode ubatches only
    counters window; // since the last log line
    mutable uint64_t window_logs = 0;
    int64_t sched_prev_copy[Q4E_MAX_BACKENDS]    = {};
    int64_t sched_prev_compute[Q4E_MAX_BACKENDS] = {};

    impl(const llama_model & model, ggml_backend_t backend_cpu, const std::vector<ggml_backend_t> & backends, const llama_q4e_params & params) :
            params(params), model(model), backend_cpu(backend_cpu), layers(model.layers.size()) {
        const auto & hp = model.hparams;
        if (hp.n_expert == 0) {
            throw std::runtime_error("q4e requires a MoE model");
        }

        // tuning knobs for experiments
        if (const char * e = getenv("Q4E_DECAY")) {
            Q4E_DECAY = (float) atof(e);
        }
        if (const char * e = getenv("Q4E_HYST")) {
            Q4E_HYSTERESIS = (float) atof(e);
        }
        if (const char * e = getenv("Q4E_RAM_SWAP_MB")) {
            Q4E_RAM_SWAP_BYTES = (size_t) atoi(e)*1048576;
        }
        if (const char * e = getenv("Q4E_SWAP_MB")) {
            Q4E_SWAP_BYTES = (size_t) atoi(e)*1048576;
        }

        // iGPU + dGPU: an integrated GPU among the backends computes the blocks that are not in VRAM in place of the
        // CPU, from its own copy of every block (uploaded at load, no RAM tier). Q4E_TIER2=0 keeps the CPU
        const char * tier2_env = getenv("Q4E_TIER2");
        if (!tier2_env || atoi(tier2_env) != 0) {
            for (ggml_backend_t b : backends) {
                if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    tier2 = b;
                    break;
                }
            }
        }
        for (ggml_backend_t b : backends) {
            const auto type = ggml_backend_dev_type(ggml_backend_get_device(b));
            if (b != tier2 && (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
                auto & d = devices.emplace_back();
                d.backend = b;
                d.buft    = ggml_backend_get_default_buffer_type(b);
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("q4e requires a GPU backend");
        }

        // layers whose experts stay in host memory and whose other weights are on a GPU
        for (size_t il = 0; il < model.layers.size(); ++il) {
            const llama_layer & ll = model.layers[il];
            if (ll.ffn_gate_up_exps || !ll.ffn_gate_exps || !ll.ffn_up_exps || !ll.ffn_down_exps) {
                continue;
            }
            const ggml_tensor * src[N_PARTS] = { ll.ffn_gate_exps, ll.ffn_up_exps, ll.ffn_down_exps };
            bool ok = true;
            for (const ggml_tensor * t : src) {
                ok = ok && t->buffer && ggml_backend_buffer_is_host(t->buffer) && ggml_is_contiguous(t) &&
                     ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            }
            if (!ok) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) {
                return ggml_backend_get_device(d.backend) == model.dev_layer(il);
            });
            if (it_dev == devices.end()) {
                continue;
            }

            const int64_t n_embd   = src[P_GATE]->ne[0];
            const int64_t n_ff     = src[P_GATE]->ne[1];
            const int64_t n_expert = src[P_GATE]->ne[2];
            const int32_t n_blocks = pick_blocks(src, n_ff);

            const int32_t dev = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) {
                return g.dev == dev && g.n_embd == n_embd && g.n_ff == n_ff && g.n_expert == n_expert && g.n_blocks == n_blocks &&
                       g.type[P_GATE] == src[P_GATE]->type && g.type[P_UP] == src[P_UP]->type && g.type[P_DOWN] == src[P_DOWN]->type;
            });
            if (it == groups.end()) {
                group & g = groups.emplace_back();
                g.dev      = dev;
                g.n_embd   = n_embd;
                g.n_ff     = n_ff;
                g.n_expert = n_expert;
                g.n_blocks = n_blocks;
                g.f        = n_ff / n_blocks;
                for (int p = 0; p < N_PARTS; ++p) {
                    g.type[p]      = src[p]->type;
                    g.blk_bytes[p] = ggml_nbytes(src[p]) / (n_expert*n_blocks);
                    g.slot_bytes  += g.blk_bytes[p];
                }
                it = groups.end() - 1;
            }
            it->layers.push_back((int32_t) il);
            for (const ggml_tensor * t : src) {
                it->host_bytes += ggml_nbytes(t);
            }

            layer & l = layers[il];
            l.ig     = (int32_t) (it - groups.begin());
            l.n_virt = n_expert*n_blocks;
            for (int p = 0; p < N_PARTS; ++p) {
                l.src[p] = src[p];
            }
        }
        if (groups.empty()) {
            throw std::runtime_error("q4e: no MoE layer keeps its experts in host memory (use --cpu-moe)");
        }

        const bool need_pack = std::any_of(groups.begin(), groups.end(), [](const group & g) { return g.n_blocks > 1; });
        if (tier2) {
            alloc_tier2();
        }
        if (need_pack) {
            load_pack();
        }
        init_host_views();
        init_pool();
        init_side();
        worker = std::thread([this]() { work(); });
        load_profile();
        if (tier2) {
            init_tier2();
        } else {
            init_ram_tier();
        }

        for (const group & g : groups) {
            LLAMA_LOG_INFO("q4e: %2zu layers %s/%s/%s, %d blocks of %lld rows per expert, block %.3f MiB: VRAM %d blocks (%.2f GiB, %.1f%%)\n",
                g.layers.size(), ggml_type_name(g.type[P_GATE]), ggml_type_name(g.type[P_UP]), ggml_type_name(g.type[P_DOWN]),
                g.n_blocks, (long long) g.f, g.slot_bytes/1048576.0, g.n_slots, g.n_slots*(double) g.slot_bytes/1073741824.0,
                100.0*g.n_slots/(g.layers.size()*g.n_expert*g.n_blocks));
        }
    }

    ~impl() {
        finish_fills();
        {
            std::lock_guard<std::mutex> lk(mtx);
            quit = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        log_stats(all, "total", true);
        save_profile();
        for (device & d : devices) {
            if (d.side) {
                ggml_backend_free(d.side);
            }
        }
    }

    int32_t pick_blocks(const ggml_tensor * const * src, int64_t n_ff) const {
        // a block is valid when its down columns are whole quant blocks
        auto valid = [&](int32_t b) {
            if (b < 1 || n_ff % b != 0) {
                return false;
            }
            const int64_t f = n_ff / b;
            return f % ggml_blck_size(src[P_DOWN]->type) == 0;
        };
        if (params.blocks > 0) {
            if (!valid(params.blocks) || (params.blocks > 1 && params.pack_path.empty())) {
                LLAMA_LOG_WARN("q4e: %d blocks per expert do not fit the down rows of %lld (%s), using 1\n",
                    params.blocks, (long long) n_ff, ggml_type_name(src[P_DOWN]->type));
                return 1;
            }
            return params.blocks;
        }
        if (params.pack_path.empty()) {
            return 1;
        }
        // auto: the smallest number of blocks that keeps a block within 1 MiB
        size_t expert_bytes = 0;
        for (int p = 0; p < N_PARTS; ++p) {
            expert_bytes += ggml_nbytes(src[p]) / src[p]->ne[2];
        }
        for (int32_t b = 1; b <= 16; ++b) {
            if (valid(b) && expert_bytes / b <= 1024*1024) {
                return b;
            }
        }
        return 1;
    }

    // the down blocks are column slices, so they are written once to a file with one contiguous block after another
    void load_pack() {
        std::vector<q4e_pack_entry> want;
        uint64_t offset = 0;
        auto align = [](uint64_t x) { return (x + Q4E_PACK_ALIGN - 1) / Q4E_PACK_ALIGN * Q4E_PACK_ALIGN; };
        size_t header = sizeof(uint64_t) + sizeof(uint32_t);
        for (const group & g : groups) {
            if (g.n_blocks > 1) {
                header += g.layers.size()*sizeof(q4e_pack_entry);
            }
        }
        offset = align(header);
        for (const group & g : groups) {
            if (g.n_blocks == 1) {
                continue;
            }
            for (int32_t il : g.layers) {
                const ggml_tensor * t = layers[il].src[P_DOWN];
                q4e_pack_entry e = {};
                e.il       = il;
                e.type     = t->type;
                e.n_ff     = t->ne[0];
                e.n_embd   = t->ne[1];
                e.n_expert = t->ne[2];
                e.n_blocks = g.n_blocks;
                e.offset   = offset;
                e.bytes    = ggml_nbytes(t);
                offset     = align(offset + e.bytes);
                want.push_back(e);
            }
        }
        const uint64_t total = offset;

        // the iGPU mode keeps other layers on the dGPU (-ot): its own file, so that switching modes does not rewrite it
        const std::string path = tier2 ? params.pack_path + ".igpu" : params.pack_path;
        bool valid = false;
        {
            FILE * fp = ggml_fopen(path.c_str(), "rb");
            if (fp) {
                uint64_t magic = 0;
                uint32_t n = 0;
                if (fread(&magic, sizeof(magic), 1, fp) == 1 && fread(&n, sizeof(n), 1, fp) == 1 && magic == Q4E_PACK_MAGIC && n == want.size()) {
                    std::vector<q4e_pack_entry> have(n);
                    valid = fread(have.data(), sizeof(q4e_pack_entry), n, fp) == n &&
                            memcmp(have.data(), want.data(), n*sizeof(q4e_pack_entry)) == 0;
                }
                fseek(fp, 0, SEEK_END);
                valid = valid && (uint64_t) _ftelli64_or_tell(fp) == total;
                fclose(fp);
            }
        }

        if (!valid) {
            const auto t0 = q4e_clock::now();
            LLAMA_LOG_INFO("q4e: writing the down blocks to %s (%.2f GiB, once) ...\n", path.c_str(), total/1073741824.0);
            {
                llama_file out(path.c_str(), "wb");
                const uint64_t magic = Q4E_PACK_MAGIC;
                const uint32_t n     = (uint32_t) want.size();
                out.write_raw(&magic, sizeof(magic));
                out.write_raw(&n, sizeof(n));
                out.write_raw(want.data(), want.size()*sizeof(q4e_pack_entry));
                std::vector<uint8_t> buf;
                uint64_t pos = header;
                for (const q4e_pack_entry & e : want) {
                    if (pos < e.offset) {
                        std::vector<uint8_t> zero(e.offset - pos, 0);
                        out.write_raw(zero.data(), zero.size());
                        pos = e.offset;
                    }
                    const ggml_tensor * t = layers[e.il].src[P_DOWN];
                    const size_t row  = t->nb[1];                         // one down row of an expert: n_ff columns
                    const size_t brow = ggml_row_size(t->type, e.n_ff/e.n_blocks); // the columns of one block
                    buf.resize(t->nb[2]);
                    for (int64_t x = 0; x < e.n_expert; ++x) {
                        const uint8_t * src = (const uint8_t *) t->data + x*t->nb[2];
                        uint8_t * dst = buf.data();
                        for (int32_t b = 0; b < e.n_blocks; ++b) {
                            for (int64_t r = 0; r < e.n_embd; ++r) {
                                memcpy(dst, src + r*row + b*brow, brow);
                                dst += brow;
                            }
                        }
                        out.write_raw(buf.data(), buf.size());
                    }
                    pos += e.bytes;
                }
                if (pos < total) {
                    std::vector<uint8_t> zero(total - pos, 0);
                    out.write_raw(zero.data(), zero.size());
                }
            }
            LLAMA_LOG_INFO("q4e: wrote the down blocks in %.1f s.\n", q4e_ms(t0, q4e_clock::now())/1000.0);
        }

        pack_file = std::make_unique<llama_file>(path.c_str(), "rb");
        pack_map  = std::make_unique<llama_mmap>(pack_file.get(), 0, false);
        pack_buf.reset(ggml_backend_cpu_buffer_from_ptr(pack_map->addr(), pack_map->size()));
        ggml_backend_buffer_set_usage(pack_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (const q4e_pack_entry & e : want) {
            layers[e.il].host[P_DOWN] = (const uint8_t *) pack_map->addr() + e.offset;
        }
    }

    static int64_t _ftelli64_or_tell(FILE * fp) {
#ifdef _WIN32
        return _ftelli64(fp);
#else
        return ftello(fp);
#endif
    }

    // the host experts seen as blocks: gate/up rows are already contiguous, down comes from the pack
    void init_host_views() {
        size_t n_tensors = 0;
        for (const group & g : groups) {
            n_tensors += g.layers.size()*N_PARTS;
        }
        ggml_init_params ip = { n_tensors*ggml_tensor_overhead(), nullptr, true };
        ctx_host.reset(ggml_init(ip));

        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                ggml_tensor * gate = const_cast<ggml_tensor *>(l.src[P_GATE]);
                ggml_tensor * up   = const_cast<ggml_tensor *>(l.src[P_UP]);
                ggml_tensor * down = const_cast<ggml_tensor *>(l.src[P_DOWN]);

                l.t.gate_host = ggml_view_3d(ctx_host.get(), gate, g.n_embd, g.f, l.n_virt, gate->nb[1], gate->nb[1]*g.f, 0);
                l.t.up_host   = ggml_view_3d(ctx_host.get(), up,   g.n_embd, g.f, l.n_virt, up->nb[1],   up->nb[1]*g.f,   0);
                ggml_backend_view_init(l.t.gate_host);
                ggml_backend_view_init(l.t.up_host);
                ggml_format_name(l.t.gate_host, "q4e.host.%s", gate->name);
                ggml_format_name(l.t.up_host,   "q4e.host.%s", up->name);

                l.host[P_GATE] = (const uint8_t *) gate->data;
                l.host[P_UP]   = (const uint8_t *) up->data;

                if (g.n_blocks == 1) {
                    l.t.down_host  = down;
                    l.host[P_DOWN] = (const uint8_t *) down->data;
                } else {
                    ggml_tensor * t = ggml_new_tensor_3d(ctx_host.get(), down->type, g.f, g.n_embd, l.n_virt);
                    ggml_format_name(t, "q4e.host.%s", down->name);
                    GGML_ASSERT(ggml_nbytes(t) == ggml_nbytes(down));
                    if (ggml_backend_tensor_alloc(pack_buf.get(), t, (void *) l.host[P_DOWN]) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("q4e: cannot map the down blocks");
                    }
                    l.t.down_host = t;
                }
                l.t.n_blocks = g.n_blocks;
            }
        }
    }

    // the VRAM tier: what is free after the weights, the KV cache and the compute buffers, less the reserve
    void init_pool() {
        std::vector<size_t> budget(devices.size(), 0);
        std::vector<size_t> host_bytes(devices.size(), 0);
        for (const group & g : groups) {
            host_bytes[g.dev] += g.host_bytes;
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            if (host_bytes[id] == 0) {
                continue;
            }
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(devices[id].backend), &free, &total);
            if (params.vram_mib >= 0) {
                budget[id] = (size_t) params.vram_mib*1048576;
            } else {
                const size_t reserve = (size_t) params.reserve_mib*1048576;
                budget[id] = free > reserve ? free - reserve : 0;
            }
            LLAMA_LOG_INFO("q4e: %s: %.2f GiB free of %.2f GiB, VRAM tier %.2f GiB\n", ggml_backend_name(devices[id].backend),
                free/1073741824.0, total/1073741824.0, budget[id]/1073741824.0);
        }

        std::vector<size_t> n_tensors(devices.size(), 0);
        for (group & g : groups) {
            const size_t tables = g.layers.size()*GGML_PAD(g.n_expert*g.n_blocks*sizeof(float), 512);
            const size_t share  = (size_t) ((double) budget[g.dev]*g.host_bytes/host_bytes[g.dev]);
            const int64_t max_slots = (int64_t) g.layers.size()*g.n_expert*g.n_blocks;
            g.n_zero = (int32_t) (model.hparams.n_expert_used_max()*g.n_blocks);
            int64_t n = share > tables ? (int64_t) ((share - tables)/(g.slot_bytes + 3*512)) - g.n_zero - 1 : 0;
            g.n_slots = (int32_t) std::max<int64_t>(0, std::min(n, max_slots));
            n_tensors[g.dev] += N_PARTS + g.layers.size();
            for (int32_t il : g.layers) {
                layers[il].table_off = devices[g.dev].n_table;
                devices[g.dev].n_table += layers[il].n_virt;
            }
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] == 0) {
                continue;
            }
            ggml_init_params ip = { (n_tensors[id] + 2)*ggml_tensor_overhead(), nullptr, true };
            devices[id].ctx.reset(ggml_init(ip));
            devices[id].sel_all = ggml_new_tensor_1d(devices[id].ctx.get(), GGML_TYPE_I32,
                (int64_t) model.hparams.n_expert_used_max()*params.max_batch*layers.size());
            ggml_set_name(devices[id].sel_all, "q4e.selections");
            devices[id].tables = ggml_new_tensor_1d(devices[id].ctx.get(), GGML_TYPE_F32, devices[id].n_table);
            ggml_set_name(devices[id].tables, "q4e.tables");
            devices[id].tables_host.assign(devices[id].n_table, -1.0f);
        }

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            ggml_context * ctx = devices[g.dev].ctx.get();
            const int64_t ne[N_PARTS][2] = { { g.n_embd, g.f }, { g.n_embd, g.f }, { g.f, g.n_embd } };
            for (int p = 0; p < N_PARTS; ++p) {
                g.bank[p] = ggml_new_tensor_3d(ctx, g.type[p], ne[p][0], ne[p][1], g.n_slots + g.n_zero + 1);
                GGML_ASSERT(g.bank[p]->nb[2] == g.blk_bytes[p]);
                ggml_format_name(g.bank[p], "q4e.pool.%zu.%d", ig, p);
            }
            g.owner.assign(g.n_slots, -1);
            g.free_slots.resize(g.n_slots);
            for (int32_t s = 0; s < g.n_slots; ++s) {
                g.free_slots[s] = g.n_slots - 1 - s;
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.t.table = ggml_view_2d(ctx, devices[g.dev].tables, 1, l.n_virt, sizeof(float), l.table_off*sizeof(float));
                ggml_format_name(l.t.table, "q4e.table-%d", il);
                l.t.gate_pool   = g.bank[P_GATE];
                l.t.up_pool     = g.bank[P_UP];
                l.t.down_pool   = g.bank[P_DOWN];
                l.t.zero_slot   = g.n_slots;
                l.t.backend_gpu = devices[g.dev].backend;
                l.t.sel_all     = devices[g.dev].sel_all;
                l.table.assign(l.n_virt, -1.0f);
                l.slot_of.assign(l.n_virt, -1);
                l.in_ram.assign(l.n_virt, 0);
                l.score.assign(g.n_expert, 0.0f);
                l.last.assign(g.n_expert, 0);
                l.uses.assign(g.n_expert, 0);
                l.sel.assign((size_t) model.hparams.n_expert_used_max()*params.max_batch, -1);
                l.t.n_expert   = g.n_expert;
                l.t.table_host = l.table.data();
                l.t.sel_host   = l.sel.data();
            }
        }

        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
            if (!d.buf) {
                throw std::runtime_error("q4e: failed to allocate the VRAM tier");
            }
            ggml_backend_buffer_clear(d.buf.get(), 0);
            ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                upload_table(layers[il]);
            }
        }
    }

    void init_side() {
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (!d.buf) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            d.side = ggml_backend_dev_init(dev, nullptr);
            if (!d.side) {
                throw std::runtime_error("q4e: cannot create a second stream");
            }
            size_t max_slot = 0;
            for (const group & g : groups) {
                if (g.dev == (int32_t) id) {
                    max_slot = std::max(max_slot, g.slot_bytes);
                }
            }
            d.staging_size = std::max<size_t>({ (size_t) 256u*1048576, 8*max_slot, 2*Q4E_SWAP_BYTES });
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
            ggml_backend_buffer_t sb = host_buft ? ggml_backend_buft_alloc_buffer(host_buft, d.staging_size) : nullptr;
            if (!sb) {
                sb = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), d.staging_size);
            }
            d.staging_buf.reset(sb);
            d.staging = (uint8_t *) ggml_backend_buffer_get_base(sb);
        }
    }

    // the dirty tables of each device: one copy of the range that holds them
    void upload_tables() {
        std::vector<int64_t> lo(devices.size(), INT64_MAX);
        std::vector<int64_t> hi(devices.size(), 0);
        for (layer & l : layers) {
            if (l.ig < 0 || !l.dirty) {
                continue;
            }
            const size_t id = (size_t) groups[l.ig].dev;
            device & d = devices[id];
            memcpy(d.tables_host.data() + l.table_off, l.table.data(), l.n_virt*sizeof(float));
            lo[id] = std::min(lo[id], l.table_off);
            hi[id] = std::max(hi[id], l.table_off + l.n_virt);
            l.dirty = false;
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            if (lo[id] < hi[id]) {
                ggml_backend_tensor_set(devices[id].tables, devices[id].tables_host.data() + lo[id], lo[id]*sizeof(float), (hi[id] - lo[id])*sizeof(float));
            }
        }
    }

    void upload_table(layer & l) {
        ggml_backend_tensor_set(l.t.table, l.table.data(), 0, l.table.size()*sizeof(float));
        l.dirty = false;
    }

    // score of expert e of layer l at the current token
    float score_now(const layer & l, int64_t e) const {
        return l.score[e]*std::pow(Q4E_DECAY, (float) (token - l.last[e]));
    }

    void touch(layer & l, int64_t e) {
        l.score[e] = score_now(l, e) + 1.0f;
        l.last[e]  = token;
        l.uses[e]++;
    }

    // the working set since the start: the experts used, and the bytes of the most used experts that serve 90/95/99%
    // of the uses
    std::string working_set() const {
        std::vector<std::pair<uint32_t, size_t>> v; // (uses, bytes)
        uint64_t total = 0;
        for (const layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            const size_t bytes = groups[l.ig].slot_bytes*groups[l.ig].n_blocks;
            for (uint32_t u : l.uses) {
                if (u > 0) {
                    v.push_back({ u, bytes });
                    total += u;
                }
            }
        }
        if (total == 0) {
            return "";
        }
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        size_t used = 0;
        for (const auto & x : v) {
            used += x.second;
        }
        const double cut[3] = { 0.90, 0.95, 0.99 };
        size_t at[3] = { 0, 0, 0 };
        uint64_t acc = 0;
        size_t bytes = 0;
        int k = 0;
        for (const auto & x : v) {
            acc   += x.first;
            bytes += x.second;
            while (k < 3 && acc >= cut[k]*total) {
                at[k++] = bytes;
            }
        }
        return format(" | working set %zu experts %.2f GiB, 90%% of uses in %.2f GiB, 95%% in %.2f GiB, 99%% in %.2f GiB",
            v.size(), used/1073741824.0, at[0]/1073741824.0, at[1]/1073741824.0, at[2]/1073741824.0);
    }

    void block_ptrs(const layer & l, const group & g, int32_t v, const uint8_t * ptr[N_PARTS]) const {
        for (int p = 0; p < N_PARTS; ++p) {
            ptr[p] = l.host[p] + (size_t) v*g.blk_bytes[p];
        }
    }

    bool ram_lock(layer & l, const group & g, int32_t v) {
        if (l.in_ram[v]) {
            return true;
        }
        if (locker.locked + g.slot_bytes > ram_budget) {
            return false;
        }
        const uint8_t * ptr[N_PARTS];
        block_ptrs(l, g, v, ptr);
        for (int p = 0; p < N_PARTS; ++p) {
            if (!locker.lock(ptr[p], g.blk_bytes[p])) {
                return false;
            }
        }
        l.in_ram[v] = 1;
        return true;
    }

    // the same, done by the worker with the next copies: the system calls left the decode thread. ram_used counts
    // the planned RAM tier, locker.locked follows when the worker gets to it
    size_t ram_used = 0;

    void ram_lock_later(int32_t il, int32_t v) {
        layer & l = layers[il];
        const group & g = groups[l.ig];
        if (tier2 || l.in_ram[v] || ram_used + g.slot_bytes > ram_budget) {
            return;
        }
        l.in_ram[v] = 1;
        ram_used += g.slot_bytes;
        ram_moves.push_back({ il, v, true });
    }

    void ram_unlock_later(int32_t il, int32_t v) {
        layer & l = layers[il];
        if (tier2 || !l.in_ram[v]) {
            return;
        }
        l.in_ram[v] = 0;
        ram_used -= std::min(ram_used, groups[l.ig].slot_bytes);
        ram_moves.push_back({ il, v, false });
    }

    void ram_unlock(layer & l, const group & g, int32_t v) {
        if (!l.in_ram[v]) {
            return;
        }
        const uint8_t * ptr[N_PARTS];
        block_ptrs(l, g, v, ptr);
        for (int p = 0; p < N_PARTS; ++p) {
            locker.unlock(ptr[p], g.blk_bytes[p]);
        }
        l.in_ram[v] = 0;
    }

    // plan the copy of block v of layer il into slot: issue_fills hands it to the worker, the table points at it
    // after finish_fills
    void start_fill(int32_t il, int32_t v, int32_t slot) {
        layer & l = layers[il];
        group & g = groups[l.ig];
        device & d = devices[g.dev];
        l.slot_of[v] = slot;
        g.owner[slot] = (int64_t) il*l.n_virt + v;
        pending.push_back({ il, v, slot, d.staging_used });
        d.staging_used += g.slot_bytes;
    }

    // worker: read the planned blocks into the staging buffers and queue the copies on the second stream
    void copy_fills() {
        // the copies go out in chunks, each one done before the next is queued: a long queue on the copy engine
        // delayed every other transfer of the device behind it, the logits after each ubatch included (on a PCIe 3.0
        // x4 link 128 MiB took 40 ms). Q4E_DMA_CHUNK_KB, 0 = queue everything at once
        static const size_t chunk = [] {
            const char * e = getenv("Q4E_DMA_CHUNK_KB");
            return (size_t) (e ? atoi(e) : 256)*1024;
        }();
        std::vector<size_t> queued(devices.size(), 0);
        for (const fill & f : pending) {
            const layer & l = layers[f.il];
            const group & g = groups[l.ig];
            const device & d = devices[g.dev];
            const uint8_t * ptr[N_PARTS];
            block_ptrs(l, g, f.v, ptr);
            size_t off = f.off;
            size_t & q = queued[g.dev];
            for (int p = 0; p < N_PARTS; ++p) {
                memcpy(d.staging + off, ptr[p], g.blk_bytes[p]);
                ggml_backend_tensor_set_async(d.side, g.bank[p], d.staging + off, (size_t) f.slot*g.blk_bytes[p], g.blk_bytes[p]);
                off += g.blk_bytes[p];
                q += g.blk_bytes[p];
                if (chunk > 0 && q >= chunk) {
                    ggml_backend_synchronize(d.side);
                    q = 0;
                }
            }
        }
    }

    // worker: the RAM tier changes - unlocking first frees the budget, locking reads the block from the SSD
    void do_ram_moves() {
        for (int pass = 0; pass < 2; ++pass) {
            for (const ram_move & m : ram_moves) {
                if (m.lock != (pass == 1)) {
                    continue;
                }
                const layer & l = layers[m.il];
                const group & g = groups[l.ig];
                const uint8_t * ptr[N_PARTS];
                block_ptrs(l, g, m.v, ptr);
                if (m.lock) {
                    llama_memory_ranges r;
                    for (int p = 0; p < N_PARTS; ++p) {
                        r.push_back({ ptr[p], g.blk_bytes[p] });
                    }
                    llama_prefetch(r);
                }
                for (int p = 0; p < N_PARTS; ++p) {
                    if (m.lock) {
                        locker.lock(ptr[p], g.blk_bytes[p]);
                    } else {
                        locker.unlock(ptr[p], g.blk_bytes[p]);
                    }
                }
            }
        }
    }

    void work() {
        std::unique_lock<std::mutex> lk(mtx);
        while (true) {
            cv.wait(lk, [this]() { return job || quit; });
            if (quit) {
                return;
            }
            lk.unlock();
            copy_fills();
            do_ram_moves();
            lk.lock();
            job = false;
            cv.notify_all();
        }
    }

    bool worker_idle() {
        std::lock_guard<std::mutex> lk(mtx);
        return !job;
    }

    // the RAM tier: the best scored blocks read from the SSD in this ubatch replace the worst scored RAM blocks
    void plan_ram(std::vector<blk_ref> & cand) {
        std::sort(cand.begin(), cand.end(), [](const blk_ref & a, const blk_ref & b) {
            return a.s != b.s ? a.s > b.s : a.il != b.il ? a.il < b.il : a.v < b.v;
        });
        cand.erase(std::unique(cand.begin(), cand.end(), [](const blk_ref & a, const blk_ref & b) { return a.il == b.il && a.v == b.v; }), cand.end());

        size_t budget = Q4E_RAM_SWAP_BYTES;
        size_t n_try = 0;
        for (; n_try < cand.size(); ++n_try) {
            const size_t bytes = groups[layers[cand[n_try].il].ig].slot_bytes;
            if (bytes > budget) {
                break;
            }
            budget -= bytes;
        }
        if (n_try == 0) {
            return;
        }

        // the worst scored blocks of the RAM tier
        std::vector<blk_ref> victims;
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            const int32_t nb = groups[l.ig].n_blocks;
            for (int32_t v = 0; v < l.n_virt; ++v) {
                if (l.in_ram[v]) {
                    victims.push_back({ score_now(l, v / nb), (int32_t) il, v });
                }
            }
        }
        const size_t k = std::min(victims.size(), n_try);
        std::partial_sort(victims.begin(), victims.begin() + k, victims.end(), [](const blk_ref & a, const blk_ref & b) { return a.s < b.s; });

        size_t planned = locker.locked;
        size_t iv = 0;
        for (size_t i = 0; i < n_try; ++i) {
            layer & l = layers[cand[i].il];
            const size_t bytes = groups[l.ig].slot_bytes;
            if (planned + bytes > ram_budget) {
                if (iv >= k || cand[i].s <= victims[iv].s*Q4E_HYSTERESIS + 0.5f) {
                    break;
                }
                layer & lv = layers[victims[iv].il];
                lv.in_ram[victims[iv].v] = 0;
                ram_moves.push_back({ victims[iv].il, victims[iv].v, false });
                planned -= std::min(planned, groups[lv.ig].slot_bytes);
                iv++;
            }
            l.in_ram[cand[i].v] = 1;
            ram_moves.push_back({ cand[i].il, cand[i].v, true });
            planned += bytes;
        }
    }

    void issue_fills() {
        if (pending.empty() && ram_moves.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            job = true;
        }
        busy = true;
        cv.notify_all();
    }

    void finish_fills() {
        if (busy) {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this]() { return !job; });
            busy = false;
        } else if (!pending.empty() || !ram_moves.empty()) {
            copy_fills(); // planned but not handed to the worker (start-up)
            do_ram_moves();
        }
        ram_moves.clear();
        if (pending.empty()) {
            return;
        }
        for (device & d : devices) {
            if (d.side) {
                ggml_backend_synchronize(d.side);
            }
            d.staging_used = 0;
        }
        for (const fill & f : pending) {
            layer & l = layers[f.il];
            l.table[f.v] = (float) f.slot;
            l.dirty = true;
            // exclusive tiers: a block in VRAM leaves the RAM tier
            ram_unlock_later(f.il, f.v);
        }
        pending.clear();
    }

    void evict(group & g, int32_t slot) {
        const int64_t key = g.owner[slot];
        if (key < 0) {
            return;
        }
        const int32_t il = (int32_t) (key / layers[g.layers[0]].n_virt);
        layer & l = layers[il];
        const int32_t v = (int32_t) (key % l.n_virt);
        l.table[v]   = -1.0f;
        l.slot_of[v] = -1;
        l.dirty      = true;
        g.owner[slot] = -1;
        // back to the RAM tier if it fits, else it is read from the SSD when needed
        ram_lock_later(il, v);
    }

    // the starting state: the pool gets the best scored blocks of the profile, if any
    void load_profile() {
        if (params.profile_path.empty()) {
            return;
        }
        FILE * fp = ggml_fopen(params.profile_path.c_str(), "rb");
        if (!fp) {
            LLAMA_LOG_INFO("q4e: no profile at %s yet, the tiers start empty and learn\n", params.profile_path.c_str());
            return;
        }
        uint32_t magic = 0;
        uint32_t n_layer = 0;
        bool ok = fread(&magic, 4, 1, fp) == 1 && fread(&n_layer, 4, 1, fp) == 1 && magic == Q4E_PROF_MAGIC && n_layer == layers.size();
        for (uint32_t il = 0; ok && il < n_layer; ++il) {
            uint32_t n = 0;
            ok = fread(&n, 4, 1, fp) == 1;
            std::vector<float> s(n);
            ok = ok && fread(s.data(), sizeof(float), n, fp) == n;
            if (ok && layers[il].ig >= 0 && n == layers[il].score.size()) {
                layers[il].score = s;
            }
        }
        fclose(fp);
        if (!ok) {
            LLAMA_LOG_WARN("q4e: ignoring the profile %s (other model)\n", params.profile_path.c_str());
            for (layer & l : layers) {
                std::fill(l.score.begin(), l.score.end(), 0.0f);
            }
            return;
        }

        const auto t0 = q4e_clock::now();
        size_t bytes = 0;
        for (group & g : groups) {
            struct cand { float s; int32_t il; int32_t v; };
            std::vector<cand> c;
            for (int32_t il : g.layers) {
                const layer & l = layers[il];
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    if (l.score[e] > 0.0f) {
                        for (int32_t b = 0; b < g.n_blocks; ++b) {
                            c.push_back({ l.score[e], il, (int32_t) (e*g.n_blocks + b) });
                        }
                    }
                }
            }
            std::sort(c.begin(), c.end(), [](const cand & a, const cand & b) { return a.s > b.s; });
            device & d = devices[g.dev];
            for (const cand & x : c) {
                if (g.free_slots.empty()) {
                    break;
                }
                if (d.staging_used + g.slot_bytes > d.staging_size) {
                    finish_fills();
                }
                const int32_t slot = g.free_slots.back();
                g.free_slots.pop_back();
                start_fill(x.il, x.v, slot);
                bytes += g.slot_bytes;
            }
        }
        finish_fills();
        for (layer & l : layers) {
            if (l.ig >= 0 && l.dirty) {
                upload_table(l);
            }
        }
        LLAMA_LOG_INFO("q4e: profile %s: filled the VRAM tier with %.2f GiB in %.1f s\n", params.profile_path.c_str(),
            bytes/1073741824.0, q4e_ms(t0, q4e_clock::now())/1000.0);
    }

    void save_profile() const {
        if (params.profile_path.empty()) {
            return;
        }
        FILE * fp = ggml_fopen(params.profile_path.c_str(), "wb");
        if (!fp) {
            return;
        }
        const uint32_t magic = Q4E_PROF_MAGIC;
        const uint32_t n_layer = (uint32_t) layers.size();
        fwrite(&magic, 4, 1, fp);
        fwrite(&n_layer, 4, 1, fp);
        for (const layer & l : layers) {
            std::vector<float> s(l.score.size());
            for (size_t e = 0; e < s.size(); ++e) {
                s[e] = score_now(l, (int64_t) e);
            }
            const uint32_t n = (uint32_t) s.size();
            fwrite(&n, 4, 1, fp);
            fwrite(s.data(), sizeof(float), n, fp);
        }
        fclose(fp);
    }

    // the iGPU tier: a copy of every block in the iGPU's memory, the graphs compute the blocks not in VRAM there.
    // allocated first: once the pack and the profile have touched gigabytes of mapped pages, the system grants the
    // iGPU less memory
    void alloc_tier2() {
        size_t n_tensors = 0;
        for (const group & g : groups) {
            n_tensors += g.layers.size()*N_PARTS;
        }
        ggml_init_params ip = { n_tensors*ggml_tensor_overhead(), nullptr, true };
        ctx_t2.reset(ggml_init(ip));
        for (const group & g : groups) {
            const int64_t ne[N_PARTS][2] = { { g.n_embd, g.f }, { g.n_embd, g.f }, { g.f, g.n_embd } };
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                for (int p = 0; p < N_PARTS; ++p) {
                    l.t2[p] = ggml_new_tensor_3d(ctx_t2.get(), g.type[p], ne[p][0], ne[p][1], l.n_virt);
                    ggml_format_name(l.t2[p], "q4e.t2.%d.%d", il, p);
                }
            }
        }
        size_t free = 0, total = 0, need = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(tier2), &free, &total);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_t2.get()); t; t = ggml_get_next_tensor(ctx_t2.get(), t)) {
            need += ggml_nbytes(t);
        }
        LLAMA_LOG_INFO("q4e: %s: %.2f GiB free of %.2f GiB, the iGPU tier needs %.2f GiB\n", ggml_backend_name(tier2),
            free/1073741824.0, total/1073741824.0, need/1073741824.0);
        buf_t2.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_t2.get(), ggml_backend_get_default_buffer_type(tier2)));
        if (!buf_t2) {
            throw std::runtime_error("q4e: cannot allocate the iGPU tier (put some expert layers on the dGPU with -ot)");
        }
        ggml_backend_buffer_set_usage(buf_t2.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // the copies, from the host views (gate/up rows of the model, down from the pack); the graphs read them
    void init_tier2() {
        const auto t0 = q4e_clock::now();
        size_t total = 0;
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                for (int p = 0; p < N_PARTS; ++p) {
                    GGML_ASSERT(ggml_nbytes(l.t2[p]) == g.blk_bytes[p]*l.n_virt);
                    ggml_backend_tensor_set(l.t2[p], l.host[p], 0, ggml_nbytes(l.t2[p]));
                    total += ggml_nbytes(l.t2[p]);
                }
                l.t.gate_host     = l.t2[P_GATE];
                l.t.up_host       = l.t2[P_UP];
                l.t.down_host     = l.t2[P_DOWN];
                l.t.backend_tier2 = tier2;
                std::fill(l.in_ram.begin(), l.in_ram.end(), (uint8_t) 1);
            }
        }
        size_t vram = 0;
        for (const group & g : groups) {
            vram += (size_t) g.n_slots*g.slot_bytes;
        }
        LLAMA_LOG_INFO("q4e: experts %.2f GiB: VRAM %.2f GiB | %s %.2f GiB (a copy of every block, uploaded in %.1f s), no CPU tier\n",
            total/1073741824.0, vram/1073741824.0, ggml_backend_name(tier2), total/1073741824.0, q4e_ms(t0, q4e_clock::now())/1000.0);
    }

    // the RAM tier: the blocks that are not in VRAM, best scored first, locked up to the budget
    void init_ram_tier() {
        size_t need = 0;
        for (const group & g : groups) {
            need += g.host_bytes - (size_t) g.n_slots*g.slot_bytes;
        }
        const size_t headroom = (size_t) 6*1073741824ull;
        const size_t free_ram = q4e_free_ram() + q4e_resident_ram();
        ram_budget = params.ram_mib >= 0 ? (size_t) params.ram_mib*1048576 : (free_ram > headroom ? free_ram - headroom : 0);
        ram_budget = std::min(ram_budget, need);
        locker.reserve(ram_budget + 256u*1048576);

        const auto t0 = q4e_clock::now();
        struct cand { float s; int32_t il; int32_t v; };
        std::vector<cand> c;
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                const layer & l = layers[il];
                for (int32_t v = 0; v < l.n_virt; ++v) {
                    if (l.slot_of[v] < 0) {
                        c.push_back({ l.score[v / g.n_blocks], il, v });
                    }
                }
            }
        }
        std::stable_sort(c.begin(), c.end(), [](const cand & a, const cand & b) { return a.s > b.s; });

        // prefetch a batch, then lock it: locking alone faults the pages in one by one
        size_t planned = 0;
        size_t n_plan  = 0;
        for (; n_plan < c.size(); ++n_plan) {
            const size_t bytes = groups[layers[c[n_plan].il].ig].slot_bytes;
            if (planned + bytes > ram_budget) {
                break;
            }
            planned += bytes;
        }
        constexpr size_t batch_bytes = 512u*1048576;
        for (size_t i0 = 0; i0 < n_plan && !locker.failed; ) {
            llama_memory_ranges batch;
            size_t bytes = 0;
            size_t i1 = i0;
            for (; i1 < n_plan && bytes < batch_bytes; ++i1) {
                const layer & l = layers[c[i1].il];
                const group & g = groups[l.ig];
                const uint8_t * ptr[N_PARTS];
                block_ptrs(l, g, c[i1].v, ptr);
                for (int p = 0; p < N_PARTS; ++p) {
                    batch.push_back({ ptr[p], g.blk_bytes[p] });
                }
                bytes += g.slot_bytes;
            }
            llama_prefetch(batch);
            for (size_t i = i0; i < i1; ++i) {
                if (!ram_lock(layers[c[i].il], groups[layers[c[i].il].ig], c[i].v)) {
                    break;
                }
            }
            i0 = i1;
        }

        size_t total = 0;
        for (const group & g : groups) {
            total += g.host_bytes;
        }
        size_t vram = 0;
        for (const group & g : groups) {
            vram += (size_t) g.n_slots*g.slot_bytes;
        }
        ram_used = 0;
        for (const layer & l : layers) {
            if (l.ig >= 0) {
                ram_used += (size_t) std::count(l.in_ram.begin(), l.in_ram.end(), (uint8_t) 1)*groups[l.ig].slot_bytes;
            }
        }
        const size_t ssd = total - std::min(total, vram + locker.locked);
        LLAMA_LOG_INFO("q4e: experts %.2f GiB: VRAM %.2f GiB | RAM %.2f GiB (locked in %.1f s, budget %.2f GiB, %.2f GiB free) | SSD %.2f GiB%s\n",
            total/1073741824.0, vram/1073741824.0, locker.locked/1073741824.0, q4e_ms(t0, q4e_clock::now())/1000.0,
            ram_budget/1073741824.0, free_ram/1073741824.0, ssd/1073741824.0,
            ssd < 1048576 ? " - every block is in VRAM or RAM" : " - VRAM + RAM do not hold every block");
    }

    const layer_tensors * get(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || lane_offset + n_tokens > params.max_batch) {
            return nullptr;
        }
        return lane == 1 ? &lane_t[il] : &layers[il].t;
    }

    // the scheduler of lane 1 for the timing of the next end_ubatch
    ggml_backend_sched_t timing_lane      = nullptr;
    ggml_backend_sched_t timing_lane_prev = nullptr;
    int64_t lane_prev_copy[Q4E_MAX_BACKENDS]    = {};
    int64_t lane_prev_compute[Q4E_MAX_BACKENDS] = {};

    // lane 1 of a pipelined pair: copies of the layer tensors whose selections go after those of lane 0
    int lane = 0;
    int64_t lane_offset = 0;
    std::vector<layer_tensors> lane_t;

    void set_lane(int l, int64_t offset) {
        lane        = l;
        lane_offset = l == 1 ? offset : 0;
        if (l != 1) {
            return;
        }
        if (lane_t.empty()) {
            lane_t.resize(layers.size());
        }
        const int64_t n_used = model.hparams.n_expert_used_max();
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            lane_t[il] = layers[il].t;
            lane_t[il].sel_host = layers[il].sel.data() + offset*n_used;
        }
    }

    void end_ubatch(ggml_backend_sched_t sched, int64_t n_tokens) {
        for (const device & d : devices) {
            ggml_backend_synchronize(d.backend);
        }
        const auto t0 = q4e_clock::now();
        const double ms_wall = q4e_ms(t_begin, t0);

        // the moves planned after an earlier ubatch: finished, or still running (blocks read from the SSD are slow),
        // then this ubatch plans none and does not wait
        const bool ready = !busy || worker_idle();
        if (ready) {
            finish_fills();
        }

        // count the selections, the routing ops wrote them on the host
        const int64_t n_used = model.hparams.n_expert_used_max();
        const int64_t n_sel  = n_used*n_tokens;

        struct cand { float s; int32_t il; int32_t v; };
        std::vector<std::vector<cand>> cands(groups.size());
        std::vector<blk_ref> ram_cands; // blocks read from the SSD
        counters c;
        for (size_t il = 0; il < layers.size(); ++il) {
            layer & l = layers[il];
            if (l.ig < 0 || !l.has_sel || n_tokens > params.max_batch) {
                continue;
            }
            const group & g = groups[l.ig];
            const int32_t * ids_l = l.sel.data();
            for (int64_t i = 0; i < n_sel; ++i) {
                const int32_t e = ids_l[i];
                if (e < 0 || e >= g.n_expert) {
                    continue;
                }
                touch(l, e);
                for (int32_t b = 0; b < g.n_blocks; ++b) {
                    const int32_t v = e*g.n_blocks + b;
                    if (l.table[v] >= 0.0f) {
                        c.blk_vram++;
                    } else {
                        if (l.in_ram[v]) {
                            c.blk_ram++;
                        } else {
                            c.blk_ssd++;
                        }
                        if (l.slot_of[v] < 0) {
                            cands[l.ig].push_back({ l.score[e], (int32_t) il, v });
                            if (!l.in_ram[v]) {
                                ram_cands.push_back({ l.score[e], (int32_t) il, v });
                            }
                        }
                    }
                }
            }
        }
        token += n_tokens;

        // move the best scored missing blocks into VRAM, in place of worse scored ones
        for (size_t ig = 0; ig < groups.size() && ready; ++ig) {
            group & g = groups[ig];
            auto & cg = cands[ig];
            if (cg.empty() || g.n_slots == 0) {
                continue;
            }
            // the blocks of one expert share its score, so duplicates are adjacent only when il and v break the ties
            std::sort(cg.begin(), cg.end(), [](const cand & a, const cand & b) {
                return a.s != b.s ? a.s > b.s : a.il != b.il ? a.il < b.il : a.v < b.v;
            });
            cg.erase(std::unique(cg.begin(), cg.end(), [](const cand & a, const cand & b) { return a.il == b.il && a.v == b.v; }), cg.end());

            // free slots fill as fast as the staging buffer allows, replacing blocks is limited so that the copies
            // end before the next ubatch does
            device & d = devices[g.dev];
            const size_t budget = g.free_slots.empty() ? std::min(d.staging_size, (size_t) Q4E_SWAP_BYTES) : d.staging_size;
            const size_t room   = budget > d.staging_used ? (budget - d.staging_used) / g.slot_bytes : 0;
            const size_t n_try  = std::min(cg.size(), room);

            // the worst scored blocks in VRAM, only needed when no slot is free
            std::vector<std::pair<float, int32_t>> victims;
            if (g.free_slots.size() < n_try) {
                victims.reserve(g.n_slots);
                for (int32_t s = 0; s < g.n_slots; ++s) {
                    const int64_t key = g.owner[s];
                    if (key < 0) {
                        continue;
                    }
                    const layer & lo = layers[key / layers[g.layers[0]].n_virt];
                    const int32_t v = (int32_t) (key % lo.n_virt);
                    if (lo.table[v] < 0.0f) {
                        continue; // in flight
                    }
                    victims.push_back({ score_now(lo, v / g.n_blocks), s });
                }
                const size_t k = std::min(victims.size(), n_try);
                std::partial_sort(victims.begin(), victims.begin() + k, victims.end());
                victims.resize(k);
            }
            size_t iv = 0;
            for (size_t i = 0; i < n_try; ++i) {
                int32_t slot = -1;
                if (!g.free_slots.empty()) {
                    slot = g.free_slots.back();
                    g.free_slots.pop_back();
                } else if (iv < victims.size() && cg[i].s > victims[iv].first*Q4E_HYSTERESIS + 0.5f) {
                    slot = victims[iv++].second;
                    evict(g, slot);
                } else {
                    break;
                }
                start_fill(cg[i].il, cg[i].v, slot);
                c.promoted++;
                c.promoted_bytes += g.slot_bytes;
            }
        }

        // the RAM tier follows too: the best scored blocks read from the SSD replace the worst scored RAM blocks
        if (ready && !ram_cands.empty()) {
            plan_ram(ram_cands);
        }

        upload_tables();

        // the worker reads the blocks and queues the copies while the next ubatch computes
        if (ready) {
            issue_fills();
        }

        // a killed process does not run the destructor: keep the profile on disk
        if (q4e_ms(t_saved, t0) > 60000.0) {
            save_profile();
            t_saved = t0;
        }

        const auto t1 = q4e_clock::now();
        c.ms_wall  = ms_wall;
        c.ms_promo = q4e_ms(t0, t1);
        c.tokens   = (uint64_t) n_tokens;
        c.ubatches = 1;

        // the scheduler timing, per backend
        const int n_backends = ggml_backend_sched_get_n_backends(sched);
        for (int i = 0; i < n_backends && i < Q4E_MAX_BACKENDS; ++i) {
            int64_t cu = 0;
            int64_t co = 0;
            int64_t ns = 0;
            if (ggml_backend_sched_get_timing(sched, i, &cu, &co, &ns)) {
                c.sched_copy_us[i]    = cu - sched_prev_copy[i];
                c.sched_compute_us[i] = co - sched_prev_compute[i];
                sched_prev_copy[i]    = cu;
                sched_prev_compute[i] = co;
            }
        }
        // lane 1 of a pipelined pair: its backends have the same order, their times add up
        if (timing_lane != nullptr) {
            if (timing_lane != timing_lane_prev) {
                timing_lane_prev = timing_lane;
                for (int i = 0; i < Q4E_MAX_BACKENDS; ++i) {
                    lane_prev_copy[i]    = 0;
                    lane_prev_compute[i] = 0;
                }
            }
            for (int i = 0; i < ggml_backend_sched_get_n_backends(timing_lane) && i < Q4E_MAX_BACKENDS; ++i) {
                int64_t cu = 0;
                int64_t co = 0;
                int64_t ns = 0;
                if (ggml_backend_sched_get_timing(timing_lane, i, &cu, &co, &ns)) {
                    c.sched_copy_us[i]    += cu - lane_prev_copy[i];
                    c.sched_compute_us[i] += co - lane_prev_compute[i];
                    lane_prev_copy[i]    = cu;
                    lane_prev_compute[i] = co;
                }
            }
            timing_lane = nullptr;
        }

        // the statistics are about generation, prompt ubatches only train the scores
        if (n_tokens <= 4) {
            add(all, c);
            add(window, c);
            if (window.tokens >= 64) {
                log_stats(window, "last 64", false, sched);
                window = counters();
            }
        }
    }

    static void add(counters & a, const counters & b) {
        a.tokens   += b.tokens;
        a.ubatches += b.ubatches;
        a.ms_wall  += b.ms_wall;
        a.ms_promo += b.ms_promo;
        a.blk_vram += b.blk_vram;
        a.blk_ram  += b.blk_ram;
        a.blk_ssd  += b.blk_ssd;
        a.promoted += b.promoted;
        a.promoted_bytes += b.promoted_bytes;
        for (int i = 0; i < Q4E_MAX_BACKENDS; ++i) {
            a.sched_copy_us[i]    += b.sched_copy_us[i];
            a.sched_compute_us[i] += b.sched_compute_us[i];
        }
    }

    void log_stats(const counters & c, const char * name, bool final, ggml_backend_sched_t sched = nullptr) const {
        if (c.tokens == 0) {
            return;
        }
        const double n   = (double) c.tokens;
        const double blk = (double) (c.blk_vram + c.blk_ram + c.blk_ssd);
        std::string timing;
        if (sched) {
            const int n_backends = ggml_backend_sched_get_n_backends(sched);
            for (int i = 0; i < n_backends && i < Q4E_MAX_BACKENDS; ++i) {
                if (c.sched_copy_us[i] == 0 && c.sched_compute_us[i] == 0) {
                    continue;
                }
                timing += format(" | %s compute %.2f copy-in %.2f", ggml_backend_name(ggml_backend_sched_get_backend(sched, i)),
                    c.sched_compute_us[i]/1000.0/n, c.sched_copy_us[i]/1000.0/n);
            }
        }
        LLAMA_LOG_INFO("q4e: %s: %llu tok, %.2f ms/tok (%.1f t/s): graph %.2f + tiers %.2f%s | blocks VRAM %.1f%% RAM %.1f%% SSD %.1f%% | promoted %llu blocks %.1f MiB%s\n",
            name, (unsigned long long) c.tokens, (c.ms_wall + c.ms_promo)/n, 1000.0*n/(c.ms_wall + c.ms_promo),
            c.ms_wall/n, c.ms_promo/n, timing.c_str(),
            100.0*c.blk_vram/std::max(blk, 1.0), 100.0*c.blk_ram/std::max(blk, 1.0), 100.0*c.blk_ssd/std::max(blk, 1.0),
            (unsigned long long) c.promoted, c.promoted_bytes/1048576.0, final ? " (final)" : "");
        if (final || window_logs++ % 4 == 0) {
            LLAMA_LOG_INFO("q4e: %llu tokens%s\n", (unsigned long long) token, working_set().c_str());
        }
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const {
        std::map<ggml_backend_buffer_type_t, size_t> res;
        for (const device & d : devices) {
            if (d.buf) {
                res[ggml_backend_buffer_get_type(d.buf.get())] += ggml_backend_buffer_get_size(d.buf.get());
            }
            if (d.staging_buf) {
                res[ggml_backend_buffer_get_type(d.staging_buf.get())] += ggml_backend_buffer_get_size(d.staging_buf.get());
            }
        }
        return res;
    }
};

llama_q4e::llama_q4e(const llama_model & model, ggml_backend_t backend_cpu, const std::vector<ggml_backend_t> & backends, const llama_q4e_params & params) :
    pimpl(new impl(model, backend_cpu, backends, params)) {
}

llama_q4e::~llama_q4e() = default;

const llama_q4e::layer_tensors * llama_q4e::get(int32_t il, int64_t n_tokens) const {
    return pimpl->get(il, n_tokens);
}

void llama_q4e::set_lane(int lane, int64_t offset) {
    pimpl->set_lane(lane, offset);
}

void llama_q4e::set_timing_lane(ggml_backend_sched_t sched) {
    pimpl->timing_lane = sched;
}

void llama_q4e::set_selection(int32_t il) const {
    pimpl->layers[il].has_sel = true;
}

void llama_q4e::new_graph() {
    for (auto & l : pimpl->layers) {
        l.has_sel = false;
    }
}

void llama_q4e::begin_ubatch() {
    pimpl->t_begin = q4e_clock::now();
}

void llama_q4e::end_ubatch(ggml_backend_sched_t sched, int64_t n_tokens) {
    pimpl->end_ubatch(sched, n_tokens);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_q4e::memory_breakdown() const {
    return pimpl->memory_breakdown();
}
