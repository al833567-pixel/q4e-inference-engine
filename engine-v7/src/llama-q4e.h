#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

struct llama_q4e_params {
    int32_t     blocks      = 0;  // sub-expert blocks per expert, 0 = auto
    int32_t     vram_mib    = -1; // VRAM tier, -1 = the free VRAM less reserve_mib
    int32_t     ram_mib     = -1; // RAM tier, -1 = the free RAM less 4 GiB
    int32_t     reserve_mib = 640;
    int32_t     max_batch   = 32; // ubatches with more tokens use the plain path
    std::string pack_path;        // file with the repacked down blocks (blocks > 1)
    std::string profile_path;     // block scores to start from and to save, empty = none
};

// q4e: three tiers for the routed experts of the MoE layers kept in host memory
//
//   VRAM: a pool of expert blocks, computed by the GPU
//   RAM:  blocks not in the pool, locked in memory, computed by the CPU in the same ubatch
//   SSD:  blocks beyond the RAM budget, the CPU reads them through the file mapping
//
// a block is a slice of the FFN rows of an expert (gate/up rows and the matching down columns), so it is a
// small expert of its own: the output of an expert is the sum of the outputs of its blocks, and the GPU and
// the CPU each compute only their blocks. blocks that are used often move to VRAM between ubatches; the copies
// run on a second stream while the next ubatch computes
class llama_q4e {
public:
    llama_q4e(const llama_model & model, ggml_backend_t backend_cpu, const std::vector<ggml_backend_t> & backends, const llama_q4e_params & params);
    ~llama_q4e();

    struct layer_tensors {
        int32_t n_blocks  = 0;
        int32_t zero_slot = 0; // first of n_expert_used*n_blocks slots of zeros, row j of a token reads zero_slot + j for a block not in VRAM

        ggml_tensor * gate_pool = nullptr; // [n_embd, n_ff/n_blocks, n_slots]
        ggml_tensor * up_pool   = nullptr;
        ggml_tensor * down_pool = nullptr; // [n_ff/n_blocks, n_embd, n_slots]
        ggml_tensor * gate_host = nullptr; // [n_embd, n_ff/n_blocks, n_expert*n_blocks]
        ggml_tensor * up_host   = nullptr;
        ggml_tensor * down_host = nullptr; // [n_ff/n_blocks, n_embd, n_expert*n_blocks]
        ggml_tensor * table     = nullptr; // F32 [1, n_expert*n_blocks]: the slot of each block, -1 if not in VRAM
        ggml_tensor * sel_all   = nullptr; // I32: layer il writes its [n_used, n_tokens] selections at il*n_used*n_tokens

        // host side, read by the routing op on the CPU while the ubatch computes; they change only between ubatches
        int32_t       n_expert   = 0;
        const float * table_host = nullptr; // [n_expert*n_blocks]: the slot of each block, -1 if not in VRAM
        int32_t *     sel_host   = nullptr; // [n_used*max_batch]: the routing op writes the selections here

        ggml_backend_t backend_gpu = nullptr;
        // the iGPU that computes the *_host blocks (their copies in its memory), nullptr: the CPU
        ggml_backend_t backend_tier2 = nullptr;
    };

    // the tensors of layer il if the tiers compute it for ubatches of n_tokens, nullptr otherwise
    const layer_tensors * get(int32_t il, int64_t n_tokens) const;

    // the graphs built next are for lane 1 of a pipelined pair, whose tokens follow the offset tokens of lane 0:
    // its routing ops write the selections after those of lane 0 (lane 0: offset 0)
    void set_lane(int lane, int64_t offset);

    // the next end_ubatch also counts the timing of this scheduler (lane 1 of a pipelined pair)
    void set_timing_lane(ggml_backend_sched_t sched);

    // the current graph copies the selected experts of layer il to sel_all, they are read after each ubatch
    // (on_device: a copy op writes them to sel_all at il*n_used*n_tokens, else the routing op to sel_host)
    void set_selection(int32_t il, bool on_device = false) const;

    // a new graph is about to be built: the selections of the old one are gone
    void new_graph();

    void begin_ubatch();
    // after the ubatch: count the selections and move blocks between the tiers
    void end_ubatch(ggml_backend_sched_t sched, int64_t n_tokens);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_q4e_ptr = std::unique_ptr<llama_q4e>;
