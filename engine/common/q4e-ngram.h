#pragma once
#include <cstdint>
#include <string>
#include <vector>

// PLE ngram table (per_layer_token_embd) contract.
// Layout: IQ4_NL blocks, d = f16 LE (2 B) + qs[16], byte j low nibble = elt 2j, high = elt 2j+1.
// x[i] = (q[i] - 8) * d. Row = 5 blocks x 18 B = 90 B (160 vals).
// A-side row ids: FNV-1a64(head salt, ids) % n_entries * 16 + head (deterministic oracle).

#define Q4E_NGRAM_TENSOR  "per_layer_token_embd.weight"
#define Q4E_NGRAM_BLOCK_ELEMS 32
#define Q4E_NGRAM_BLOCK_BYTES 18
#define Q4E_NGRAM_ROWS_PER_ENTRY 16

struct q4e_ngram_tensor_info {
    std::string file;
    std::string name;
    std::vector<int64_t> dims;
    int32_t gtype;
    int64_t data_start; // absolute file position where the data section begins (aligned)
    int64_t offset;     // tensor offset relative to data_start
    int64_t size;
};

// decode one 18-byte block (32 values) into out32; throws std::invalid_argument if len < 18
void q4e_ngram_decode_block_45(const uint8_t * block, size_t len, float * out32);

// FNV-1a 64 (offset 0xCBF29CE484222325, prime 0x100000001B3)
uint64_t q4e_ngram_fnv1a64(const uint8_t * data, size_t len);

// read GGUF tensor index (header only); path is a file or a dir of .gguf shards
std::vector<q4e_ngram_tensor_info> q4e_ngram_read_tensor_index(const std::string & path);

// RNG matching Python random.Random(int) (MT19937, version-2 int seeding) for replay parity
struct q4e_ngram_rng {
    uint32_t mt[624];
    int mti;
    void seed(uint64_t seed);
    uint32_t genrand();
    uint64_t getrandbits(int k);
    int64_t randbelow(int64_t n);
};

struct q4e_ngram_bench_result {
    int64_t n;
    int64_t bytes_per_read;
    double mean_ms;
    double p50_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
    double throughput_mbps;
    std::string mode;
};

class q4e_ngram_table {
public:
    // find Q4E_NGRAM_TENSOR in path (file or dir), validate 4.5-bit layout
    // throws std::runtime_error if not found, std::invalid_argument on layout mismatch
    void parse(const std::string & path);

    const std::string & path() const;
    int64_t data_offset() const;
    int64_t row_len() const;
    int64_t n_rows() const;
    int32_t gtype() const;
    int64_t size() const;
    int64_t n_entries() const;
    int64_t row_bytes() const;

    // 16 row ids: head 0-7 bigram (t_prev1, t), head 8-15 trigram (t_prev2, t_prev1, t)
    std::vector<int64_t> row_ids_for_token(int64_t t, int64_t t_prev1, int64_t t_prev2) const;

    // read + decode one row (160 vals); throws std::out_of_range
    std::vector<float> read_row(int64_t row_id);

    // last 3 tokens -> 16 x 160 vals; throws std::invalid_argument if empty
    std::vector<float> lookup(const std::vector<int64_t> & token_ids);

    // fake override points (B: real host/SSD offload + async prefetch)
    void offload(const std::string & path);
    const std::string & offload_path() const;
    void prefetch(const std::vector<int64_t> & token_ids);

    q4e_ngram_bench_result bench_random_reads(int64_t n, int64_t seed, int64_t warmup = 10) const;
    // replayable: same seed -> same row id sequence (parity with python read_sequence)
    std::vector<int64_t> read_sequence(int64_t n, int64_t seed, int64_t warmup = 10) const;

    void close();
    ~q4e_ngram_table();

private:
    void ensure_mapped() const;

    std::string file_path;
    int64_t data_off = 0;
    int64_t row_len_v = 0;
    int64_t rows_total = 0;
    int32_t tensor_gtype = -1;
    int64_t tensor_size = 0;
    int64_t entries_total = 0;
    std::string offload_target;

    mutable void * map_handle = nullptr;
    mutable void * file_handle = nullptr;
    mutable const uint8_t * view = nullptr;
};
