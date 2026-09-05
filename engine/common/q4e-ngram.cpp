#include "q4e-ngram.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

// observed sink for bench reads, so the compiler cannot eliminate them
static volatile uint64_t g_q4e_bench_sink;

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7C00u) >> 10;
    const uint32_t mant = h & 0x03FFu;
    if (exp == 0) {
        if (mant == 0) {
            return *(float *)&sign;
        }
        float v = (float)mant;
        v *= (float)ldexp(1.0f, -24);
        return sign ? -v : v;
    }
    uint32_t bits;
    if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return *(float *)&bits;
}

}

void q4e_ngram_decode_block_45(const uint8_t * block, size_t len, float * out32) {
    if (len < Q4E_NGRAM_BLOCK_BYTES) {
        throw std::invalid_argument("block too short");
    }
    const uint16_t d16 = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
    const float d = f16_to_f32(d16);
    for (int j = 0; j < 16; j++) {
        const uint8_t b = block[2 + j];
        out32[2 * j]     = ((float)(b & 0xFu) - 8.0f) * d;
        out32[2 * j + 1] = ((float)((b >> 4) & 0xFu) - 8.0f) * d;
    }
}

uint64_t q4e_ngram_fnv1a64(const uint8_t * data, size_t len) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x00000100000001B3ull;
    }
    return h;
}

std::vector<q4e_ngram_tensor_info> q4e_ngram_read_tensor_index(const std::string & path) {
    std::vector<std::string> files;
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        for (const auto & entry : std::filesystem::directory_iterator(path, ec)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            throw std::runtime_error("no .gguf in " + path);
        }
    } else {
        files.push_back(path);
    }

    std::vector<q4e_ngram_tensor_info> out;
    for (const auto & fn : files) {
        struct gguf_init_params params;
        params.no_alloc = true;
        params.ctx = nullptr;
        struct gguf_context * ctx = gguf_init_from_file(fn.c_str(), params);
        if (ctx == nullptr) {
            throw std::runtime_error(fn + ": not a GGUF file (bad magic)");
        }
        const int64_t nt = gguf_get_n_tensors(ctx);
        for (int64_t i = 0; i < nt; i++) {
            q4e_ngram_tensor_info e;
            e.file = fn;
            e.name = gguf_get_tensor_name(ctx, i);
            const int64_t * ne = gguf_get_tensor_ne(ctx, i);
            e.dims.assign(ne, ne + 4);
            e.gtype = (int32_t)gguf_get_tensor_type(ctx, i);
            e.data_start = (int64_t)gguf_get_data_offset(ctx);
            e.offset = (int64_t)gguf_get_tensor_offset(ctx, i);
            e.size = (int64_t)gguf_get_tensor_size(ctx, i);
            out.push_back(std::move(e));
        }
        gguf_free(ctx);
    }
    return out;
}

// ---------------------------------------------------------------------------
// RNG: MT19937, seeding and randbelow matching Python random.Random(int)
// ---------------------------------------------------------------------------

// CPython random.Random(int): init_by_array over the 32-bit LE chunks of the seed
void q4e_ngram_rng::seed(uint64_t s) {
    uint32_t key[8];
    int keyused = 0;
    if (s == 0) {
        key[0] = 0;
        keyused = 1;
    } else {
        while (s) {
            key[keyused++] = (uint32_t)(s & 0xFFFFFFFFu);
            s >>= 32;
        }
    }
    mt[0] = 19650218u;
    for (int i = 1; i < 624; i++) {
        mt[i] = (uint32_t)(1812433253ul * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i);
    }
    int i = 1;
    int j = 0;
    int k = 624 > keyused ? 624 : keyused;
    for (; k; k--) {
        mt[i] = (uint32_t)((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525ul)) + key[j] + (uint32_t)j);
        i++;
        j++;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
        if (j >= keyused) {
            j = 0;
        }
    }
    for (k = 623; k; k--) {
        mt[i] = (uint32_t)((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941ul)) - (uint32_t)i);
        i++;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
    }
    mt[0] = 0x80000000u;
    mti = 624;
}

uint32_t q4e_ngram_rng::genrand() {
    uint32_t y;
    if (mti >= 624) {
        for (int k = 0; k < 227; k++) {
            const uint32_t u = (mt[k] & 0x80000000ul) | (mt[k + 1] & 0x7FFFFFFFUL);
            mt[k] = mt[k + 397] ^ (u >> 1) ^ (u & 1 ? 0x9908B0DFul : 0u);
        }
        for (int k = 227; k < 623; k++) {
            const uint32_t u = (mt[k] & 0x80000000ul) | (mt[k + 1] & 0x7FFFFFFFUL);
            mt[k] = mt[k - 227] ^ (u >> 1) ^ (u & 1 ? 0x9908B0DFul : 0u);
        }
        {
            const uint32_t u = (mt[623] & 0x80000000ul) | (mt[0] & 0x7FFFFFFFUL);
            mt[623] = mt[396] ^ (u >> 1) ^ (u & 1 ? 0x9908B0DFul : 0u);
        }
        mti = 0;
    }
    y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680ul;
    y ^= (y << 15) & 0xEFC60000ul;
    y ^= (y >> 18);
    return y;
}

uint64_t q4e_ngram_rng::getrandbits(int k) {
    if (k <= 0) {
        return 0;
    }
    uint64_t result = 0;
    for (int i = (k + 31) / 32 - 1; i > 0; i--) {
        result = (result << 32) | genrand();
    }
    result = (result << 32) | genrand();
    if (k % 32 == 0) {
        return result;
    }
    return result >> (32 - (k % 32));
}

int64_t q4e_ngram_rng::randbelow(int64_t n) {
    if (n <= 0) {
        throw std::invalid_argument("n must be > 0");
    }
    int bits = 0;
    for (int64_t v = n; v; v >>= 1) {
        bits++;
    }
    int64_t r;
    do {
        r = (int64_t)getrandbits(bits);
    } while (r >= n);
    return r;
}

// ---------------------------------------------------------------------------
// q4e_ngram_table
// ---------------------------------------------------------------------------

q4e_ngram_table::~q4e_ngram_table() {
    close();
}

void q4e_ngram_table::close() {
#ifdef _WIN32
    if (view) {
        UnmapViewOfFile((void *)view);
        view = nullptr;
    }
    if (map_handle) {
        CloseHandle((HANDLE)map_handle);
        map_handle = nullptr;
    }
    if (file_handle) {
        CloseHandle((HANDLE)file_handle);
        file_handle = nullptr;
    }
#else
    view = nullptr;
#endif
}

void q4e_ngram_table::ensure_mapped() const {
    if (view) {
        return;
    }
#ifdef _WIN32
    file_handle = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("open failed: " + file_path);
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx((HANDLE)file_handle, &sz)) {
        throw std::runtime_error("GetFileSizeEx failed: " + file_path);
    }
    // PAGE_READONLY: handle opened with GENERIC_READ only
    map_handle = CreateFileMappingA((HANDLE)file_handle, nullptr, PAGE_READONLY,
        (DWORD)(sz.QuadPart >> 32), (DWORD)(sz.QuadPart & 0xFFFFFFFF), nullptr);
    if (map_handle == nullptr) {
        throw std::runtime_error("CreateFileMapping failed: " + file_path);
    }
    view = (const uint8_t *)MapViewOfFile((HANDLE)map_handle, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        throw std::runtime_error("MapViewOfFile failed: " + file_path);
    }
#else
    throw std::runtime_error("mmap not supported on this platform (A-4 is Windows-only)");
#endif
}

void q4e_ngram_table::parse(const std::string & path) {
    const auto entries = q4e_ngram_read_tensor_index(path);
    bool found = false;
    for (const auto & e : entries) {
        if (e.name != Q4E_NGRAM_TENSOR) {
            continue;
        }
        if (row_len_v != 0) {
            throw std::runtime_error(std::string("multiple ") + Q4E_NGRAM_TENSOR + " tensors found");
        }
        file_path = e.file;
        data_off = e.data_start + e.offset; // absolute file position (mmap covers the whole file)
        row_len_v = e.dims[0];
        rows_total = e.dims[1];
        tensor_gtype = e.gtype;
        tensor_size = e.size;
        found = true;
        break;
    }
    if (!found) {
        throw std::runtime_error(std::string(Q4E_NGRAM_TENSOR) + " not found under " + path);
    }
    if (row_len_v % Q4E_NGRAM_BLOCK_ELEMS) {
        throw std::invalid_argument("row_len not multiple of 32");
    }
    const int64_t expected = rows_total * (row_len_v / Q4E_NGRAM_BLOCK_ELEMS) * Q4E_NGRAM_BLOCK_BYTES;
    if (std::llabs(expected - tensor_size) > 2) {
        throw std::invalid_argument("tensor size != 4.5-bit expected");
    }
    if (rows_total % Q4E_NGRAM_ROWS_PER_ENTRY) {
        throw std::invalid_argument("n_rows not multiple of 16");
    }
    entries_total = rows_total / Q4E_NGRAM_ROWS_PER_ENTRY;
}

const std::string & q4e_ngram_table::path() const { return file_path; }
int64_t q4e_ngram_table::data_offset() const { return data_off; }
int64_t q4e_ngram_table::row_len() const { return row_len_v; }
int64_t q4e_ngram_table::n_rows() const { return rows_total; }
int32_t q4e_ngram_table::gtype() const { return tensor_gtype; }
int64_t q4e_ngram_table::size() const { return tensor_size; }
int64_t q4e_ngram_table::n_entries() const { return entries_total; }

int64_t q4e_ngram_table::row_bytes() const {
    if (row_len_v == 0) {
        throw std::runtime_error("call parse() first");
    }
    if (row_len_v % Q4E_NGRAM_BLOCK_ELEMS) {
        throw std::invalid_argument("row_len not multiple of 32");
    }
    return row_len_v / Q4E_NGRAM_BLOCK_ELEMS * Q4E_NGRAM_BLOCK_BYTES;
}

std::vector<int64_t> q4e_ngram_table::row_ids_for_token(int64_t t, int64_t t_prev1, int64_t t_prev2) const {
    if (entries_total == 0) {
        throw std::runtime_error("call parse() first");
    }
    auto hash_head = [this](int head, const int64_t * ids, int n_ids) {
        uint8_t raw[4 + 8 * 16];
        raw[0] = (uint8_t)head;
        raw[1] = (uint8_t)(head >> 8);
        raw[2] = (uint8_t)(head >> 16);
        raw[3] = (uint8_t)(head >> 24);
        for (int i = 0; i < n_ids; i++) {
            for (int b = 0; b < 8; b++) {
                raw[4 + i * 8 + b] = (uint8_t)((ids[i] >> (8 * b)) & 0xFF);
            }
        }
        const uint64_t h = q4e_ngram_fnv1a64(raw, 4 + (size_t)n_ids * 8);
        return (int64_t)(h % (uint64_t)entries_total);
    };

    std::vector<int64_t> rids;
    rids.reserve(16);
    for (int h = 0; h < 8; h++) {
        const int64_t ids[2] = { t_prev1, t };
        rids.push_back(hash_head(h, ids, 2) * Q4E_NGRAM_ROWS_PER_ENTRY + h);
    }
    for (int h = 8; h < 16; h++) {
        const int64_t ids[3] = { t_prev2, t_prev1, t };
        rids.push_back(hash_head(h, ids, 3) * Q4E_NGRAM_ROWS_PER_ENTRY + (h - 8));
    }
    return rids;
}

std::vector<float> q4e_ngram_table::read_row(int64_t row_id) {
    if (file_path.empty()) {
        throw std::runtime_error("call parse() first");
    }
    if (row_id < 0 || row_id >= rows_total) {
        throw std::out_of_range("row_id out of range");
    }
    ensure_mapped();
    const int64_t rb = row_bytes();
    const int64_t n_blk = row_len_v / Q4E_NGRAM_BLOCK_ELEMS;
    const uint8_t * p = view + data_off + row_id * rb;
    std::vector<float> out((size_t)row_len_v);
    for (int64_t b = 0; b < n_blk; b++) {
        q4e_ngram_decode_block_45(p + b * Q4E_NGRAM_BLOCK_BYTES, Q4E_NGRAM_BLOCK_BYTES, &out[(size_t)b * Q4E_NGRAM_BLOCK_ELEMS]);
    }
    return out;
}

std::vector<float> q4e_ngram_table::lookup(const std::vector<int64_t> & token_ids) {
    if (token_ids.empty()) {
        throw std::invalid_argument("token_ids is empty");
    }
    const int64_t n = (int64_t)token_ids.size();
    const int64_t t = token_ids[n - 1];
    const int64_t t1 = n >= 2 ? token_ids[n - 2] : 0;
    const int64_t t2 = n >= 3 ? token_ids[n - 3] : 0;
    const auto rids = row_ids_for_token(t, t1, t2);
    std::vector<float> out;
    out.reserve(16 * (size_t)row_len_v);
    for (int64_t rid : rids) {
        const auto r = read_row(rid);
        out.insert(out.end(), r.begin(), r.end());
    }
    return out;
}

void q4e_ngram_table::offload(const std::string & path) {
    offload_target = path;
}

const std::string & q4e_ngram_table::offload_path() const { return offload_target; }

void q4e_ngram_table::prefetch(const std::vector<int64_t> & token_ids) {
    if (token_ids.empty()) {
        return;
    }
    const int64_t n = (int64_t)token_ids.size();
    const int64_t t = token_ids[n - 1];
    const int64_t t1 = n >= 2 ? token_ids[n - 2] : 0;
    const int64_t t2 = n >= 3 ? token_ids[n - 3] : 0;
    for (int64_t rid : row_ids_for_token(t, t1, t2)) {
        read_row(rid);
    }
}

q4e_ngram_bench_result q4e_ngram_table::bench_random_reads(int64_t n, int64_t seed, int64_t warmup) const {
    if (file_path.empty()) {
        throw std::runtime_error("call parse() first");
    }
    if (n <= 0) {
        throw std::invalid_argument("n must be > 0");
    }
    ensure_mapped();
    q4e_ngram_rng rng;
    rng.seed((uint64_t)seed);
    const int64_t rb = row_bytes();

    auto touch = [this](int64_t rid, int64_t rb) {
        const uint8_t * p = view + data_off + rid * rb;
        const uint8_t * end = p + rb;
        uint64_t sum = 0;
        for (; p < end; p++) {
            sum += (uint64_t)*p;
        }
        g_q4e_bench_sink += sum;
    };

    for (int64_t i = 0; i < warmup; i++) {
        touch(rng.randbelow(rows_total), rb);
    }
    std::vector<double> lat;
    lat.reserve((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        const int64_t rid = rng.randbelow(rows_total);
        const auto t0 = std::chrono::steady_clock::now();
        touch(rid, rb);
        lat.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(lat.begin(), lat.end());
    double total = 0.0;
    for (double v : lat) {
        total += v;
    }
    const int64_t i99 = (n - 1) < (int64_t)(n * 0.99) ? (n - 1) : (int64_t)(n * 0.99);
    q4e_ngram_bench_result r;
    r.n = n;
    r.bytes_per_read = rb;
    r.mean_ms = std::round(total / (double)n * 1e4) / 1e4;
    r.p50_ms = std::round(lat[(size_t)(n / 2)] * 1e4) / 1e4;
    r.p99_ms = std::round(lat[(size_t)i99] * 1e4) / 1e4;
    r.min_ms = std::round(lat[0] * 1e4) / 1e4;
    r.max_ms = std::round(lat[(size_t)(n - 1)] * 1e4) / 1e4;
    r.throughput_mbps = total > 0 ? std::round((double)n * (double)rb / 1e6 / (total / 1000.0) * 100.0) / 100.0 : 0.0;
    r.mode = "mmap (page-cache random reads; B re-measures with real offload)";
    return r;
}

std::vector<int64_t> q4e_ngram_table::read_sequence(int64_t n, int64_t seed, int64_t warmup) const {
    if (file_path.empty()) {
        throw std::runtime_error("call parse() first");
    }
    if (n <= 0) {
        throw std::invalid_argument("n must be > 0");
    }
    q4e_ngram_rng rng;
    rng.seed((uint64_t)seed);
    for (int64_t i = 0; i < warmup; i++) {
        rng.randbelow(rows_total);
    }
    std::vector<int64_t> out;
    out.reserve((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        out.push_back(rng.randbelow(rows_total));
    }
    return out;
}
