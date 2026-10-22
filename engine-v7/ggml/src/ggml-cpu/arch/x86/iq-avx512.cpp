// AVX-512 dot products of one weight row against 1-4 activation rows for the formats of 2-bit MoE experts.
// IQ2_S / IQ2_XXS follow Project Maya's iq_avx512.cpp (MIT, Peasant Smith and the Project Maya contributors): 64
// values are decoded once (8 grid lookups, one 64-bit sign mask, the scales) and every activation row applies them
// with a masked subtract, a maddubs and a madd; the integer sums are ggml's and the floats take ggml's AVX2 steps.
// The weight rows stream from RAM, so the blocks a few hundred bytes ahead are prefetched.

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ggml-impl.h"

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512BW__)

namespace {

constexpr int prefetch_ahead = 2048;
#ifndef Q4E_EXP
#define Q4E_EXP 3
#endif

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t * p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t * p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// ggml's hsum_float_8
inline float hsum8(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// 16 int32 lanes (values 4m..4m+3 of a 64-value chunk) -> ggml's 8 lanes (values 4k..4k+3 of each 32-value chunk)
inline __m256i fold(__m512i a) { return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1)); }

// int16 lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

struct fmt_iq2_xxs {
    static constexpr int bytes = sizeof(block_iq2_xxs);
    static inline void decode(const uint8_t * b, int j, __m512i & g, __mmask64 & m, __m512i & sc) {
        const uint8_t * q = b + 2 + 16*j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2*(int) (a1 >> 28) + 1, sb = 2*(int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

struct fmt_iq2_s {
    static constexpr int bytes = sizeof(block_iq2_s);
    static inline void decode(const uint8_t * b, int j, __m512i & g, __mmask64 & m, __m512i & sc) {
        const uint8_t * qs = b + 2 + 8*j;
        const uint8_t h0 = b[66 + 2*j], h1 = b[66 + 2*j + 1];
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8*j));
        const uint8_t s0 = b[74 + 2*j], s1 = b[74 + 2*j + 1];
        sc = scales4(2*(s0 & 15) + 1, 2*(s0 >> 4) + 1, 2*(s1 & 15) + 1, 2*(s1 >> 4) + 1);
    }
};

#if defined(__AVX512VBMI__)
// the same g, m and sc as the formats above (so the same sums), decoded for a whole super-block with vector ops: the
// 32 grid indices are computed in vector lanes and fetched with 4 gathers, the signs and the scales come from a few
// shuffles instead of ~40 scalar instructions per 64 values
struct fmt_iq2_s_v {
    static constexpr int bytes = sizeof(block_iq2_s);
    struct state {
        __m512i g[4];
        __m512i sc;
        const uint8_t * b;
    };
    static inline void prep(const uint8_t * b, state & s) {
        // index n: qs[n] | bits 2n..2n+1 of the 64-bit qh << 8
        const __m512i shift = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        const __m512i m300  = _mm512_set1_epi32(0x300);
        const uint64_t qh = u64(b + 66);
        const __m512i h0 = _mm512_srlv_epi32(_mm512_set1_epi32((int) (uint32_t) qh), shift);
        const __m512i h1 = _mm512_srlv_epi32(_mm512_set1_epi32((int) (uint32_t) (qh >> 32)), shift);
        const __m512i i0 = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (b + 2))),
                                           _mm512_and_si512(_mm512_slli_epi32(h0, 8), m300));
        const __m512i i1 = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (b + 18))),
                                           _mm512_and_si512(_mm512_slli_epi32(h1, 8), m300));
        const long long * grid = (const long long *) iq2s_grid;
#if Q4E_EXP == 3
        alignas(64) uint32_t ix[32]; alignas(64) uint64_t gb[32];
        _mm512_store_si512((void *) ix, i0); _mm512_store_si512((void *) (ix + 16), i1);
        for (int k = 0; k < 32; ++k) { gb[k] = iq2s_grid[ix[k]]; }
        for (int k = 0; k < 4; ++k) { s.g[k] = _mm512_load_si512((const void *) (gb + 8*k)); } (void) grid;
#elif Q4E_EXP == 2
        s.g[0] = _mm512_i32gather_epi32(i0, (const int *) grid, 4); s.g[1] = _mm512_i32gather_epi32(i1, (const int *) grid, 4); s.g[2] = _mm512_add_epi8(s.g[0], i1); s.g[3] = _mm512_sub_epi8(s.g[1], i0);
#elif Q4E_EXP == 1
        s.g[0] = i0; s.g[1] = i1; s.g[2] = _mm512_add_epi8(i0, i1); s.g[3] = _mm512_sub_epi8(i0, i1); (void) grid;
#else
        s.g[0] = _mm512_i32gather_epi64(_mm512_castsi512_si256(i0), grid, 8);
        s.g[1] = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(i0, 1), grid, 8);
        s.g[2] = _mm512_i32gather_epi64(_mm512_castsi512_si256(i1), grid, 8);
        s.g[3] = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(i1, 1), grid, 8);
#endif
        // 16 scales 2*nibble + 1 as int16, sub-block m in lane m
        const __m128i x  = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *) (b + 74)));
        const __m128i lo = _mm_and_si128(x, _mm_set1_epi16(15));
        const __m128i hi = _mm_srli_epi16(x, 4);
        const __m256i sc = _mm256_set_m128i(_mm_unpackhi_epi16(lo, hi), _mm_unpacklo_epi16(lo, hi));
        s.sc = _mm512_castsi256_si512(_mm256_add_epi16(_mm256_add_epi16(sc, sc), _mm256_set1_epi16(1)));
        s.b  = b;
    }
    static inline void decode(const state & s, int j, __m512i & g, __mmask64 & m, __m512i & sc) {
        // int16 lane l -> scale 4j + l/8
        const __m512i idx = _mm512_add_epi16(_mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                                              0, 0, 0, 0, 0, 0, 0, 0), _mm512_set1_epi16((short) (4*j)));
        g  = s.g[j];
        m  = _cvtu64_mask64(u64(s.b + 2 + 32 + 8*j));
        sc = _mm512_permutexvar_epi16(idx, s.sc);
    }
};

struct fmt_iq2_xxs_v {
    static constexpr int bytes = sizeof(block_iq2_xxs);
    struct state {
        __m512i g[4];
        __m512i sw;      // the 32-bit words of the block: a1 (word 4j+1) and b1 (word 4j+3) hold the signs and scales
        uint64_t sgn[4]; // the 64 signs of chunk j
    };
    static inline void prep(const uint8_t * b, state & s) {
        const __m512i q = _mm512_loadu_si512((const void *) (b + 2));
        // grid index bytes of chunk j: 16j + 0..3 and 16j + 8..11 -> bytes 8j..8j+7
        const __m512i pidx = _mm512_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            59, 58, 57, 56, 51, 50, 49, 48, 43, 42, 41, 40, 35, 34, 33, 32, 27, 26, 25, 24, 19, 18, 17, 16, 11, 10, 9, 8, 3, 2, 1, 0);
        const __m512i ib = _mm512_permutexvar_epi8(pidx, q);
        const __m512i i0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(ib));
        const __m512i i1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(ib, 1));
        const long long * grid = (const long long *) iq2xxs_grid;
        s.g[0] = _mm512_i32gather_epi64(_mm512_castsi512_si256(i0), grid, 8);
        s.g[1] = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(i0, 1), grid, 8);
        s.g[2] = _mm512_i32gather_epi64(_mm512_castsi512_si256(i1), grid, 8);
        s.g[3] = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(i1, 1), grid, 8);
        // qword j = a1 | b1 << 32 (bytes 16j + 4..7, 16j + 12..15), then the 7-bit sign fields at 0, 7, 14, 21 of each
        const __m512i psgn = _mm512_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            63, 62, 61, 60, 55, 54, 53, 52, 47, 46, 45, 44, 39, 38, 37, 36, 31, 30, 29, 28, 23, 22, 21, 20, 15, 14, 13, 12, 7, 6, 5, 4);
        const __m512i w   = _mm512_permutexvar_epi8(psgn, q);
        const __m512i ctl = _mm512_set1_epi64(0x352e2720150e0700LL); // 0, 7, 14, 21, 32, 39, 46, 53
        const __m512i m7f = _mm512_set1_epi8(0x7f);
        const __m512i m0f = _mm512_set1_epi8(0x0f);
        __m512i x = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, w), m7f);
        // ksigns_iq2xs: bit 7 = parity of the 7 bits
        const __m512i ptab = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0));
        const __m512i par  = _mm512_xor_si512(_mm512_shuffle_epi8(ptab, _mm512_and_si512(x, m0f)),
                                              _mm512_shuffle_epi8(ptab, _mm512_and_si512(_mm512_srli_epi16(x, 4), m0f)));
        x = _mm512_or_si512(x, _mm512_and_si512(_mm512_slli_epi16(par, 7), _mm512_set1_epi8((char) 0x80)));
        _mm256_storeu_si256((__m256i *) s.sgn, _mm512_castsi512_si256(x));
        // scales 2*(word >> 28) + 1 in the low half of each word
        const __m512i sw = _mm512_srli_epi32(q, 28);
        s.sw = _mm512_add_epi32(_mm512_add_epi32(sw, sw), _mm512_set1_epi32(1));
    }
    static inline void decode(const state & s, int j, __m512i & g, __mmask64 & m, __m512i & sc) {
        // int16 lanes 0-15 <- low half of word 4j+1, lanes 16-31 <- word 4j+3
        const __m512i idx = _mm512_add_epi16(_mm512_set_epi16(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                                              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                                             _mm512_set1_epi16((short) (8*j)));
        g  = s.g[j];
        m  = _cvtu64_mask64(s.sgn[j]);
        sc = _mm512_permutexvar_epi16(idx, s.sw);
    }
};
#endif

// the formats above decode one 64-value chunk at a time (no state)
template <typename F>
struct fmt_chunked {
    static constexpr int bytes = F::bytes;
    struct state {
        const uint8_t * b;
    };
    static inline void prep(const uint8_t * b, state & s) { s.b = b; }
    static inline void decode(const state & s, int j, __m512i & g, __mmask64 & m, __m512i & sc) { F::decode(s.b, j, g, m, sc); }
};

template <typename F, int NT>
inline void row_dot_iq2(const uint8_t * row, int nb, const block_q8_K * const * y, float * res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm256_setzero_ps();
    }
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nb; ++i) {
        const uint8_t * blk = row + (size_t) i*F::bytes;
        _mm_prefetch((const char *) blk + prefetch_ahead, _MM_HINT_T0);
        _mm_prefetch((const char *) blk + prefetch_ahead + 64, _MM_HINT_T0);
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) {
            acci[t] = _mm512_setzero_si512();
        }
        typename F::state st;
        F::prep(blk, st);
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            F::decode(st, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void *) (y[t][i].qs + 64*j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx*y[t][i].d), _mm256_cvtepi32_ps(fold(acci[t])), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        res[t] = 0.125f*hsum8(accf[t]);
    }
}

// Q2_0: 64 two-bit codes c, value c - 1, against two Q8_0 blocks: sum((c - 1)*y) = sum(c*y) - sum(y) per 32 values
template <int NT>
inline void row_dot_q2_0(const uint8_t * row, int nb, const block_q8_0 * const * y, float * res) {
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm512_setzero_ps();
    }
    const __m512i ones8  = _mm512_set1_epi8(1);
    const __m512i ones16 = _mm512_set1_epi16(1);
    const __m128i m3     = _mm_set1_epi8(3);
    for (int i = 0; i < nb; ++i) {
        const block_q2_0 * x = (const block_q2_0 *) row + i;
        _mm_prefetch((const char *) x + prefetch_ahead, _MM_HINT_T0);
        // value k is in byte k/4, bits 2*(k%4): 64 codes in value order
        const __m128i b  = _mm_loadu_si128((const __m128i *) x->qs);
        const __m128i c0 = _mm_and_si128(b, m3);
        const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
        const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
        const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
        const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);
        const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);
        const __m256i lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));
        const __m256i hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));
        const __m512i c  = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const float dx = GGML_FP16_TO_FP32(x->d);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0 * yb = y[t] + 2*i;
            const __m512i yv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) yb[0].qs)),
                                                  _mm256_loadu_si256((const __m256i *) yb[1].qs), 1);
            const __m512i p  = _mm512_sub_epi16(_mm512_maddubs_epi16(c, yv), _mm512_maddubs_epi16(ones8, yv));
            const __m512  f  = _mm512_cvtepi32_ps(_mm512_madd_epi16(p, ones16));
            const __m512  d  = _mm512_insertf32x8(_mm512_set1_ps(dx*GGML_FP16_TO_FP32(yb[0].d)),
                                                  _mm256_set1_ps(dx*GGML_FP16_TO_FP32(yb[1].d)), 1);
            acc[t] = _mm512_fmadd_ps(d, f, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        res[t] = _mm512_reduce_add_ps(acc[t]);
    }
}

#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
// Q2_0 rows against prepared Q8_0 activations, the float steps of ggml's AVX2 kernels: per Q2_0 block two FMAs of 8
// lanes (the scale of each Q8_0 block), hsum_float_8 at the end. two_acc: even and odd blocks in two accumulators
// added before the sum (ggml_vec_dot_q2_0_q8_0), else one (ggml_vec_dot_q2_0_q8_0_nc). The integer lanes are the
// same sums of 4 values (c - 1)*y: c*y by vpdpbusd onto minus the sum of the 4 y.
template <int NT>
inline void rows_q2_0(int nb, const uint8_t * vx, size_t bx, int nr, const int8_t * const * yq, const float * const * yd,
                      const int32_t * const * ys, bool two_acc, float * const * s) {
    // byte i of qword q <- bits 2*(8*(q%4) + i) of qword q/4 of the 16 code bytes
    const __m512i ctl  = _mm512_set_epi64(0x3e3c3a3836343230LL, 0x2e2c2a2826242220LL, 0x1e1c1a1816141210LL, 0x0e0c0a0806040200LL,
                                          0x3e3c3a3836343230LL, 0x2e2c2a2826242220LL, 0x1e1c1a1816141210LL, 0x0e0c0a0806040200LL);
    const __m512i pidx = _mm512_set_epi64(1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i m3   = _mm512_set1_epi8(3);
    for (int r = 0; r < nr; ++r) {
        const block_q2_0 * x = (const block_q2_0 *) (vx + (size_t) r*bx);
        __m256 acc0[NT], acc1[NT];
        for (int t = 0; t < NT; ++t) {
            acc0[t] = _mm256_setzero_ps();
            acc1[t] = _mm256_setzero_ps();
        }
        for (int i = 0; i < nb; ++i) {
            _mm_prefetch((const char *) (x + i) + prefetch_ahead, _MM_HINT_T0);
            const __m512i q = _mm512_permutexvar_epi64(pidx, _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) x[i].qs)));
            const __m512i c = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, q), m3);
            const float dx = GGML_FP16_TO_FP32(x[i].d);
            const bool odd = two_acc && (i & 1);
            for (int t = 0; t < NT; ++t) {
                const __m512i p = _mm512_dpbusd_epi32(_mm512_loadu_si512((const void *) (ys[t] + 16*i)), c,
                                                      _mm512_loadu_si512((const void *) (yq[t] + 64*i)));
                const __m512 f = _mm512_cvtepi32_ps(p);
                __m256 a = odd ? acc1[t] : acc0[t];
                a = _mm256_fmadd_ps(_mm256_set1_ps(dx*yd[t][2*i + 0]), _mm512_castps512_ps256(f), a);
                a = _mm256_fmadd_ps(_mm256_set1_ps(dx*yd[t][2*i + 1]), _mm512_extractf32x8_ps(f, 1), a);
                if (odd) {
                    acc1[t] = a;
                } else {
                    acc0[t] = a;
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            s[t][r] = hsum8(two_acc ? _mm256_add_ps(acc0[t], acc1[t]) : acc0[t]);
        }
    }
}
#endif

template <template <int> class K, typename Y>
inline void dispatch(int nc, const uint8_t * row, int nb, const void * const * vy, float * s) {
    const Y * y[4];
    for (int c = 0; c < nc; ++c) {
        y[c] = (const Y *) vy[c];
    }
    switch (nc) {
        case 1: K<1>::run(row, nb, y, s); break;
        case 2: K<2>::run(row, nb, y, s); break;
        case 3: K<3>::run(row, nb, y, s); break;
        default: K<4>::run(row, nb, y, s); break;
    }
}

#if defined(__AVX512VBMI__)
using fmt_iq2_s_best   = fmt_iq2_s_v;
using fmt_iq2_xxs_best = fmt_iq2_xxs_v;
#else
using fmt_iq2_s_best   = fmt_chunked<fmt_iq2_s>;
using fmt_iq2_xxs_best = fmt_chunked<fmt_iq2_xxs>;
#endif

template <int NT> struct k_iq2_s   { static void run(const uint8_t * r, int nb, const block_q8_K * const * y, float * s) { row_dot_iq2<fmt_iq2_s_best,   NT>(r, nb, y, s); } };
template <int NT> struct k_iq2_xxs { static void run(const uint8_t * r, int nb, const block_q8_K * const * y, float * s) { row_dot_iq2<fmt_iq2_xxs_best, NT>(r, nb, y, s); } };
template <int NT> struct k_iq2_s_ref   { static void run(const uint8_t * r, int nb, const block_q8_K * const * y, float * s) { row_dot_iq2<fmt_chunked<fmt_iq2_s>,   NT>(r, nb, y, s); } };
template <int NT> struct k_iq2_xxs_ref { static void run(const uint8_t * r, int nb, const block_q8_K * const * y, float * s) { row_dot_iq2<fmt_chunked<fmt_iq2_xxs>, NT>(r, nb, y, s); } };
template <int NT> struct k_q2_0    { static void run(const uint8_t * r, int nb, const block_q8_0 * const * y, float * s) { row_dot_q2_0<NT>(r, nb, y, s); } };

} // namespace

extern "C" {

void ggml_vec_dot_iq2_s_q8_K_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_iq2_s, block_q8_K>(nc, (const uint8_t *) vx, n / QK_K, vy, s);
}

void ggml_vec_dot_iq2_xxs_q8_K_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_iq2_xxs, block_q8_K>(nc, (const uint8_t *) vx, n / QK_K, vy, s);
}

// the scalar-decode versions, kept for q4e-kbench (same results, slower)
void ggml_vec_dot_iq2_s_q8_K_avx512_nc_ref(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_iq2_s_ref, block_q8_K>(nc, (const uint8_t *) vx, n / QK_K, vy, s);
}

void ggml_vec_dot_iq2_xxs_q8_K_avx512_nc_ref(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_iq2_xxs_ref, block_q8_K>(nc, (const uint8_t *) vx, n / QK_K, vy, s);
}

void ggml_vec_dot_q2_0_q8_0_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_q2_0, block_q8_0>(nc, (const uint8_t *) vx, n / QK2_0, vy, s);
}

bool ggml_q4e_dot_q2_0_rows(int n, const void * vx, size_t bx, int nr, const int8_t * const * yq, const float * const * yd,
                            const int32_t * const * ys, int nc, bool two_acc, float * const * s) {
#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
    const int nb = n / QK2_0;
    const uint8_t * x = (const uint8_t *) vx;
    switch (nc) {
        case 1:  rows_q2_0<1>(nb, x, bx, nr, yq, yd, ys, two_acc, s); break;
        case 2:  rows_q2_0<2>(nb, x, bx, nr, yq, yd, ys, two_acc, s); break;
        case 3:  rows_q2_0<3>(nb, x, bx, nr, yq, yd, ys, two_acc, s); break;
        default: rows_q2_0<4>(nb, x, bx, nr, yq, yd, ys, two_acc, s); break;
    }
    return true;
#else
    GGML_UNUSED(n); GGML_UNUSED(vx); GGML_UNUSED(bx); GGML_UNUSED(nr); GGML_UNUSED(yq); GGML_UNUSED(yd);
    GGML_UNUSED(ys); GGML_UNUSED(nc); GGML_UNUSED(two_acc); GGML_UNUSED(s);
    return false;
#endif
}

}

#endif // __AVX512F__ && __AVX512BW__
