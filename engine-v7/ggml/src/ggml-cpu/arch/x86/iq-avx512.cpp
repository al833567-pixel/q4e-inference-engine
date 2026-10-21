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
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            F::decode(blk, j, g, m, sc);
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

template <int NT> struct k_iq2_s   { static void run(const uint8_t * r, int nb, const block_q8_K * const * y, float * s) { row_dot_iq2<fmt_iq2_s,   NT>(r, nb, y, s); } };
template <int NT> struct k_iq2_xxs { static void run(const uint8_t * r, int nb, const block_q8_K * const * y, float * s) { row_dot_iq2<fmt_iq2_xxs, NT>(r, nb, y, s); } };
template <int NT> struct k_q2_0    { static void run(const uint8_t * r, int nb, const block_q8_0 * const * y, float * s) { row_dot_q2_0<NT>(r, nb, y, s); } };

} // namespace

extern "C" {

void ggml_vec_dot_iq2_s_q8_K_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_iq2_s, block_q8_K>(nc, (const uint8_t *) vx, n / QK_K, vy, s);
}

void ggml_vec_dot_iq2_xxs_q8_K_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_iq2_xxs, block_q8_K>(nc, (const uint8_t *) vx, n / QK_K, vy, s);
}

void ggml_vec_dot_q2_0_q8_0_avx512_nc(int n, float * s, const void * vx, const void * const * vy, int nc) {
    dispatch<k_q2_0, block_q8_0>(nc, (const uint8_t *) vx, n / QK2_0, vy, s);
}

}

#endif // __AVX512F__ && __AVX512BW__
