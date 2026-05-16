/**
 * @file SIMD_sincos_ps.h
 * @brief Vectorized sin, cos, sincos for AVX2 float (ps).
 *
 * All functions operate on 8-wide float vectors (__m256).
 * Polynomials are evaluated via Estrin's scheme (SIMD_poly_estrin.h).
 * Coefficients are from minimax_coeffs.h.
 *
 * @par Accuracy
 *   ~1 ULP for |x| <= ~10^5. Above that, Cody-Waite reduction loses
 *   precision; for huge-angle correctness use a Payne-Hanek reducer.
 *
 * @note Requires AVX2 + FMA (-mavx2 -mfma).
 */

#ifndef MC_SIMD_SINCOS_PS_H
#define MC_SIMD_SINCOS_PS_H

#include <immintrin.h>
#include "minimax_coeffs.h"
#include "SIMD_poly_estrin.h"

/* ============================================================
 * Range reduction constants — pi/2 split for Cody-Waite
 * ============================================================
 * PI/2 = PIO2_HI + PIO2_LO  (the two floats below sum to pi/2
 * with ~48 bits of effective precision).
 */
#define MC_TWO_OVER_PI  (0.636619772367581343f)
#define MC_PIO2_HI      ( 1.5707963705062866f)    /* round(pi/2) as float */
#define MC_PIO2_LO      (-4.3711388286737929e-8f) /* pi/2 - PIO2_HI       */

/* ============================================================
 * sincos(x)
 * ============================================================ */

/**
 * @brief Vectorized sine and cosine, computed together.
 *
 * @par Algorithm
 *   1. Reduce: n = round(x * 2/pi), r = x - n*(pi/2) via two-step FMA.
 *      After reduction, r is in [-pi/4, pi/4].
 *   2. Evaluate both polynomials at r^2:
 *        sin(r) ~ r * P_sin(r^2)
 *        cos(r) ~ 1 - (r^2/2) * P_cos(r^2)
 *   3. Quadrant select using low 2 bits of n. Let q = n mod 4:
 *        q=0:  sin =  sin_r,  cos =  cos_r
 *        q=1:  sin =  cos_r,  cos = -sin_r
 *        q=2:  sin = -sin_r,  cos = -cos_r
 *        q=3:  sin = -cos_r,  cos =  sin_r
 *      Implemented as: swap if (n&1), then XOR sign bit based on (n&2)
 *      for sin and ((n+1)&2) for cos.
 *
 * @param[in]  x      Input angles (radians).
 * @param[out] s_out  Pointer to receive sin(x) per lane.
 * @param[out] c_out  Pointer to receive cos(x) per lane.
 *
 * @note Both polynomials are evaluated regardless, since SIMD lanes may
 *       land in any quadrant. There is no perf win in calling mc_sin_ps
 *       vs. mc_sincos_ps when you need both.
 */
static inline void mc_sincos_ps(__m256 x, __m256* s_out, __m256* c_out) {
    /* --- 1. Range reduction ------------------------------------------- */
    const __m256 two_over_pi = _mm256_set1_ps(MC_TWO_OVER_PI);
    __m256 fn = _mm256_round_ps(
        _mm256_mul_ps(x, two_over_pi),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );

    const __m256 pio2_hi = _mm256_set1_ps(MC_PIO2_HI);
    const __m256 pio2_lo = _mm256_set1_ps(MC_PIO2_LO);
    __m256 r = _mm256_fnmadd_ps(fn, pio2_hi, x);   /* x - n*PIO2_HI */
    r        = _mm256_fnmadd_ps(fn, pio2_lo, r);   /* - n*PIO2_LO   */

    /* --- 2. Evaluate both polynomials --------------------------------- */
    __m256 r2 = _mm256_mul_ps(r, r);

    /* sin(r) = r * P_sin(r^2) */
    __m256 sp    = mc_poly_estrin_deg3_ps(r2, SIN_COEFFS);
    __m256 sin_r = _mm256_mul_ps(r, sp);

    /* cos(r) = 1 - (r^2/2) * P_cos(r^2) */
    __m256 cp     = mc_poly_estrin_deg3_ps(r2, COS_COEFFS);
    __m256 half_r2 = _mm256_mul_ps(r2, _mm256_set1_ps(0.5f));
    __m256 cos_r  = _mm256_fnmadd_ps(half_r2, cp, _mm256_set1_ps(1.0f));

    /* --- 3. Quadrant reconstruction ----------------------------------- */
    __m256i ni    = _mm256_cvtps_epi32(fn);
    __m256i v_one = _mm256_set1_epi32(1);
    __m256i v_two = _mm256_set1_epi32(2);

    /* Swap sin/cos lanes where (n & 1) */
    __m256 swap_mask = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(ni, v_one), v_one)
    );
    __m256 s_pre = _mm256_blendv_ps(sin_r, cos_r, swap_mask);
    __m256 c_pre = _mm256_blendv_ps(cos_r, sin_r, swap_mask);

    /* Sign flip via XOR. (n & 2) is 0 or 2; shift bit 1 -> bit 31. */
    __m256i sin_sign = _mm256_slli_epi32(_mm256_and_si256(ni, v_two), 30);
    __m256i cos_sign = _mm256_slli_epi32(
        _mm256_and_si256(_mm256_add_epi32(ni, v_one), v_two), 30
    );

    /* Unaligned stores: Windows x64 ABI doesn't guarantee 32-byte stack
     * alignment so an aligned *s_out = ... can fault from callers whose
     * __m256 locals land underaligned. vmovups is at most one cycle
     * slower and has no alignment requirement. */
    _mm256_storeu_ps((float*)s_out, _mm256_xor_ps(s_pre, _mm256_castsi256_ps(sin_sign)));
    _mm256_storeu_ps((float*)c_out, _mm256_xor_ps(c_pre, _mm256_castsi256_ps(cos_sign)));
}

/* ============================================================
 * sin(x), cos(x) — wrappers
 * ============================================================ */

/** @brief Vectorized sine. See mc_sincos_ps for algorithm. */
static inline __m256 mc_sin_ps(__m256 x) {
    __m256 s, c;
    mc_sincos_ps(x, &s, &c);
    return s;
}

/** @brief Vectorized cosine. See mc_sincos_ps for algorithm. */
static inline __m256 mc_cos_ps(__m256 x) {
    __m256 s, c;
    mc_sincos_ps(x, &s, &c);
    return c;
}

#endif /* MC_SIMD_SINCOS_PS_H */
