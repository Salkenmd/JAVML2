/**
 * @file SIMD_asin_acos_ps.h
 * @brief Vectorized asin and acos for AVX2 float (ps).
 *
 * @par Accuracy
 *   ~1-2 ULP across [-1, 1]. Both endpoints land exactly on 0 or pi/2/pi
 *   thanks to the sqrt-based reduction near +/-1.
 *
 * @note Requires AVX2 + FMA. Uses SIMD_foundation_ps.h for mc_abs_ps.
 */

#ifndef MC_SIMD_ASIN_ACOS_PS_H
#define MC_SIMD_ASIN_ACOS_PS_H

#include <immintrin.h>
#include "minimax_coeffs.h"
#include "SIMD_poly_estrin.h"
#include "SIMD_foundation_ps.h"

/* ============================================================
 * asin(x)
 * ============================================================ */

/**
 * @brief Vectorized arcsine.
 *
 * @par Algorithm
 *   The polynomial P(t) approximates asin(s)/s on t = s^2 in [0, 0.25].
 *   To cover the full [-1, 1] domain, fold lanes with |x| > 0.5 via:
 *     asin(x) = pi/2 - 2 * asin( sqrt((1 - |x|) / 2) )
 *   The argument inside the inner asin is in [0, sqrt(0.5/2)] = [0, 0.5],
 *   so the same polynomial works on its square.
 *
 *   Per-lane (branchless):
 *     ax     = |x|
 *     big    = ax > 0.5
 *     t      = big ? (1 - ax)/2 : ax^2     (polynomial variable s^2)
 *     v      = big ? sqrt(t)    : ax       (s itself, the multiplicand)
 *     pre    = v * P(t)                    (= asin(s))
 *     result = big ? (pi/2 - 2*pre) : pre  (= asin(ax))
 *     return copysign(result, x)           (asin is odd)
 *
 * @par Cost
 *   One sqrt, one degree-5 polynomial, two blends, one XOR for sign.
 *
 * @param x  Input vector. Lanes with |x| > 1 produce NaN via sqrt of
 *           a negative number; clamp at the call site if needed.
 */
static inline __m256 mc_asin_ps(__m256 x) {
    const __m256 one      = _mm256_set1_ps(1.0f);
    const __m256 half     = _mm256_set1_ps(0.5f);
    const __m256 pio2     = _mm256_set1_ps(1.57079632679489661923f);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    __m256 sign_x = _mm256_and_ps(x, sign_bit);
    __m256 ax     = mc_abs_ps(x);
    /* Saturate to [0, 1] so |x|>1 yields +/-pi/2 instead of NaN via sqrt. */
    ax = _mm256_min_ps(ax, one);

    __m256 big = _mm256_cmp_ps(ax, half, _CMP_GT_OQ);

    __m256 t_small = _mm256_mul_ps(ax, ax);
    __m256 t_big   = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    __m256 t       = _mm256_blendv_ps(t_small, t_big, big);

    __m256 v = _mm256_blendv_ps(ax, _mm256_sqrt_ps(t_big), big);

    __m256 p   = mc_poly_estrin_deg5_ps(t, ASIN_COEFFS);
    __m256 pre = _mm256_mul_ps(v, p);

    __m256 res_big = _mm256_sub_ps(pio2, _mm256_add_ps(pre, pre));
    __m256 res     = _mm256_blendv_ps(pre, res_big, big);

    /* asin is odd: reapply sign of x. */
    return _mm256_xor_ps(res, sign_x);
}

/* ============================================================
 * acos(x)
 * ============================================================ */

/**
 * @brief Vectorized arccosine.
 *
 * @par Algorithm
 *   The naive identity acos(x) = pi/2 - asin(x) loses precision near
 *   x = +/-1 because asin(x) approaches +/-pi/2 and the subtraction
 *   cancels most of the bits. Instead, near +/-1 we use the same
 *   sqrt-based reduction as asin, applied directly:
 *     |x| in [0, 0.5]:  acos(x) = pi/2 - asin(x)
 *     x in (0.5, 1]:    acos(x) = 2 * asin( sqrt((1 - x)/2) )
 *     x in [-1, -0.5):  acos(x) = pi - 2 * asin( sqrt((1 + x)/2) )
 *
 *   The two big-x branches share the same polynomial argument
 *   t = (1 - |x|)/2 (which equals (1+x)/2 when x < 0).
 *
 *   Per-lane:
 *     ax    = |x|
 *     big   = ax > 0.5
 *     t     = big ? (1 - ax)/2 : x^2
 *     v     = big ? sqrt(t)    : x       (note: x, not |x|, so sign
 *                                          propagates through the small case)
 *     pre   = v * P(t)
 *     small result:  pi/2 - pre
 *     big   result:  (x < 0 ? pi : 0) + (x < 0 ? -2*pre : 2*pre)
 *
 * @par Cost
 *   One sqrt, one degree-5 polynomial, three blends, one XOR + one AND
 *   for the big-branch sign and pi-offset.
 */
static inline __m256 mc_acos_ps(__m256 x) {
    const __m256 one      = _mm256_set1_ps(1.0f);
    const __m256 half     = _mm256_set1_ps(0.5f);
    const __m256 zero     = _mm256_setzero_ps();
    const __m256 pio2     = _mm256_set1_ps(1.57079632679489661923f);
    const __m256 pi       = _mm256_set1_ps(3.14159265358979323846f);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    /* Saturate to [-1, 1] so out-of-domain inputs yield 0 or pi, not NaN. */
    x = _mm256_max_ps(_mm256_min_ps(x, one), _mm256_sub_ps(zero, one));
    __m256 ax    = mc_abs_ps(x);
    __m256 big   = _mm256_cmp_ps(ax, half, _CMP_GT_OQ);
    __m256 x_neg = _mm256_cmp_ps(x,  zero, _CMP_LT_OQ);

    __m256 t_small = _mm256_mul_ps(x, x);             /* x^2 == ax^2     */
    __m256 t_big   = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    __m256 t       = _mm256_blendv_ps(t_small, t_big, big);

    /* small: v = x (carries sign through asin's oddness)
     * big:   v = sqrt(t) (always non-negative) */
    __m256 v = _mm256_blendv_ps(x, _mm256_sqrt_ps(t_big), big);

    __m256 p   = mc_poly_estrin_deg5_ps(t, ASIN_COEFFS);
    __m256 pre = _mm256_mul_ps(v, p);

    /* Small: acos(x) = pi/2 - asin(x) = pi/2 - pre. */
    __m256 res_small = _mm256_sub_ps(pio2, pre);

    /* Big: x>=0 -> 2*pre;  x<0 -> pi - 2*pre.
     * Flip the sign of 2*pre when x<0, and add pi when x<0. */
    __m256 two_pre        = _mm256_add_ps(pre, pre);
    __m256 signed_two_pre = _mm256_xor_ps(two_pre, _mm256_and_ps(x_neg, sign_bit));
    __m256 offset_pi      = _mm256_and_ps(x_neg, pi);
    __m256 res_big        = _mm256_add_ps(offset_pi, signed_two_pre);

    return _mm256_blendv_ps(res_small, res_big, big);
}

#endif /* MC_SIMD_ASIN_ACOS_PS_H */
