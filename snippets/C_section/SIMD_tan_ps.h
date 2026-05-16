/**
 * @file SIMD_tan_ps.h
 * @brief Vectorized tangent for AVX2 float (ps).
 *
 * @par Accuracy
 *   ~1-2 ULP across [-pi/4, pi/4] reduced range. Singularities near
 *   x = (k + 1/2)*pi return very large finite values or +/-inf as the
 *   1/tan branch divides by ~0.
 *
 * @note Requires AVX2 + FMA. Depends on SIMD_sincos_ps.h for the
 *       pi/2 reduction constants.
 */

#ifndef MC_SIMD_TAN_PS_H
#define MC_SIMD_TAN_PS_H

#include <immintrin.h>
#include "minimax_coeffs.h"
#include "SIMD_poly_estrin.h"
#include "SIMD_sincos_ps.h"   /* for MC_TWO_OVER_PI, MC_PIO2_HI, MC_PIO2_LO */

/**
 * @brief Vectorized tangent.
 *
 * @par Algorithm
 *   1. Reduce by pi/2: n = round(x * 2/pi), r = x - n*(pi/2), r in [-pi/4, pi/4].
 *   2. Halve unconditionally: y = r/2, so |y| <= pi/8. This is the range
 *      where TAN_COEFFS is accurate. Halving every lane (rather than per-lane
 *      branching) keeps SIMD uniform and only adds a few cheap ops.
 *   3. Evaluate tan(y) = y + y^3 * P(y^2) via mc_poly_estrin_deg3_ps + TAN_COEFFS.
 *   4. Double-angle identity to get tan(r):
 *        tan(r) = 2t / (1 - t^2),   where t = tan(y)
 *   5. Quadrant correction. For odd n we need -cot(r) instead of tan(r):
 *        n even:  result = 2t / (1 - t^2)
 *        n odd:   result = (t^2 - 1) / (2t)   == -1/tan(r)
 *      Implemented by blending the numerator/denominator pair, so the
 *      whole function uses a single division.
 */
static inline __m256 mc_tan_ps(__m256 x) {
    /* --- 1. Range reduction by pi/2 ----------------------------------- */
    const __m256 two_over_pi = _mm256_set1_ps(MC_TWO_OVER_PI);
    __m256 fn = _mm256_round_ps(
        _mm256_mul_ps(x, two_over_pi),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );

    const __m256 pio2_hi = _mm256_set1_ps(MC_PIO2_HI);
    const __m256 pio2_lo = _mm256_set1_ps(MC_PIO2_LO);
    __m256 r = _mm256_fnmadd_ps(fn, pio2_hi, x);
    r        = _mm256_fnmadd_ps(fn, pio2_lo, r);

    /* --- 2. Halve so |y| <= pi/8 -------------------------------------- */
    __m256 y = _mm256_mul_ps(r, _mm256_set1_ps(0.5f));

    /* --- 3. tan(y) = y + y^3 * P(y^2) --------------------------------- */
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 p  = mc_poly_estrin_deg3_ps(y2, TAN_COEFFS);
    __m256 y3 = _mm256_mul_ps(y2, y);
    __m256 t  = _mm256_fmadd_ps(y3, p, y);

    /* --- 4. Double-angle pieces --------------------------------------- */
    __m256 t2          = _mm256_mul_ps(t, t);
    __m256 two_t       = _mm256_add_ps(t, t);
    __m256 one_minus_t2 = _mm256_sub_ps(_mm256_set1_ps(1.0f), t2);
    __m256 t2_minus_one = _mm256_sub_ps(t2, _mm256_set1_ps(1.0f));

    /* --- 5. Quadrant select ------------------------------------------- */
    __m256i ni    = _mm256_cvtps_epi32(fn);
    __m256i v_one = _mm256_set1_epi32(1);
    __m256  odd_mask = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(ni, v_one), v_one)
    );

    /* even: 2t / (1 - t^2);  odd: (t^2 - 1) / (2t) */
    __m256 num = _mm256_blendv_ps(two_t,        t2_minus_one, odd_mask);
    __m256 den = _mm256_blendv_ps(one_minus_t2, two_t,        odd_mask);

    return _mm256_div_ps(num, den);
}

#endif /* MC_SIMD_TAN_PS_H */
