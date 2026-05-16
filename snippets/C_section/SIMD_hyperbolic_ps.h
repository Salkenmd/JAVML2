/**
 * @file SIMD_hyperbolic_ps.h
 * @brief Vectorized sinh, cosh, tanh for AVX2 float (ps).
 *
 * @par Accuracy
 *   Small-|x| (|x| <= ln2/2) handled by dedicated polynomials from
 *   minimax_coeffs.h, avoiding the catastrophic cancellation that
 *   (e^x - e^-x)/2 suffers near zero. Large-|x| handled via mc_exp_ps.
 *   ~1-2 ULP across the float range.
 *
 * @note Requires AVX2 + FMA. Depends on SIMD_explog_ps.h (mc_exp_ps).
 */

#ifndef MC_SIMD_HYPERBOLIC_PS_H
#define MC_SIMD_HYPERBOLIC_PS_H

#include <immintrin.h>
#include "minimax_coeffs.h"
#include "SIMD_poly_estrin.h"
#include "SIMD_foundation_ps.h"
#include "SIMD_explog_ps.h"

/** @brief Polynomial fit boundary for hyperbolic small-x branch. */
#define MC_LN2_OVER_2  (0.34657359027997265471f)

/* ============================================================
 * sinh(x)
 * ============================================================ */

/**
 * @brief Vectorized hyperbolic sine.
 *
 * @par Algorithm
 *   |x| <= ln2/2:  sinh(x) = x * P(x^2)         (SINH_COEFFS, deg-3)
 *   |x| >  ln2/2:  sinh(x) = (e^x - e^-x) / 2   (e^-x via reciprocal)
 *
 *   Both branches are evaluated per lane and blended on |x| > ln2/2.
 *   The polynomial branch is essential near zero to avoid the
 *   cancellation in (e^x - e^-x) when x is small.
 *
 * @par Cost
 *   One mc_exp_ps + one reciprocal + one degree-3 polynomial + one blend.
 */
static inline __m256 mc_sinh_ps(__m256 x) {
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 bound = _mm256_set1_ps(MC_LN2_OVER_2);

    __m256 ax  = mc_abs_ps(x);
    __m256 big = _mm256_cmp_ps(ax, bound, _CMP_GT_OQ);

    /* Small: x * P(x^2). */
    __m256 x2         = _mm256_mul_ps(x, x);
    __m256 p_small    = mc_poly_estrin_deg3_ps(x2, SINH_COEFFS);
    __m256 sinh_small = _mm256_mul_ps(x, p_small);

    /* Big: (e^x - 1/e^x) / 2. */
    __m256 ex       = mc_exp_ps(x);
    __m256 inv_ex   = _mm256_div_ps(one, ex);
    __m256 sinh_big = _mm256_mul_ps(_mm256_sub_ps(ex, inv_ex), half);

    return _mm256_blendv_ps(sinh_small, sinh_big, big);
}

/* ============================================================
 * cosh(x)
 * ============================================================ */

/**
 * @brief Vectorized hyperbolic cosine.
 *
 * @par Algorithm
 *   |x| <= ln2/2:  cosh(x) = P(x^2)             (COSH_COEFFS, even poly)
 *   |x| >  ln2/2:  cosh(x) = (e^|x| + e^-|x|) / 2
 *
 *   cosh is even, so feeding |x| to the exp branch is a wash but keeps
 *   the exponent argument non-negative (mc_exp_ps handles either fine).
 *   The polynomial branch is technically also fine via (e^x + e^-x)/2
 *   near zero (no cancellation since both terms are positive), but the
 *   dedicated polynomial is cheaper and tighter than two exps.
 *
 * @par Cost
 *   One mc_exp_ps + one reciprocal + one degree-3 polynomial + one blend.
 */
static inline __m256 mc_cosh_ps(__m256 x) {
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 bound = _mm256_set1_ps(MC_LN2_OVER_2);

    __m256 ax  = mc_abs_ps(x);
    __m256 big = _mm256_cmp_ps(ax, bound, _CMP_GT_OQ);

    /* Small: P(x^2) directly — even polynomial, no x-multiply needed. */
    __m256 x2         = _mm256_mul_ps(x, x);
    __m256 cosh_small = mc_poly_estrin_deg3_ps(x2, COSH_COEFFS);

    /* Big: (e^|x| + 1/e^|x|) / 2. */
    __m256 ex       = mc_exp_ps(ax);
    __m256 inv_ex   = _mm256_div_ps(one, ex);
    __m256 cosh_big = _mm256_mul_ps(_mm256_add_ps(ex, inv_ex), half);

    return _mm256_blendv_ps(cosh_small, cosh_big, big);
}

/* ============================================================
 * tanh(x)
 * ============================================================ */

/**
 * @brief Vectorized hyperbolic tangent.
 *
 * @par Algorithm
 *   |x| <= ln2/2:  tanh(x) = x * P(x^2)        (TANH_COEFFS, deg-4)
 *   |x| >  ln2/2:  tanh(x) = 1 - 2 / (e^(2x) + 1)
 *
 *   The exp-based formula handles the full sign range of x naturally:
 *     x → +inf:  e^(2x) → +inf, result → 1
 *     x → -inf:  e^(2x) → 0,   result → -1
 *   mc_exp_ps's input clamp keeps e^(2x) finite, and the rest of the
 *   chain stays well-defined.
 *
 * @par Cost
 *   One mc_exp_ps + one degree-4 polynomial + one blend + one division.
 */
static inline __m256 mc_tanh_ps(__m256 x) {
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 two   = _mm256_set1_ps(2.0f);
    const __m256 bound = _mm256_set1_ps(MC_LN2_OVER_2);

    __m256 ax  = mc_abs_ps(x);
    __m256 big = _mm256_cmp_ps(ax, bound, _CMP_GT_OQ);

    /* Small: x * P(x^2). */
    __m256 x2         = _mm256_mul_ps(x, x);
    __m256 p_small    = mc_poly_estrin_deg4_ps(x2, TANH_COEFFS);
    __m256 tanh_small = _mm256_mul_ps(x, p_small);

    /* Big: 1 - 2 / (e^(2x) + 1). */
    __m256 two_x    = _mm256_add_ps(x, x);
    __m256 e2x      = mc_exp_ps(two_x);
    __m256 denom    = _mm256_add_ps(e2x, one);
    __m256 tanh_big = _mm256_sub_ps(one, _mm256_div_ps(two, denom));

    return _mm256_blendv_ps(tanh_small, tanh_big, big);
}

#endif /* MC_SIMD_HYPERBOLIC_PS_H */
