/**
 * @file SIMD_atan_ps.h
 * @brief Vectorized atan and atan2 for AVX2 float (ps).
 *
 * @par Accuracy
 *   ~1-2 ULP across the full real line. atan2 inherits this plus the
 *   conditioning of the y/x ratio at the call site.
 *
 * @note Requires AVX2 + FMA. Uses SIMD_foundation_ps.h for mc_abs_ps.
 */

#ifndef MC_SIMD_ATAN_PS_H
#define MC_SIMD_ATAN_PS_H

#include <immintrin.h>
#include "minimax_coeffs.h"
#include "SIMD_poly_estrin.h"
#include "SIMD_foundation_ps.h"

/* ============================================================
 * atan(x)
 * ============================================================ */

/**
 * @brief Vectorized arctangent.
 *
 * @par Algorithm
 *   Two-stage range reduction so the polynomial only ever sees
 *   |v| <= tan(pi/8) ~ 0.4142.
 *
 *   Stage 1 — fold |x| > 1 down via reciprocal:
 *     ax = |x|
 *     if ax > 1:  u = 1/ax,   atan(ax) = pi/2 - atan(u)
 *     else:        u = ax,     atan(ax) = atan(u)
 *     Now u in [0, 1].
 *
 *   Stage 2 — fold u > tan(pi/8) via half-angle shift:
 *     if u > tan(pi/8):  v = (u-1)/(u+1),  atan(u) = pi/4 + atan(v)
 *     else:               v = u,            atan(u) = atan(v)
 *     Now v in [-tan(pi/8), tan(pi/8)].
 *
 *   Stage 3 — polynomial:
 *     atan(v) = v * P(v^2),  P is degree-4 via ATAN_COEFFS.
 *
 *   Stage 4 — reapply original sign:
 *     atan is odd, so just XOR the sign bit of x onto the result.
 *
 * @par Cost
 *   Two divisions (1/ax and (u-1)/(u+1)), one degree-4 polynomial,
 *   two blends, one XOR for sign.
 */
static inline __m256 mc_atan_ps(__m256 x) {
    const __m256 one      = _mm256_set1_ps(1.0f);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    /* Save sign, work on |x|. */
    __m256 sign_x = _mm256_and_ps(x, sign_bit);
    __m256 ax     = mc_abs_ps(x);

    /* --- Stage 1: u = (ax > 1) ? 1/ax : ax --------------------------- */
    __m256 swap_mask = _mm256_cmp_ps(ax, one, _CMP_GT_OQ);
    __m256 inv_ax    = _mm256_div_ps(one, ax);
    __m256 u         = _mm256_blendv_ps(ax, inv_ax, swap_mask);

    /* --- Stage 2: v = (u > tan(pi/8)) ? (u-1)/(u+1) : u -------------- */
    const __m256 tan_pio8 = _mm256_set1_ps(0.41421356237309504880f);
    __m256 shift_mask = _mm256_cmp_ps(u, tan_pio8, _CMP_GT_OQ);
    __m256 v_red      = _mm256_div_ps(_mm256_sub_ps(u, one),
                                      _mm256_add_ps(u, one));
    __m256 v          = _mm256_blendv_ps(u, v_red, shift_mask);

    /* --- Stage 3: atan(v) = v * P(v^2) ------------------------------- */
    __m256 v2     = _mm256_mul_ps(v, v);
    __m256 p      = mc_poly_estrin_deg4_ps(v2, ATAN_COEFFS);
    __m256 atan_v = _mm256_mul_ps(v, p);

    /* --- Stage 4: reconstruct shifts --------------------------------- */
    /* Inner: add pi/4 where shift_mask. (AND with mask gives pi/4 or 0.) */
    const __m256 pio4 = _mm256_set1_ps(0.78539816339744830962f);
    __m256 atan_u = _mm256_add_ps(atan_v, _mm256_and_ps(shift_mask, pio4));

    /* Outer: if swap, atan(ax) = pi/2 - atan(u); else atan(u). */
    const __m256 pio2 = _mm256_set1_ps(1.57079632679489661923f);
    __m256 atan_ax_swap = _mm256_sub_ps(pio2, atan_u);
    __m256 atan_ax      = _mm256_blendv_ps(atan_u, atan_ax_swap, swap_mask);

    /* Reapply sign of x via XOR. atan is odd. */
    return _mm256_xor_ps(atan_ax, sign_x);
}

/* ============================================================
 * atan2(y, x)
 * ============================================================ */

/**
 * @brief Vectorized two-argument arctangent.
 *
 * @par Algorithm
 *   Compute atan(y/x), then offset by quadrant:
 *     x > 0:           result = atan(y/x)
 *     x < 0, y >= 0:   result = atan(y/x) + pi
 *     x < 0, y <  0:   result = atan(y/x) - pi
 *     x == 0, y > 0:   result = +pi/2   (falls out naturally: y/x = +inf, atan = pi/2)
 *     x == 0, y < 0:   result = -pi/2   (y/x = -inf, atan = -pi/2)
 *
 * @par Edge cases
 *   - (y, x) == (0, 0): returns NaN (0/0 = NaN propagates). If you need
 *     0 here, mask it at the call site.
 *   - Very large |y| and |x| of similar magnitude can lose precision in
 *     the y/x ratio. For shipping code that handles huge inputs, scale
 *     both by 1/max(|x|, |y|) before the ratio.
 */
static inline __m256 mc_atan2_ps(__m256 y, __m256 x) {
    const __m256 zero     = _mm256_setzero_ps();
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 pi       = _mm256_set1_ps(3.14159265358979323846f);

    __m256 ratio    = _mm256_div_ps(y, x);
    __m256 atan_r   = mc_atan_ps(ratio);

    /* Offset = sign(y) * pi, applied only when x < 0. */
    __m256 x_neg     = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    __m256 y_sign    = _mm256_and_ps(y, sign_bit);              /* sign bit of y */
    __m256 signed_pi = _mm256_xor_ps(pi, y_sign);                /* +pi or -pi   */
    __m256 offset    = _mm256_and_ps(x_neg, signed_pi);          /* 0 if x>=0    */

    return _mm256_add_ps(atan_r, offset);
}

#endif /* MC_SIMD_ATAN_PS_H */
