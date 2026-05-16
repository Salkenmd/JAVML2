/**
 * @file SIMD_pow_ps.h
 * @brief Vectorized power function for AVX2 float (ps).
 *
 * @par Accuracy
 *   ~1-3 ULP for typical inputs. Error grows with |y * ln(x)| because
 *   any error in ln(x) gets multiplied by y before reaching exp. For
 *   huge |y| (e.g. >10^4) consider a dedicated extended-precision path.
 *
 * @note Requires AVX2 + FMA. Depends on SIMD_explog_ps.h.
 */

#ifndef MC_SIMD_POW_PS_H
#define MC_SIMD_POW_PS_H

#include <immintrin.h>
#include "SIMD_explog_ps.h"

/**
 * @brief Vectorized x^y.
 *
 * @par Algorithm
 *   pow(x, y) = exp(y * ln(x))
 *   Composes the existing mc_ln_ps and mc_exp_ps. No new constants or
 *   coefficient tables — all precision comes from those two pieces.
 *
 * @par Edge case behaviour (per lane)
 *   x > 0:                normal result
 *   x == 0, y > 0:        0       (ln(0) = -inf, y*-inf = -inf, exp(-inf) = 0)
 *   x == 0, y < 0:        +inf    (y*-inf = +inf, exp(+inf) saturates to FLT_MAX)
 *   x == 0, y == 0:       NaN     (0 * -inf = NaN propagates through exp)
 *                                 — conventional pow returns 1; mask at call site if needed.
 *   x <  0:               NaN/garbage — mc_ln_ps returns -inf for x<0, so the
 *                                  result is the same path as x==0. If you need
 *                                  integer-exponent handling for negative bases
 *                                  (e.g. (-2)^3 = -8), write a dedicated mc_powi_ps.
 *   x == 1:               1       (ln(1) = 0)
 *   y == 0, x > 0:        1
 *
 * @par Cost
 *   One mc_ln_ps + one FMA-free multiply + one mc_exp_ps.
 *   (The multiply is plain because there's nothing to add to it.)
 *
 * @param x  Base. Lanes <= 0 produce non-finite or NaN per the table above.
 * @param y  Exponent.
 */
static inline __m256 mc_pow_ps(__m256 x, __m256 y) {
    __m256 lx = mc_ln_ps(x);
    __m256 yl = _mm256_mul_ps(y, lx);
    return mc_exp_ps(yl);
}

#endif /* MC_SIMD_POW_PS_H */
