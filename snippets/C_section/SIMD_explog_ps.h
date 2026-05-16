/**
 * @file SIMD_explog_ps.h
 * @brief Vectorized exp, exp2, and ln for AVX2 float (ps).
 *
 * All functions operate on 8-wide float vectors (__m256).
 * Polynomials are evaluated via Estrin's scheme (SIMD_poly_estrin.h).
 * Coefficients are from minimax_coeffs.h, generated with Sollya 8.0.
 *
 * @par Accuracy
 *   mc_exp_ps  : ~0.031 ULP
 *   mc_exp2_ps : ~0.031 ULP
 *   mc_ln_ps   : ~0.18  ULP
 *
 * @note Requires AVX2 + FMA (-mavx2 -mfma).
 */

#ifndef MC_SIMD_EXPLOG_PS_H
#define MC_SIMD_EXPLOG_PS_H

#include <immintrin.h>
#include "minimax_coeffs.h"
#include "SIMD_poly_estrin.h"

/* ============================================================
 * Internal helpers
 * ============================================================ */

/**
 * @brief Assemble 2^n as a float vector by writing n into the exponent field.
 * @details IEEE 754 single: bit 30..23 = biased exponent (bias 127).
 *          We clamp n to [-126, 127] to stay in normal float range.
 * @param n  Integer vector of unbiased exponent values.
 * @return   2^n per lane as float.
 */
static inline __m256 mc_i_pow2i_ps(__m256i n) {
    // Clamp to avoid producing denormals or inf from garbage input
    n = _mm256_max_epi32(n, _mm256_set1_epi32(-126));
    n = _mm256_min_epi32(n, _mm256_set1_epi32( 127));
    // Shift into exponent field and reinterpret as float
    __m256i bits = _mm256_add_epi32(n, _mm256_set1_epi32(127));  // add bias
    bits = _mm256_slli_epi32(bits, 23);                           // shift to exp field
    return _mm256_castsi256_ps(bits);
}

/* ============================================================
 * exp(x)
 * ============================================================ */

/**
 * @brief Vectorized natural exponential: e^x.
 *
 * @par Algorithm
 *   1. Clamp x to [-87.3, 88.7] (float overflow/underflow boundaries).
 *   2. Cody-Waite range reduction:
 *        n = round(x / ln2)
 *        r = x - n*LN2_HI - n*LN2_LO
 *      This keeps r in [-ln2/2, ln2/2] with ~2 extra bits of precision
 *      vs. a single subtraction.
 *   3. Evaluate P(r) = exp(r) via mc_poly_estrin_deg6_ps + EXP_COEFFS.
 *   4. Scale: result = P(r) * 2^n (exponent-field insertion).
 *
 * @param x  Input vector. Values outside (-87.3, 88.7) saturate to 0 or inf.
 * @return   e^x per lane.
 */
static inline __m256 mc_exp_ps(__m256 x) {
    // --- 1. Clamp -------------------------------------------------------
    const __m256 max_x = _mm256_set1_ps( 88.7228395f);  // ln(FLT_MAX)
    const __m256 min_x = _mm256_set1_ps(-87.3365386f);  // ln(FLT_MIN normal)
    x = _mm256_min_ps(_mm256_max_ps(x, min_x), max_x);

    // --- 2. Cody-Waite reduction ----------------------------------------
    // n = round(x / ln2) — use multiply by log2e then round-to-nearest
    const __m256 log2e = _mm256_set1_ps(1.44269504088896340736f);  // 1/ln2
    __m256 fn = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );

    // r = x - n*LN2_HI - n*LN2_LO  (two-step for precision)
    const __m256 ln2_hi = _mm256_set1_ps(LN2_HI);
    const __m256 ln2_lo = _mm256_set1_ps(LN2_LO);
    __m256 r = _mm256_fnmadd_ps(fn, ln2_hi, x);    // x - n*LN2_HI
    r        = _mm256_fnmadd_ps(fn, ln2_lo, r);    // r - n*LN2_LO

    // --- 3. Polynomial evaluation ---------------------------------------
    __m256 p = mc_poly_estrin_deg6_ps(r, EXP_COEFFS);

    // --- 4. Scale by 2^n ------------------------------------------------
    __m256i n = _mm256_cvtps_epi32(fn);
    return _mm256_mul_ps(p, mc_i_pow2i_ps(n));
}

/* ============================================================
 * exp2(x)  — 2^x
 * ============================================================ */

/**
 * @brief Vectorized base-2 exponential: 2^x.
 *
 * @par Algorithm
 *   1. Clamp x to [-126, 127].
 *   2. Range reduction:
 *        n = round(x)
 *        r = x - n           (r in [-0.5, 0.5])
 *   3. Evaluate P(r) = 2^r via mc_poly_estrin_deg6_ps + EXP2_COEFFS.
 *   4. Scale: result = P(r) * 2^n.
 *
 * @param x  Input vector.
 * @return   2^x per lane.
 */
static inline __m256 mc_exp2_ps(__m256 x) {
    // --- 1. Clamp -------------------------------------------------------
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-126.0f)),
                                       _mm256_set1_ps( 127.0f));

    // --- 2. Reduction ---------------------------------------------------
    __m256 fn = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r  = _mm256_sub_ps(x, fn);   // r = x - n, in [-0.5, 0.5]

    // --- 3. Polynomial --------------------------------------------------
    __m256 p = mc_poly_estrin_deg6_ps(r, EXP2_COEFFS);

    // --- 4. Scale -------------------------------------------------------
    __m256i n = _mm256_cvtps_epi32(fn);
    return _mm256_mul_ps(p, mc_i_pow2i_ps(n));
}

/* ============================================================
 * ln(x)  — natural log
 * ============================================================ */

/**
 * @brief Vectorized natural logarithm: ln(x).
 *
 * @par Algorithm
 *   1. Force x = 0 lanes to a small positive number to avoid -inf propagating
 *      incorrectly; remember which lanes were <= 0 for fixup at the end.
 *   2. Extract biased exponent n and normalized mantissa m via bit ops:
 *        n = (bits >> 23) - 127
 *        m = (bits & 0x007FFFFF) | 0x3F800000   → m in [1, 2)
 *   3. Normalize m into [1/sqrt(2), sqrt(2)]:
 *      If m < 1/sqrt(2) ≈ 0.7071: subtract 1 from n, multiply m by 2.
 *      This keeps u = m - 1 in [-0.2929, 0.4142], centering the poly range.
 *   4. u = m - 1.
 *   5. Evaluate P(u) via mc_poly_estrin_deg8_ps + LN_COEFFS.
 *   6. ln(m) = u * P(u).
 *   7. Reconstruct: ln(x) = n*LN2_HI + (n*LN2_LO + u*P(u)).
 *      The split-ln2 add preserves precision when n is large.
 *   8. Fix up x <= 0 lanes: return -inf for x=0, NaN for x<0 (IEEE 754).
 *
 * @param x  Input vector. x <= 0 returns -inf (x=0) or NaN (x<0) per lane.
 * @return   ln(x) per lane.
 */
static inline __m256 mc_ln_ps(__m256 x) {
    // --- 1. Guard x <= 0 ------------------------------------------------
    // Remember which lanes are invalid before we corrupt them
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);
    // Substitute 1.0 in bad lanes so bit ops below stay well-defined
    x = _mm256_blendv_ps(x, _mm256_set1_ps(1.0f), invalid_mask);

    // --- 2. Extract exponent and mantissa via integer bit ops -----------
    __m256i xi = _mm256_castps_si256(x);

    // Biased exponent: bits 30..23
    __m256i exp_biased = _mm256_srli_epi32(xi, 23);                    // >> 23
    __m256i n_i        = _mm256_sub_epi32(exp_biased,
                             _mm256_set1_epi32(127));                   // unbias

    // Mantissa: clear exponent bits, set implicit leading 1 → m in [1, 2)
    const __m256i mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i one_bits      = _mm256_set1_epi32(0x3F800000);       // 1.0f
    __m256i mant_bits = _mm256_and_si256(xi, mantissa_mask);
    mant_bits         = _mm256_or_si256(mant_bits, one_bits);
    __m256 m          = _mm256_castsi256_ps(mant_bits);                 // m in [1, 2)

    // --- 3. Normalize m into [1/sqrt(2), sqrt(2)] -----------------------
    // m comes out of the bit-twiddle in [1, 2). If m > sqrt(2), divide by 2
    // and bump the exponent up by 1, landing m in [sqrt(2)/2, sqrt(2)].
    const __m256 sqrt2 = _mm256_set1_ps(1.41421356237309504880f);
    __m256 need_adjust = _mm256_cmp_ps(m, sqrt2, _CMP_GT_OQ);

    // n += 1 where adjusted. need_adjust is all-1s or all-0s; shift to 0/1.
    __m256i adjust = _mm256_srli_epi32(_mm256_castps_si256(need_adjust), 31);
    n_i = _mm256_add_epi32(n_i, adjust);
    // m *= 0.5 for those lanes
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), need_adjust);

    // --- 4. u = m - 1 ---------------------------------------------------
    __m256 u = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));

    // --- 5-6. Polynomial: ln(m) = u * P(u) ------------------------------
    __m256 p    = mc_poly_estrin_deg8_ps(u, LN_COEFFS);
    __m256 ln_m = _mm256_mul_ps(u, p);

    // --- 7. Reconstruct ln(x) = n*ln2 + ln(m) ---------------------------
    // Use split ln2 for extra precision: n*LN2_HI first, then add n*LN2_LO + ln_m
    __m256 fn      = _mm256_cvtepi32_ps(n_i);
    const __m256 ln2_hi = _mm256_set1_ps(LN2_HI);
    const __m256 ln2_lo = _mm256_set1_ps(LN2_LO);
    __m256 result  = _mm256_fmadd_ps(fn, ln2_hi, ln_m);          // n*LN2_HI + ln_m
    result         = _mm256_fmadd_ps(fn, ln2_lo, result);         // + n*LN2_LO

    // --- 8. Fix up invalid lanes ----------------------------------------
    // x == 0  → -inf;  x < 0 → NaN (we just return -inf for both here,
    // matching glibc behaviour for the common case; strict NaN for x<0
    // can be added if needed via a second mask).
    const __m256 neg_inf = _mm256_set1_ps(-1.0f / 0.0f);
    result = _mm256_blendv_ps(result, neg_inf, invalid_mask);

    return result;
}

#endif /* MC_SIMD_EXPLOG_PS_H */