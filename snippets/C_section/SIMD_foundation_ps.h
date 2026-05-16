/**
 * @file SIMD_foundation_ps.h
 * @brief Common gamedev math utilities for AVX2 float (ps).
 *
 * Covers basic operations, clamping, interpolation, step functions,
 * and sign utilities. All functions operate on 8-wide float vectors (__m256).
 *
 * @note Requires AVX2 and FMA support (compile with -mavx2 -mfma).
 */

#ifndef MC_SIMD_FOUNDATION_PS_H
#define MC_SIMD_FOUNDATION_PS_H
#include <immintrin.h>

/* ============================================================
 * Basic Operations
 * ============================================================ */

/**
 * @brief Absolute value |x|.
 * @details Clears the sign bit (bit 31) of each IEEE 754 float lane.
 * @param x Input vector.
 * @return |x| per lane.
 */
static inline __m256 mc_abs_ps(__m256 x) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
}

/**
 * @brief Per-lane minimum.
 * @return min(a, b) per lane. NaN behaviour follows _mm256_min_ps.
 */
static inline __m256 mc_min_ps(__m256 a, __m256 b) {
    return _mm256_min_ps(a, b);
}

/**
 * @brief Per-lane maximum.
 * @return max(a, b) per lane. NaN behaviour follows _mm256_max_ps.
 */
static inline __m256 mc_max_ps(__m256 a, __m256 b) {
    return _mm256_max_ps(a, b);
}

/**
 * @brief Clamp x to [min_val, max_val].
 * @math max(min_val, min(max_val, x))
 */
static inline __m256 mc_clamp_ps(__m256 x, __m256 min_val, __m256 max_val) {
    return _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);
}

/**
 * @brief Saturate x to [0, 1].
 * @math clamp(x, 0, 1)
 */
static inline __m256 mc_saturate_ps(__m256 x) {
    return mc_clamp_ps(x, _mm256_setzero_ps(), _mm256_set1_ps(1.0f));
}

/* ============================================================
 * Interpolation
 * ============================================================ */

/**
 * @brief Linear interpolation.
 * @math a + t * (b - a)
 * @note Uses FMA for one fewer rounding step vs. a + t*(b-a) naively.
 * @param a Start value (t=0).
 * @param b End value (t=1).
 * @param t Blend factor. Values outside [0,1] extrapolate.
 */
static inline __m256 mc_lerp_ps(__m256 a, __m256 b, __m256 t) {
    __m256 diff = _mm256_sub_ps(b, a);
    return _mm256_fmadd_ps(t, diff, a);
}

/**
 * @brief Inverse linear interpolation.
 * @math (x - a) / (b - a)
 * @warning Result is undefined (inf/NaN) when a == b. No guard is applied;
 *          caller must ensure the range is non-degenerate.
 * @param a Range start.
 * @param b Range end.
 * @param x Value to invert.
 * @return t such that lerp(a, b, t) == x.
 */
static inline __m256 mc_inverse_lerp_ps(__m256 a, __m256 b, __m256 x) {
    __m256 num = _mm256_sub_ps(x, a);
    __m256 den = _mm256_sub_ps(b, a);
    return _mm256_div_ps(num, den);
}

/* ============================================================
 * Steps
 * ============================================================ */

/**
 * @brief Step function.
 * @math 1.0 if x >= edge, else 0.0
 */
static inline __m256 mc_step_ps(__m256 edge, __m256 x) {
    __m256 mask = _mm256_cmp_ps(x, edge, _CMP_GE_OQ);
    return _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
}

/**
 * @brief Smoothstep (Hermite interpolation).
 * @math 3t^2 - 2t^3,  where t = clamp((x - a) / (b - a), 0, 1)
 * @details Provides C1 continuity: zero derivative at both edges.
 *          Common use: smooth LOD blends, edge fades, shader masks.
 *
 * @param a  Lower edge. Below this, result is 0.
 * @param b  Upper edge. Above this, result is 1.
 * @param x  Input value.
 * @return   Smoothly interpolated value in [0, 1].
 *
 * @note When a == b the function returns 0 for all x to avoid a
 *       divide-by-zero. This is a safe sentinel, not a mathematically
 *       meaningful result — callers should avoid degenerate ranges.
 */
static inline __m256 mc_smoothstep_ps(__m256 a, __m256 b, __m256 x) {
    __m256 num = _mm256_sub_ps(x, a);
    __m256 den = _mm256_sub_ps(b, a);

    // Detect degenerate range (a == b)
    __m256 is_zero = _mm256_cmp_ps(den, _mm256_setzero_ps(), _CMP_EQ_OQ);

    // Safe divide: replace zero denominator with 1 to avoid NaN/inf
    __m256 safe_den = _mm256_blendv_ps(den, _mm256_set1_ps(1.0f), is_zero);
    __m256 t = _mm256_div_ps(num, safe_den);

    // Clamp t to [0, 1]
    t = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(t, _mm256_set1_ps(1.0f)));

    // Polynomial: t^2 * (3 - 2t)
    __m256 poly = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), t, _mm256_set1_ps(3.0f));
    __m256 result = _mm256_mul_ps(t, _mm256_mul_ps(t, poly));

    // Zero out degenerate lanes
    return _mm256_andnot_ps(is_zero, result);
}

/* ============================================================
 * Select / Sign
 * ============================================================ */

/**
 * @brief Sign function.
 * @math 1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0
 */
static inline __m256 mc_sign_ps(__m256 x) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one  = _mm256_set1_ps( 1.0f);
    const __m256 v_neg  = _mm256_set1_ps(-1.0f);

    __m256 pos_mask = _mm256_cmp_ps(x, v_zero, _CMP_GT_OQ);
    __m256 neg_mask = _mm256_cmp_ps(x, v_zero, _CMP_LT_OQ);

    __m256 res = _mm256_blendv_ps(v_zero, v_one, pos_mask);
    return _mm256_blendv_ps(res, v_neg, neg_mask);
}

/* ============================================================
 * Rounding family
 * ============================================================ */

/** @brief floor(x) — round toward -inf. */
static inline __m256 mc_floor_ps(__m256 x) {
    return _mm256_round_ps(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

/** @brief ceil(x) — round toward +inf. */
static inline __m256 mc_ceil_ps(__m256 x) {
    return _mm256_round_ps(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

/** @brief round(x) — round to nearest, ties to even (IEEE default). */
static inline __m256 mc_round_ps(__m256 x) {
    return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

/** @brief trunc(x) — round toward zero. */
static inline __m256 mc_trunc_ps(__m256 x) {
    return _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

/**
 * @brief Fractional part: x - floor(x), result in [0, 1) for finite inputs.
 * GLSL-style: fract(-0.3) = 0.7. For truncated form, use x - mc_trunc_ps(x).
 */
static inline __m256 mc_fract_ps(__m256 x) {
    return _mm256_sub_ps(x, mc_floor_ps(x));
}

/* ============================================================
 * Modulo, wrap, remap
 * ============================================================ */

/**
 * @brief GLSL-style float modulo: x - y * floor(x/y).
 * Result has the same sign as y. Undefined when y == 0.
 */
static inline __m256 mc_mod_ps(__m256 x, __m256 y) {
    __m256 q = _mm256_div_ps(x, y);
    return _mm256_fnmadd_ps(y, mc_floor_ps(q), x);
}

/**
 * @brief Wrap x into the half-open interval [lo, hi). Undefined when hi == lo.
 */
static inline __m256 mc_wrap_ps(__m256 x, __m256 lo, __m256 hi) {
    __m256 range = _mm256_sub_ps(hi, lo);
    return _mm256_add_ps(lo, mc_mod_ps(_mm256_sub_ps(x, lo), range));
}

/**
 * @brief Linear remap: x in [a0, a1] -> result in [b0, b1].
 * Single FMA for the final mul-add. Undefined when a0 == a1.
 */
static inline __m256 mc_remap_ps(__m256 x,
                                 __m256 a0, __m256 a1,
                                 __m256 b0, __m256 b1)
{
    __m256 t = _mm256_div_ps(_mm256_sub_ps(x,  a0),
                             _mm256_sub_ps(a1, a0));
    return _mm256_fmadd_ps(t, _mm256_sub_ps(b1, b0), b0);
}

/* ============================================================
 * Sign and selection helpers
 * ============================================================ */

/**
 * @brief copysign(mag, sign): magnitude of mag with the sign of sign.
 * Bitwise: clear sign bit of mag, OR with sign bit of sign.
 */
static inline __m256 mc_copysign_ps(__m256 mag, __m256 sign) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 abs_mag = _mm256_andnot_ps(sign_mask, mag);
    __m256 sign_of = _mm256_and_ps(sign_mask, sign);
    return _mm256_or_ps(abs_mag, sign_of);
}

/**
 * @brief select(mask, on_true, on_false): named blendv with natural order.
 * Compiles to a single VBLENDVPS.
 */
static inline __m256 mc_select_ps(__m256 mask, __m256 on_true, __m256 on_false) {
    return _mm256_blendv_ps(on_false, on_true, mask);
}

/* ============================================================
 * Smootherstep (C2-continuous Hermite)
 * ============================================================ */

/**
 * @brief Smootherstep: 6t^5 - 15t^4 + 10t^3 on the clamped t.
 * C2-continuous (zero first and second derivative at edges).
 * Degenerate a == b returns 0 per lane.
 */
static inline __m256 mc_smootherstep_ps(__m256 a, __m256 b, __m256 x) {
    __m256 num = _mm256_sub_ps(x, a);
    __m256 den = _mm256_sub_ps(b, a);
    __m256 is_zero  = _mm256_cmp_ps(den, _mm256_setzero_ps(), _CMP_EQ_OQ);
    __m256 safe_den = _mm256_blendv_ps(den, _mm256_set1_ps(1.0f), is_zero);
    __m256 t = _mm256_div_ps(num, safe_den);
    t = _mm256_max_ps(_mm256_setzero_ps(),
                      _mm256_min_ps(t, _mm256_set1_ps(1.0f)));

    __m256 p = _mm256_fmsub_ps(t, _mm256_set1_ps(6.0f),  _mm256_set1_ps(15.0f));
    p        = _mm256_fmadd_ps(t, p,                      _mm256_set1_ps(10.0f));
    __m256 t3 = _mm256_mul_ps(t, _mm256_mul_ps(t, t));
    __m256 result = _mm256_mul_ps(t3, p);

    return _mm256_andnot_ps(is_zero, result);
}

/* ============================================================
 * Pingpong and angle wrap
 * ============================================================ */

/**
 * @brief Bounce x within [0, length]: 0,1,...,length-1,length,length-1,...
 * Symmetric for negative x. Undefined when length <= 0.
 */
static inline __m256 mc_pingpong_ps(__m256 x, __m256 length) {
    __m256 two_len = _mm256_add_ps(length, length);
    __m256 t = mc_mod_ps(x, two_len);
    return _mm256_sub_ps(length, mc_abs_ps(_mm256_sub_ps(t, length)));
}

/** @brief Fold angle into [-pi, pi). */
static inline __m256 mc_wrap_angle_ps(__m256 angle) {
    const __m256 neg_pi = _mm256_set1_ps(-3.14159265358979323846f);
    const __m256 pos_pi = _mm256_set1_ps( 3.14159265358979323846f);
    return mc_wrap_ps(angle, neg_pi, pos_pi);
}

/* ============================================================
 * Degree / radian conversion
 * ============================================================ */

/** @brief Degrees -> radians. */
static inline __m256 mc_to_radians_ps(__m256 deg) {
    return _mm256_mul_ps(deg, _mm256_set1_ps(0.01745329251994329577f));
}

/** @brief Radians -> degrees. */
static inline __m256 mc_to_degrees_ps(__m256 rad) {
    return _mm256_mul_ps(rad, _mm256_set1_ps(57.29577951308232088f));
}

/* ============================================================
 * Classification masks
 * ============================================================ */

/** @brief Per-lane NaN mask. */
static inline __m256 mc_is_nan_ps(__m256 x) {
    return _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
}

/** @brief Per-lane +/-inf mask. */
static inline __m256 mc_is_inf_ps(__m256 x) {
    const __m256 inf = _mm256_set1_ps(1.0f / 0.0f);
    return _mm256_cmp_ps(mc_abs_ps(x), inf, _CMP_EQ_OQ);
}

/** @brief Per-lane finite mask (not NaN, not +/-inf). */
static inline __m256 mc_is_finite_ps(__m256 x) {
    const __m256 inf = _mm256_set1_ps(1.0f / 0.0f);
    return _mm256_cmp_ps(mc_abs_ps(x), inf, _CMP_LT_OQ);
}

/** @brief Absolute-tolerance fuzzy equality mask: |a-b| <= eps. */
static inline __m256 mc_approx_equal_ps(__m256 a, __m256 b, __m256 eps) {
    __m256 diff = mc_abs_ps(_mm256_sub_ps(a, b));
    return _mm256_cmp_ps(diff, eps, _CMP_LE_OQ);
}

#endif // MC_SIMD_FOUNDATION_PS_H
