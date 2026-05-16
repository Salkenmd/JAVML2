/**
 * @file mc_simd_math.h
 * @brief Single-header AVX2 + FMA math library for gamedev (float / __m256).
 *
 * Game-grade transcendentals and foundation utilities on 8-wide float vectors.
 * Polynomials are minimax fits generated with Sollya 8.0 (~0.04 ULP fit error);
 * measured end-to-end accuracy is ~1-5 ULP for typical inputs. See the test
 * harness for a per-function ULP report across normal and extreme ranges.
 *
 * @par Usage
 * @code
 *   #include "mc_simd_math.h"
 *   if (!mc_supports_avx2_fma()) { ... bail out ... }
 *   __m256 y = mc_sin_ps(x);
 * @endcode
 *
 * @par Build flags
 *   GCC/Clang:  -O2 -mavx2 -mfma
 *   MSVC:       /O2 /arch:AVX2
 *
 * @par Sections
 *   1. CPU feature detection (mc_supports_avx2_fma)
 *   2. Foundation (abs, min/max, clamp, lerp, smoothstep, sign, ...)
 *   3. Polynomial evaluators (Estrin's scheme, deg-3 / 4 / 5 / 6 / 8)
 *   4. Minimax coefficient tables
 *   5. Square root and reciprocal square root
 *   6. exp, exp2, ln
 *   7. sin, cos, sincos
 *   8. tan
 *   9. asin, acos
 *  10. atan, atan2
 *  11. sinh, cosh, tanh
 *  12. pow
 *
 * @note All functions are `static inline`. Include this header in any TU
 *       that needs them; no separate .c file required.
 */

#ifndef MC_SIMD_MATH_H
#define MC_SIMD_MATH_H

#include <stdint.h>
#include <immintrin.h>

/* =====================================================================
 * 1. CPU FEATURE DETECTION
 * ===================================================================== */

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline void mc_i_cpuid(int leaf, int subleaf, int out[4]) {
      __cpuidex(out, leaf, subleaf);
  }
  static inline uint64_t mc_i_xgetbv(unsigned idx) {
      return (uint64_t)_xgetbv(idx);
  }
#else
  #include <cpuid.h>
  static inline void mc_i_cpuid(int leaf, int subleaf, int out[4]) {
      unsigned a, b, c, d;
      __cpuid_count(leaf, subleaf, a, b, c, d);
      out[0] = (int)a; out[1] = (int)b; out[2] = (int)c; out[3] = (int)d;
  }
  static inline uint64_t mc_i_xgetbv(unsigned idx) {
      uint32_t eax, edx;
      /* xgetbv opcode 0F 01 D0, inline-encoded so we compile without -mxsave. */
      __asm__ volatile (".byte 0x0f, 0x01, 0xd0"
                        : "=a"(eax), "=d"(edx) : "c"(idx));
      return ((uint64_t)edx << 32) | (uint64_t)eax;
  }
#endif

/** @brief Detailed CPU feature snapshot. */
typedef struct {
    int has_osxsave;
    int os_ymm_enabled;
    int has_sse2;
    int has_sse4_1;
    int has_avx;
    int has_avx2;
    int has_fma;
} mc_cpu_info_t;

/**
 * @brief Probe the CPU for SIMD-relevant features.
 *
 * Safe to call on any x86-64 CPU; uses only baseline CPUID + XGETBV.
 * Compile this function with baseline flags (not -mavx2) so it runs on
 * unsupported machines without faulting.
 */
static inline mc_cpu_info_t mc_detect_cpu(void) {
    mc_cpu_info_t info = {0};
    int r[4];
    mc_i_cpuid(0, 0, r);
    int max_leaf = r[0];
    if (max_leaf < 1) return info;

    mc_i_cpuid(1, 0, r);
    int ecx = r[2], edx = r[3];
    info.has_sse2     = (edx >> 26) & 1;
    info.has_sse4_1   = (ecx >> 19) & 1;
    info.has_fma      = (ecx >> 12) & 1;
    info.has_osxsave  = (ecx >> 27) & 1;
    info.has_avx      = (ecx >> 28) & 1;

    if (info.has_osxsave) {
        uint64_t xcr0 = mc_i_xgetbv(0);
        info.os_ymm_enabled = ((xcr0 & 0x6) == 0x6);
    }

    if (max_leaf >= 7) {
        mc_i_cpuid(7, 0, r);
        info.has_avx2 = (r[1] >> 5) & 1;
    }

    return info;
}

/**
 * @brief One-shot gate: 1 iff AVX2 + FMA are safe to use right now.
 *
 * Checks CPU exposes AVX, AVX2, FMA, OSXSAVE, and the OS preserves YMM
 * state across context switches. Calling any mc_*_ps function when this
 * returns 0 will raise SIGILL or silently corrupt results.
 */
static inline int mc_supports_avx2_fma(void) {
    mc_cpu_info_t i = mc_detect_cpu();
    return i.has_avx && i.has_avx2 && i.has_fma
        && i.has_osxsave && i.os_ymm_enabled;
}

/* =====================================================================
 * 2. FOUNDATION
 * ===================================================================== */

/** @brief |x| — clears the sign bit. */
static inline __m256 mc_abs_ps(__m256 x) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
}

static inline __m256 mc_min_ps(__m256 a, __m256 b) { return _mm256_min_ps(a, b); }
static inline __m256 mc_max_ps(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }

/** @brief clamp(x, lo, hi). */
static inline __m256 mc_clamp_ps(__m256 x, __m256 lo, __m256 hi) {
    return _mm256_min_ps(_mm256_max_ps(x, lo), hi);
}

/** @brief saturate(x) = clamp(x, 0, 1). */
static inline __m256 mc_saturate_ps(__m256 x) {
    return mc_clamp_ps(x, _mm256_setzero_ps(), _mm256_set1_ps(1.0f));
}

/** @brief Linear interpolation: a + t*(b - a), via one FMA. */
static inline __m256 mc_lerp_ps(__m256 a, __m256 b, __m256 t) {
    return _mm256_fmadd_ps(t, _mm256_sub_ps(b, a), a);
}

/** @brief Inverse lerp: (x - a) / (b - a). Undefined when a == b. */
static inline __m256 mc_inverse_lerp_ps(__m256 a, __m256 b, __m256 x) {
    return _mm256_div_ps(_mm256_sub_ps(x, a), _mm256_sub_ps(b, a));
}

/** @brief Step: 1.0 if x >= edge else 0.0. */
static inline __m256 mc_step_ps(__m256 edge, __m256 x) {
    __m256 mask = _mm256_cmp_ps(x, edge, _CMP_GE_OQ);
    return _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
}

/**
 * @brief Smoothstep: Hermite blend 3t^2 - 2t^3 with t = clamp((x-a)/(b-a), 0, 1).
 * Degenerate a==b returns 0 per lane.
 */
static inline __m256 mc_smoothstep_ps(__m256 a, __m256 b, __m256 x) {
    __m256 num = _mm256_sub_ps(x, a);
    __m256 den = _mm256_sub_ps(b, a);
    __m256 is_zero  = _mm256_cmp_ps(den, _mm256_setzero_ps(), _CMP_EQ_OQ);
    __m256 safe_den = _mm256_blendv_ps(den, _mm256_set1_ps(1.0f), is_zero);
    __m256 t = _mm256_div_ps(num, safe_den);
    t = _mm256_max_ps(_mm256_setzero_ps(),
                      _mm256_min_ps(t, _mm256_set1_ps(1.0f)));
    __m256 poly = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), t, _mm256_set1_ps(3.0f));
    __m256 result = _mm256_mul_ps(t, _mm256_mul_ps(t, poly));
    return _mm256_andnot_ps(is_zero, result);
}

/** @brief Sign: +1 if x>0, -1 if x<0, 0 if x==0. */
static inline __m256 mc_sign_ps(__m256 x) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps( 1.0f);
    const __m256 neg  = _mm256_set1_ps(-1.0f);
    __m256 pos_mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    __m256 neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    __m256 res = _mm256_blendv_ps(zero, one, pos_mask);
    return _mm256_blendv_ps(res, neg, neg_mask);
}

/* ---------- Rounding family ---------- */

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

/** @brief trunc(x) — round toward zero (chop the fractional part). */
static inline __m256 mc_trunc_ps(__m256 x) {
    return _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

/**
 * @brief Fractional part: x - floor(x), result in [0, 1) for finite inputs.
 *
 * GLSL-style: fract(-0.3) = 0.7, not -0.3. If you want the truncated form
 * (sign-preserving), use `x - mc_trunc_ps(x)`.
 */
static inline __m256 mc_fract_ps(__m256 x) {
    return _mm256_sub_ps(x, mc_floor_ps(x));
}

/* ---------- Modulo, wrap, remap ---------- */

/**
 * @brief GLSL-style float modulo: x - y * floor(x / y).
 *
 * Result has the same sign as @p y, unlike C's fmod (which truncates).
 * Examples (y=10):  mod(13, 10) = 3,  mod(-1, 10) = 9.
 * Undefined when y == 0.
 */
static inline __m256 mc_mod_ps(__m256 x, __m256 y) {
    __m256 q = _mm256_div_ps(x, y);
    return _mm256_fnmadd_ps(y, mc_floor_ps(q), x);   /* x - y*floor(x/y) */
}

/**
 * @brief Wrap x into the half-open interval [lo, hi).
 *
 * Common uses: wrap world coords for tiling, wrap UV for texture repeat,
 * wrap accumulated angles. Undefined when hi == lo.
 */
static inline __m256 mc_wrap_ps(__m256 x, __m256 lo, __m256 hi) {
    __m256 range = _mm256_sub_ps(hi, lo);
    return _mm256_add_ps(lo, mc_mod_ps(_mm256_sub_ps(x, lo), range));
}

/**
 * @brief Linear remap: x in [a0, a1] maps to result in [b0, b1].
 *
 * Equivalent to `lerp(b0, b1, inverse_lerp(a0, a1, x))` but computed as
 * a single FMA chain to halve the rounding error. Out-of-range x
 * extrapolates linearly. Undefined when a0 == a1.
 */
static inline __m256 mc_remap_ps(__m256 x,
                                 __m256 a0, __m256 a1,
                                 __m256 b0, __m256 b1)
{
    __m256 t = _mm256_div_ps(_mm256_sub_ps(x,  a0),
                             _mm256_sub_ps(a1, a0));
    return _mm256_fmadd_ps(t, _mm256_sub_ps(b1, b0), b0);
}

/* ---------- Sign and selection helpers ---------- */

/**
 * @brief Copy sign from @p sign onto magnitude of @p mag.
 *
 * Bitwise: takes |mag| (clear sign bit) and ORs with the sign bit of
 * @p sign. Result has the magnitude of @p mag and the sign of @p sign.
 * Faster than the `copysignf` libc wrapper and SIMD-friendly.
 */
static inline __m256 mc_copysign_ps(__m256 mag, __m256 sign) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);          /* 0x80000000 */
    __m256 abs_mag  = _mm256_andnot_ps(sign_mask, mag);       /* clear sign */
    __m256 sign_of  = _mm256_and_ps(sign_mask, sign);         /* keep sign  */
    return _mm256_or_ps(abs_mag, sign_of);
}

/**
 * @brief Per-lane select: returns @p on_true where mask lanes are true (top
 *        bit set), @p on_false elsewhere.
 *
 * Wrapper over `_mm256_blendv_ps` with the argument order flipped so the
 * call site reads naturally — `mc_select_ps(cond, then, else)` instead of
 * `blendv(else, then, cond)`. Compiles to a single VBLENDVPS.
 */
static inline __m256 mc_select_ps(__m256 mask, __m256 on_true, __m256 on_false) {
    return _mm256_blendv_ps(on_false, on_true, mask);
}

/* ---------- Smootherstep (C2-continuous Hermite) ---------- */

/**
 * @brief Ken Perlin's smootherstep: 6t^5 - 15t^4 + 10t^3 on the clamped t.
 *
 * Same shape as smoothstep but with zero first AND second derivatives at
 * both edges, removing the slope discontinuity visible in smoothstep when
 * you stack ramps. Use when continuity of acceleration matters — animation
 * easing, noise blending, terrain LOD transitions.
 *
 * Degenerate a == b returns 0 for all lanes (same convention as smoothstep).
 */
static inline __m256 mc_smootherstep_ps(__m256 a, __m256 b, __m256 x) {
    __m256 num = _mm256_sub_ps(x, a);
    __m256 den = _mm256_sub_ps(b, a);
    __m256 is_zero  = _mm256_cmp_ps(den, _mm256_setzero_ps(), _CMP_EQ_OQ);
    __m256 safe_den = _mm256_blendv_ps(den, _mm256_set1_ps(1.0f), is_zero);
    __m256 t = _mm256_div_ps(num, safe_den);
    t = _mm256_max_ps(_mm256_setzero_ps(),
                      _mm256_min_ps(t, _mm256_set1_ps(1.0f)));

    /* Evaluate t^3 * (t*(t*6 - 15) + 10) via two FMAs. */
    __m256 p = _mm256_fmsub_ps(t, _mm256_set1_ps(6.0f),  _mm256_set1_ps(15.0f));
    p        = _mm256_fmadd_ps(t, p,                      _mm256_set1_ps(10.0f));
    __m256 t3 = _mm256_mul_ps(t, _mm256_mul_ps(t, t));
    __m256 result = _mm256_mul_ps(t3, p);

    return _mm256_andnot_ps(is_zero, result);
}

/* ---------- Pingpong and angle wrap ---------- */

/**
 * @brief Bounce x within [0, length] — `0,1,2,3,2,1,0,1,2,3,2,...` for length=3.
 *
 * Useful for looping animations that play forward then backward, oscillating
 * UV scrolls, and other "yo-yo" patterns. Derivation:
 *   t = mod(x, 2*length);  result = length - |t - length|
 * Symmetric for negative x. Undefined when length <= 0.
 */
static inline __m256 mc_pingpong_ps(__m256 x, __m256 length) {
    __m256 two_len = _mm256_add_ps(length, length);
    __m256 t = mc_mod_ps(x, two_len);
    return _mm256_sub_ps(length, mc_abs_ps(_mm256_sub_ps(t, length)));
}

/**
 * @brief Fold an angle into [-pi, pi). Common after integrating rotations.
 *
 * Exact value of +pi wraps to -pi (half-open convention from mc_wrap_ps).
 */
static inline __m256 mc_wrap_angle_ps(__m256 angle) {
    const __m256 neg_pi = _mm256_set1_ps(-3.14159265358979323846f);
    const __m256 pos_pi = _mm256_set1_ps( 3.14159265358979323846f);
    return mc_wrap_ps(angle, neg_pi, pos_pi);
}

/* ---------- Degree / radian conversion ---------- */

/** @brief Degrees -> radians (multiply by pi/180). */
static inline __m256 mc_to_radians_ps(__m256 deg) {
    return _mm256_mul_ps(deg, _mm256_set1_ps(0.01745329251994329577f));
}

/** @brief Radians -> degrees (multiply by 180/pi). */
static inline __m256 mc_to_degrees_ps(__m256 rad) {
    return _mm256_mul_ps(rad, _mm256_set1_ps(57.29577951308232088f));
}

/* ---------- Classification masks ----------
 *
 * Each returns an all-1s lane mask where the predicate holds, all-0s
 * elsewhere. Combine with mc_select_ps or use as a blendv mask directly.
 */

/** @brief Per-lane NaN test. NaN is the only value not equal to itself. */
static inline __m256 mc_is_nan_ps(__m256 x) {
    return _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
}

/** @brief Per-lane +/-inf test. */
static inline __m256 mc_is_inf_ps(__m256 x) {
    const __m256 inf = _mm256_set1_ps(1.0f / 0.0f);
    return _mm256_cmp_ps(mc_abs_ps(x), inf, _CMP_EQ_OQ);
}

/**
 * @brief Per-lane finite test (not NaN and not +/-inf).
 *
 * Implemented as `|x| < +inf` with ordered compare; NaN fails ordered
 * comparisons, so the single test catches both cases.
 */
static inline __m256 mc_is_finite_ps(__m256 x) {
    const __m256 inf = _mm256_set1_ps(1.0f / 0.0f);
    return _mm256_cmp_ps(mc_abs_ps(x), inf, _CMP_LT_OQ);
}

/**
 * @brief Absolute-tolerance fuzzy equality mask: |a - b| <= eps.
 *
 * Ordered compare, so NaN inputs always produce false (correct).
 * For relative tolerance, scale @p eps by max(|a|, |b|) at the call site.
 */
static inline __m256 mc_approx_equal_ps(__m256 a, __m256 b, __m256 eps) {
    __m256 diff = mc_abs_ps(_mm256_sub_ps(a, b));
    return _mm256_cmp_ps(diff, eps, _CMP_LE_OQ);
}

/* =====================================================================
 * 3. POLYNOMIAL EVALUATORS (Estrin's scheme)
 * =====================================================================
 * Estrin's scheme parallelizes the dependency chain of polynomial
 * evaluation, exposing more ILP than Horner. Each function takes the
 * polynomial variable directly — for sin/cos/etc. the caller passes
 * x^2 since the polynomial variable is t = x^2.
 *
 * Coefficient layout: c[0] = constant term, c[N] = highest-degree term.
 */

/** @brief P(x) = c[0] + c[1]x + c[2]x^2 + c[3]x^3. */
static inline __m256 mc_poly_estrin_deg3_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);
    return _mm256_fmadd_ps(y1, x2, y0);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[4]x^4. */
static inline __m256 mc_poly_estrin_deg4_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);
    __m256 c4 = _mm256_set1_ps(c[4]);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);
    return _mm256_fmadd_ps(c4, x4, z0);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[5]x^5. */
static inline __m256 mc_poly_estrin_deg5_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);
    __m256 c4 = _mm256_set1_ps(c[4]);
    __m256 c5 = _mm256_set1_ps(c[5]);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);
    __m256 y2 = _mm256_fmadd_ps(c5, x, c4);
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);
    return _mm256_fmadd_ps(y2, x4, z0);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[6]x^6. */
static inline __m256 mc_poly_estrin_deg6_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);
    __m256 c4 = _mm256_set1_ps(c[4]);
    __m256 c5 = _mm256_set1_ps(c[5]);
    __m256 c6 = _mm256_set1_ps(c[6]);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);
    __m256 y2 = _mm256_fmadd_ps(c5, x, c4);
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);
    __m256 z1 = _mm256_fmadd_ps(c6, x2, y2);
    return _mm256_fmadd_ps(z1, x4, z0);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[8]x^8. */
static inline __m256 mc_poly_estrin_deg8_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);
    __m256 c4 = _mm256_set1_ps(c[4]);
    __m256 c5 = _mm256_set1_ps(c[5]);
    __m256 c6 = _mm256_set1_ps(c[6]);
    __m256 c7 = _mm256_set1_ps(c[7]);
    __m256 c8 = _mm256_set1_ps(c[8]);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 x8 = _mm256_mul_ps(x4, x4);
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);
    __m256 y2 = _mm256_fmadd_ps(c5, x, c4);
    __m256 y3 = _mm256_fmadd_ps(c7, x, c6);
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);
    __m256 z1 = _mm256_fmadd_ps(y3, x2, y2);
    __m256 w0 = _mm256_fmadd_ps(z1, x4, z0);
    return _mm256_fmadd_ps(c8, x8, w0);
}

/* =====================================================================
 * 4. MINIMAX COEFFICIENT TABLES
 * =====================================================================
 * Float-precision minimax polynomials generated with Sollya 8.0, prec=128.
 * Each table covers a reduced range; argument reduction is performed in
 * the caller. The header comment above each table records the valid input
 * range, the form of the polynomial, and the reconstruction step.
 */

/* sin(x) on [-pi/4, pi/4],  sin(x) = x * P(x^2), deg-3 in t=x^2, ~0.04 ULP fit. */
static const float MC_SIN_COEFFS[4] = {
     0.999999996926342778282546052871901425708f,
    -0.166666506992021714609082680346419283499f,
     8.3320368751463668594189710376374380334e-3f,
    -1.95040219991569957621701412087918905123e-4f
};

/* cos(x) on [0, pi/4],  cos(x) = 1 - (x^2/2) * P(x^2), deg-3, ~0.003 ULP. */
static const float MC_COS_COEFFS[4] = {
     0.999999999383828582129270698455438593398f,
    -8.33333013311138576100588974476052245317e-2f,
     2.77751801301879159485208760118056498869e-3f,
    -4.89277731664569609684403388950264514135e-5f
};

/* tan(x) on [0, pi/8],  tan(x) = x + x^3 * P(x^2), deg-3, ~0.047 ULP.
 * Double-angle in caller covers up to pi/4. */
static const float MC_TAN_COEFFS[4] = {
    0.333333287350697673210307974070168833443f,
    0.133342724106033723928251509259696377785f,
    5.36695340958928597698376503512930466572e-2f,
    2.48696209035983955819911447913718818547e-2f
};

/* asin(x) on [0, 0.5],  asin(x) = x * P(x^2), deg-5, ~0.038 ULP.
 * For |x|>0.5: asin(x) = pi/2 - 2*asin(sqrt((1-x)/2)).
 * acos uses asin via reconstruction (no separate poly). */
static const float MC_ASIN_COEFFS[6] = {
    0.999999995478096608701846419443927593772f,
    0.166667922213005814888055819423805727124f,
    7.4943675002161483595064288611545911554e-2f,
    4.55576929736023952506899394496118912138e-2f,
    2.3823936588797289217393470823864621618e-2f,
    4.26903047154410196086038580757107848077e-2f
};

/* atan(x) on [0, tan(pi/8)] (~0.4142),  atan(x) = x * P(x^2), deg-4, ~0.12 ULP. */
static const float MC_ATAN_COEFFS[5] = {
     0.999999982460717905120588625597135589113f,
    -0.333328079017683915803156014462307278338f,
     0.199747243062319873757635773939061055264f,
    -0.138544576850887346340673529423546659512f,
     7.99364423130897184747222910912351542317e-2f
};

/* exp(x): r in [-ln2/2, ln2/2] after Cody-Waite by ln2, deg-6, ~0.031 ULP. */
static const float MC_EXP_COEFFS[7] = {
    0.999999999919067037197896619810258382586f,
    1.00000003772604441761669504931498793344f,
    0.500000016841291732193530788373548378825f,
    0.16666415481911346468144430935857029765f,
    4.16660835948060307131369604131088746096e-2f,
    8.3751285790288349279387284883017034716e-3f,
    1.39560559247713612085306234329346882015e-3f
};

/** @brief High part of ln(2) for Cody-Waite reduction in exp/ln. */
#define MC_LN2_HI  (6.93147180369123816490e-1f)
/** @brief Low part of ln(2) for Cody-Waite reduction in exp/ln. */
#define MC_LN2_LO  (1.90821492927058770002e-10f)

/* exp2(x): r in [-0.5, 0.5] after integer-round reduction, deg-6, ~0.031 ULP. */
static const float MC_EXP2_COEFFS[7] = {
    0.999999999919067037197896619810258382586f,
    0.693147206709646631167508230321444285403f,
    0.24022651505055008334162029128006404866f,
    5.55032721576643256179871599540553593548e-2f,
    9.6179945141780462397505203069232532594e-3f,
    1.34004316670754714578668090114651575682e-3f,
    1.54780222758443390748352500253494123394e-4f
};

/* ln(x): u = m-1 with m in [1/sqrt(2), sqrt(2)], ln(1+u) = u*P(u), deg-8, ~0.18 ULP. */
static const float MC_LN_COEFFS[9] = {
     0.999999975045684318538782514084829989036f,
    -0.499999777407575512300036757431471630087f,
     0.33334160454165577395122435673327927243f,
    -0.250029461445376526033833784745386161086f,
     0.1995841718412106778508522363336532356576f,
    -0.165437371026277935081140961987902041066f,
     0.1491737592987093474996556991856621741334f,
    -0.144821909765600057643339822163783003331f,
     8.9599230573989110379767056226651046356e-2f
};

/* sinh(x) on [0, ln2/2],  sinh(x) = x * P(x^2), deg-3, ~2.6e-5 ULP fit. */
static const float MC_SINH_COEFFS[4] = {
    0.999999999995506589113030978465441728845f,
    0.166666667863453360815279663872642870366f,
    8.3332835305493575391661845707539300956e-3f,
    1.99075738395020160195941395333857184758e-4f
};

/* cosh(x) on [0, ln2/2],  cosh(x) = P(x^2), deg-3 even, ~6.8e-4 ULP fit. */
static const float MC_COSH_COEFFS[4] = {
    0.999999999959534772527072194807811326055f,
    0.500000010776960664452566756637566496614f,
    4.16662182295045307372470182575150115655e-2f,
    1.39485833100088545140588669801341803883e-3f
};

/* tanh(x) on [0, ln2/2],  tanh(x) = x * P(x^2), deg-4, ~2.2e-3 ULP fit. */
static const float MC_TANH_COEFFS[5] = {
     0.99999999962493826654063440501532327137f,
    -0.333333175699346402857353793233040621278f,
     0.133322720162943708147471884818690998073f,
    -5.37170264940897791481321780372525368095e-2f,
     1.94110016222003854929537112793490501015e-2f
};

/* =====================================================================
 * 5. SQUARE ROOT AND RECIPROCAL SQUARE ROOT
 * ===================================================================== */

/**
 * @brief sqrt(x). Maps to VSQRTPS; IEEE 754 correctly-rounded.
 * Negative lanes produce NaN.
 */
static inline __m256 mc_sqrt_ps(__m256 x) { return _mm256_sqrt_ps(x); }

/**
 * @brief Fast 1/sqrt(x) — hardware seed only, ~11 bits. Particles, LODs.
 * x must be > 0.
 */
static inline __m256 mc_rsqrt_fast_ps(__m256 x) { return _mm256_rsqrt_ps(x); }

/**
 * @brief 1/sqrt(x) with one Newton-Raphson refinement — ~23 bits / ~0.5 ULP.
 * Standard physics / lighting / normal-map precision.
 */
static inline __m256 mc_rsqrt_ps(__m256 x) {
    __m256 y  = _mm256_rsqrt_ps(x);
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 hx = _mm256_mul_ps(x, y2);
    __m256 t  = _mm256_fnmadd_ps(hx, _mm256_set1_ps(0.5f), _mm256_set1_ps(1.5f));
    return _mm256_mul_ps(y, t);
}

/**
 * @brief 1/sqrt(x) with two NR iterations — ~24 bits (full mantissa).
 * Use when the last ULP matters.
 */
static inline __m256 mc_rsqrt_precise_ps(__m256 x) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ofh  = _mm256_set1_ps(1.5f);
    __m256 y = _mm256_rsqrt_ps(x);
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 t  = _mm256_fnmadd_ps(_mm256_mul_ps(x, y2), half, ofh);
    y = _mm256_mul_ps(y, t);
    y2 = _mm256_mul_ps(y, y);
    t  = _mm256_fnmadd_ps(_mm256_mul_ps(x, y2), half, ofh);
    return _mm256_mul_ps(y, t);
}

/**
 * @brief SoA 3D normalize (1x NR precision). 8 vectors at once.
 * Zero-length vectors produce inf/NaN; guard at call site if needed.
 */
static inline void mc_normalize_soa_ps(__m256* x, __m256* y, __m256* z) {
    __m256 x2  = _mm256_mul_ps(*x, *x);
    __m256 y2  = _mm256_mul_ps(*y, *y);
    __m256 z2  = _mm256_mul_ps(*z, *z);
    __m256 sum = _mm256_add_ps(x2, _mm256_add_ps(y2, z2));
    __m256 inv = mc_rsqrt_ps(sum);
    *x = _mm256_mul_ps(*x, inv);
    *y = _mm256_mul_ps(*y, inv);
    *z = _mm256_mul_ps(*z, inv);
}

/** @brief SoA 3D normalize, fast tier (~12 bits). For particles / billboards. */
static inline void mc_normalize_soa_fast_ps(__m256* x, __m256* y, __m256* z) {
    __m256 x2  = _mm256_mul_ps(*x, *x);
    __m256 y2  = _mm256_mul_ps(*y, *y);
    __m256 z2  = _mm256_mul_ps(*z, *z);
    __m256 sum = _mm256_add_ps(x2, _mm256_add_ps(y2, z2));
    __m256 inv = mc_rsqrt_fast_ps(sum);
    *x = _mm256_mul_ps(*x, inv);
    *y = _mm256_mul_ps(*y, inv);
    *z = _mm256_mul_ps(*z, inv);
}

/* =====================================================================
 * 6. EXP, EXP2, LN
 * ===================================================================== */

/**
 * @brief Assemble 2^n as float by writing n into the exponent field.
 * Clamps n to [-126, 127] to stay in normal-float range.
 */
static inline __m256 mc_i_pow2i_ps(__m256i n) {
    n = _mm256_max_epi32(n, _mm256_set1_epi32(-126));
    n = _mm256_min_epi32(n, _mm256_set1_epi32( 127));
    __m256i bits = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    bits = _mm256_slli_epi32(bits, 23);
    return _mm256_castsi256_ps(bits);
}

/**
 * @brief e^x. ~1-5 ULP. Saturates: x < -87.3 -> 0, x > 88.7 -> FLT_MAX.
 *
 * Algorithm: clamp; Cody-Waite reduce by ln2 to r in [-ln2/2, ln2/2];
 * P(r) via Estrin + MC_EXP_COEFFS; scale by 2^n.
 */
static inline __m256 mc_exp_ps(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3365386f)),
                                       _mm256_set1_ps( 88.7228395f));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896340736f);
    __m256 fn = _mm256_round_ps(_mm256_mul_ps(x, log2e),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(MC_LN2_HI), x);
    r        = _mm256_fnmadd_ps(fn, _mm256_set1_ps(MC_LN2_LO), r);
    __m256 p = mc_poly_estrin_deg6_ps(r, MC_EXP_COEFFS);
    __m256i n = _mm256_cvtps_epi32(fn);
    return _mm256_mul_ps(p, mc_i_pow2i_ps(n));
}

/** @brief 2^x. ~1-2 ULP. */
static inline __m256 mc_exp2_ps(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-126.0f)),
                                       _mm256_set1_ps( 127.0f));
    __m256 fn = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r  = _mm256_sub_ps(x, fn);
    __m256 p  = mc_poly_estrin_deg6_ps(r, MC_EXP2_COEFFS);
    __m256i n = _mm256_cvtps_epi32(fn);
    return _mm256_mul_ps(p, mc_i_pow2i_ps(n));
}

/**
 * @brief ln(x). ~1-3 ULP.
 *
 * x <= 0 returns -inf per lane (no separate NaN for x<0; mask at call site
 * if you need strict IEEE semantics).
 */
static inline __m256 mc_ln_ps(__m256 x) {
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);
    x = _mm256_blendv_ps(x, _mm256_set1_ps(1.0f), invalid_mask);

    __m256i xi = _mm256_castps_si256(x);
    __m256i exp_biased = _mm256_srli_epi32(xi, 23);
    __m256i n_i = _mm256_sub_epi32(exp_biased, _mm256_set1_epi32(127));

    __m256i mant_bits = _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF));
    mant_bits = _mm256_or_si256(mant_bits, _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_castsi256_ps(mant_bits);    /* m in [1, 2) */

    /* Normalize m into [1/sqrt(2), sqrt(2)]: if m > sqrt(2), halve and bump n. */
    __m256 need_adjust = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356237309504880f),
                                       _CMP_GT_OQ);
    __m256i adjust = _mm256_srli_epi32(_mm256_castps_si256(need_adjust), 31);
    n_i = _mm256_add_epi32(n_i, adjust);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), need_adjust);

    __m256 u = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    __m256 p = mc_poly_estrin_deg8_ps(u, MC_LN_COEFFS);
    __m256 ln_m = _mm256_mul_ps(u, p);

    __m256 fn = _mm256_cvtepi32_ps(n_i);
    __m256 result = _mm256_fmadd_ps(fn, _mm256_set1_ps(MC_LN2_HI), ln_m);
    result        = _mm256_fmadd_ps(fn, _mm256_set1_ps(MC_LN2_LO), result);

    const __m256 neg_inf = _mm256_set1_ps(-1.0f / 0.0f);
    return _mm256_blendv_ps(result, neg_inf, invalid_mask);
}

/* =====================================================================
 * 7. SIN, COS, SINCOS
 * =====================================================================
 * Two-part Cody-Waite reduction by pi/2. ~1-3 ULP for |x| <= 10^5;
 * gracefully degrades beyond that (use Payne-Hanek if you need huge
 * angles correctly).
 */

#define MC_TWO_OVER_PI  (0.636619772367581343f)
#define MC_PIO2_HI      ( 1.5707963705062866f)
#define MC_PIO2_LO      (-4.3711388286737929e-8f)

/** @brief Compute sin(x) and cos(x) together. */
static inline void mc_sincos_ps(__m256 x, __m256* s_out, __m256* c_out) {
    __m256 fn = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(MC_TWO_OVER_PI)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(MC_PIO2_HI), x);
    r        = _mm256_fnmadd_ps(fn, _mm256_set1_ps(MC_PIO2_LO), r);

    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 sp = mc_poly_estrin_deg3_ps(r2, MC_SIN_COEFFS);
    __m256 sin_r = _mm256_mul_ps(r, sp);

    __m256 cp = mc_poly_estrin_deg3_ps(r2, MC_COS_COEFFS);
    __m256 half_r2 = _mm256_mul_ps(r2, _mm256_set1_ps(0.5f));
    __m256 cos_r = _mm256_fnmadd_ps(half_r2, cp, _mm256_set1_ps(1.0f));

    __m256i ni = _mm256_cvtps_epi32(fn);
    __m256i v_one = _mm256_set1_epi32(1);
    __m256i v_two = _mm256_set1_epi32(2);

    __m256 swap_mask = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(ni, v_one), v_one));
    __m256 s_pre = _mm256_blendv_ps(sin_r, cos_r, swap_mask);
    __m256 c_pre = _mm256_blendv_ps(cos_r, sin_r, swap_mask);

    __m256i sin_sign = _mm256_slli_epi32(_mm256_and_si256(ni, v_two), 30);
    __m256i cos_sign = _mm256_slli_epi32(
        _mm256_and_si256(_mm256_add_epi32(ni, v_one), v_two), 30);

    /* Use unaligned stores: MSVC/MinGW under the Windows x64 ABI don't
     * guarantee 32-byte stack alignment, so an implicit `*s_out = ...`
     * (which compiles to vmovaps) can fault when callers hold the __m256
     * in an underaligned stack slot. vmovups is one cycle slower at most
     * and has no alignment requirement. */
    _mm256_storeu_ps((float*)s_out, _mm256_xor_ps(s_pre, _mm256_castsi256_ps(sin_sign)));
    _mm256_storeu_ps((float*)c_out, _mm256_xor_ps(c_pre, _mm256_castsi256_ps(cos_sign)));
}

/** @brief sin(x). */
static inline __m256 mc_sin_ps(__m256 x) {
    __m256 s, c; mc_sincos_ps(x, &s, &c); return s;
}

/** @brief cos(x). */
static inline __m256 mc_cos_ps(__m256 x) {
    __m256 s, c; mc_sincos_ps(x, &s, &c); return c;
}

/* =====================================================================
 * 8. TAN
 * =====================================================================
 * pi/2 reduction (sharing constants with sincos), then halve-and-double-angle
 * so the polynomial only ever sees |y| <= pi/8.
 * For odd quadrants we want -cot(r); implemented as a num/den swap so the
 * whole function uses a single division.
 */

static inline __m256 mc_tan_ps(__m256 x) {
    __m256 fn = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(MC_TWO_OVER_PI)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(MC_PIO2_HI), x);
    r        = _mm256_fnmadd_ps(fn, _mm256_set1_ps(MC_PIO2_LO), r);

    __m256 y  = _mm256_mul_ps(r, _mm256_set1_ps(0.5f));
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 p  = mc_poly_estrin_deg3_ps(y2, MC_TAN_COEFFS);
    __m256 y3 = _mm256_mul_ps(y2, y);
    __m256 t  = _mm256_fmadd_ps(y3, p, y);

    __m256 t2    = _mm256_mul_ps(t, t);
    __m256 two_t = _mm256_add_ps(t, t);
    __m256 one_minus_t2 = _mm256_sub_ps(_mm256_set1_ps(1.0f), t2);
    __m256 t2_minus_one = _mm256_sub_ps(t2, _mm256_set1_ps(1.0f));

    __m256i ni    = _mm256_cvtps_epi32(fn);
    __m256i v_one = _mm256_set1_epi32(1);
    __m256  odd_mask = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(ni, v_one), v_one));

    __m256 num = _mm256_blendv_ps(two_t,        t2_minus_one, odd_mask);
    __m256 den = _mm256_blendv_ps(one_minus_t2, two_t,        odd_mask);
    return _mm256_div_ps(num, den);
}

/* =====================================================================
 * 9. ASIN, ACOS
 * =====================================================================
 * Sqrt-based reduction near +/-1 (instead of pi/2 - asin) to avoid the
 * cancellation that would lose precision there. Inputs saturated to
 * [-1, 1] so slight overshoot from dot products doesn't produce NaN.
 */

static inline __m256 mc_asin_ps(__m256 x) {
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 pio2 = _mm256_set1_ps(1.57079632679489661923f);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    __m256 sign_x = _mm256_and_ps(x, sign_bit);
    __m256 ax = mc_abs_ps(x);
    ax = _mm256_min_ps(ax, one);                  /* saturate to [0,1] */

    __m256 big = _mm256_cmp_ps(ax, half, _CMP_GT_OQ);

    __m256 t_small = _mm256_mul_ps(ax, ax);
    __m256 t_big   = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    __m256 t = _mm256_blendv_ps(t_small, t_big, big);

    __m256 v = _mm256_blendv_ps(ax, _mm256_sqrt_ps(t_big), big);

    __m256 p   = mc_poly_estrin_deg5_ps(t, MC_ASIN_COEFFS);
    __m256 pre = _mm256_mul_ps(v, p);

    __m256 res_big = _mm256_sub_ps(pio2, _mm256_add_ps(pre, pre));
    __m256 res     = _mm256_blendv_ps(pre, res_big, big);

    return _mm256_xor_ps(res, sign_x);
}

static inline __m256 mc_acos_ps(__m256 x) {
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 pio2 = _mm256_set1_ps(1.57079632679489661923f);
    const __m256 pi   = _mm256_set1_ps(3.14159265358979323846f);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    /* Saturate to [-1, 1]. */
    x = _mm256_max_ps(_mm256_min_ps(x, one), _mm256_sub_ps(zero, one));
    __m256 ax    = mc_abs_ps(x);
    __m256 big   = _mm256_cmp_ps(ax, half, _CMP_GT_OQ);
    __m256 x_neg = _mm256_cmp_ps(x,  zero, _CMP_LT_OQ);

    __m256 t_small = _mm256_mul_ps(x, x);
    __m256 t_big   = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    __m256 t = _mm256_blendv_ps(t_small, t_big, big);
    __m256 v = _mm256_blendv_ps(x, _mm256_sqrt_ps(t_big), big);

    __m256 p   = mc_poly_estrin_deg5_ps(t, MC_ASIN_COEFFS);
    __m256 pre = _mm256_mul_ps(v, p);

    __m256 res_small = _mm256_sub_ps(pio2, pre);

    __m256 two_pre = _mm256_add_ps(pre, pre);
    __m256 signed_two_pre = _mm256_xor_ps(two_pre, _mm256_and_ps(x_neg, sign_bit));
    __m256 offset_pi = _mm256_and_ps(x_neg, pi);
    __m256 res_big = _mm256_add_ps(offset_pi, signed_two_pre);

    return _mm256_blendv_ps(res_small, res_big, big);
}

/* =====================================================================
 * 10. ATAN, ATAN2
 * =====================================================================
 * Two-stage reduction: fold |x|>1 via 1/x, then fold residual >tan(pi/8)
 * via (x-1)/(x+1). Polynomial then sees v in [-tan(pi/8), tan(pi/8)].
 */

static inline __m256 mc_atan_ps(__m256 x) {
    const __m256 one      = _mm256_set1_ps(1.0f);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);

    __m256 sign_x = _mm256_and_ps(x, sign_bit);
    __m256 ax = mc_abs_ps(x);

    __m256 swap_mask = _mm256_cmp_ps(ax, one, _CMP_GT_OQ);
    __m256 inv_ax    = _mm256_div_ps(one, ax);
    __m256 u         = _mm256_blendv_ps(ax, inv_ax, swap_mask);

    const __m256 tan_pio8 = _mm256_set1_ps(0.41421356237309504880f);
    __m256 shift_mask = _mm256_cmp_ps(u, tan_pio8, _CMP_GT_OQ);
    __m256 v_red = _mm256_div_ps(_mm256_sub_ps(u, one), _mm256_add_ps(u, one));
    __m256 v = _mm256_blendv_ps(u, v_red, shift_mask);

    __m256 v2 = _mm256_mul_ps(v, v);
    __m256 p  = mc_poly_estrin_deg4_ps(v2, MC_ATAN_COEFFS);
    __m256 atan_v = _mm256_mul_ps(v, p);

    const __m256 pio4 = _mm256_set1_ps(0.78539816339744830962f);
    __m256 atan_u = _mm256_add_ps(atan_v, _mm256_and_ps(shift_mask, pio4));

    const __m256 pio2 = _mm256_set1_ps(1.57079632679489661923f);
    __m256 atan_ax_swap = _mm256_sub_ps(pio2, atan_u);
    __m256 atan_ax = _mm256_blendv_ps(atan_u, atan_ax_swap, swap_mask);

    return _mm256_xor_ps(atan_ax, sign_x);
}

/**
 * @brief atan2(y, x). (0, 0) returns NaN — mask at call site if needed.
 */
static inline __m256 mc_atan2_ps(__m256 y, __m256 x) {
    const __m256 zero     = _mm256_setzero_ps();
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 pi       = _mm256_set1_ps(3.14159265358979323846f);

    __m256 ratio  = _mm256_div_ps(y, x);
    __m256 atan_r = mc_atan_ps(ratio);

    __m256 x_neg     = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    __m256 y_sign    = _mm256_and_ps(y, sign_bit);
    __m256 signed_pi = _mm256_xor_ps(pi, y_sign);
    __m256 offset    = _mm256_and_ps(x_neg, signed_pi);

    return _mm256_add_ps(atan_r, offset);
}

/* =====================================================================
 * 11. SINH, COSH, TANH
 * =====================================================================
 * Polynomial for small |x| (|x| <= ln2/2), exp-identity for the rest.
 * Both branches always evaluate; blended per lane.
 */

#define MC_LN2_OVER_2  (0.34657359027997265471f)

static inline __m256 mc_sinh_ps(__m256 x) {
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 bound = _mm256_set1_ps(MC_LN2_OVER_2);

    __m256 ax  = mc_abs_ps(x);
    __m256 big = _mm256_cmp_ps(ax, bound, _CMP_GT_OQ);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 p_small = mc_poly_estrin_deg3_ps(x2, MC_SINH_COEFFS);
    __m256 sinh_small = _mm256_mul_ps(x, p_small);

    __m256 ex = mc_exp_ps(x);
    __m256 inv_ex = _mm256_div_ps(one, ex);
    __m256 sinh_big = _mm256_mul_ps(_mm256_sub_ps(ex, inv_ex), half);

    return _mm256_blendv_ps(sinh_small, sinh_big, big);
}

static inline __m256 mc_cosh_ps(__m256 x) {
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 bound = _mm256_set1_ps(MC_LN2_OVER_2);

    __m256 ax  = mc_abs_ps(x);
    __m256 big = _mm256_cmp_ps(ax, bound, _CMP_GT_OQ);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 cosh_small = mc_poly_estrin_deg3_ps(x2, MC_COSH_COEFFS);

    __m256 ex = mc_exp_ps(ax);
    __m256 inv_ex = _mm256_div_ps(one, ex);
    __m256 cosh_big = _mm256_mul_ps(_mm256_add_ps(ex, inv_ex), half);

    return _mm256_blendv_ps(cosh_small, cosh_big, big);
}

static inline __m256 mc_tanh_ps(__m256 x) {
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 two   = _mm256_set1_ps(2.0f);
    const __m256 bound = _mm256_set1_ps(MC_LN2_OVER_2);

    __m256 ax  = mc_abs_ps(x);
    __m256 big = _mm256_cmp_ps(ax, bound, _CMP_GT_OQ);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 p_small = mc_poly_estrin_deg4_ps(x2, MC_TANH_COEFFS);
    __m256 tanh_small = _mm256_mul_ps(x, p_small);

    __m256 two_x = _mm256_add_ps(x, x);
    __m256 e2x   = mc_exp_ps(two_x);
    __m256 denom = _mm256_add_ps(e2x, one);
    __m256 tanh_big = _mm256_sub_ps(one, _mm256_div_ps(two, denom));

    return _mm256_blendv_ps(tanh_small, tanh_big, big);
}

/* =====================================================================
 * 12. POW
 * =====================================================================
 * pow(x, y) = exp(y * ln(x)). Error scales with |y| (see notes).
 * x <= 0 produces non-finite/NaN; integer-exponent path for negative
 * bases is not provided here.
 */

static inline __m256 mc_pow_ps(__m256 x, __m256 y) {
    __m256 lx = mc_ln_ps(x);
    __m256 yl = _mm256_mul_ps(y, lx);
    return mc_exp_ps(yl);
}

#endif /* MC_SIMD_MATH_H */
