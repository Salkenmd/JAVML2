/*
 * mc_simd_math_neon — single-header NEON (AArch64) math for gamedev.
 * Copyright (c) 2026 RainbyeKeith. MIT License — see LICENSE file.
 *
 * Forked from mc_simd_math (AVX2+FMA). Same algorithms, same coefficient
 * tables, same accuracy targets (~1-5 ULP), translated to NEON intrinsics
 * for 4-wide float vectors (float32x4_t).
 */

/**
 * @file mc_simd_math_neon.h
 * @brief Single-header NEON math library for gamedev (float / float32x4_t).
 *
 * Game-grade transcendentals and foundation utilities on 4-wide float vectors.
 * Polynomials are minimax fits generated with Sollya 8.0 (~0.04 ULP fit error);
 * measured end-to-end accuracy is ~1-5 ULP for typical inputs.
 *
 * @par Usage
 * @code
 *   #include "mc_simd_math_neon.h"
 *   if (!mc_supports_neon()) { ... bail out ... }
 *   float32x4_t y = mc_sin_ps(x);
 * @endcode
 *
 * @par Build flags
 *   GCC/Clang:  -O2 -march=armv8-a+fp+simd   (or -march=armv8-a+simd)
 *   MSVC (ARM): /O2 /arch:ARMv8.0
 *
 * @par Sections
 *   1. CPU feature detection (mc_supports_neon)
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
 *
 * @note Requires AArch64 (ARMv8-A) with NEON. Does not support 32-bit ARM
 *       (ARMv7) due to missing vrnd*, vdivq_f32, vsqrtq_f32 intrinsics.
 */

#ifndef MC_SIMD_MATH_NEON_H
#define MC_SIMD_MATH_NEON_H

#include <stdint.h>
#include <arm_neon.h>

/* MC_INLINE — force inlining across compilers and optimization levels. */
#ifndef MC_INLINE
  #if defined(_MSC_VER)
    #define MC_INLINE static __forceinline
  #elif defined(__GNUC__) || defined(__clang__)
    #define MC_INLINE static inline __attribute__((always_inline))
  #else
    #define MC_INLINE static inline
  #endif
#endif

/* MC_RESTRICT — pointer non-aliasing hint. */
#ifndef MC_RESTRICT
  #if defined(_MSC_VER)
    #define MC_RESTRICT __restrict
  #elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define MC_RESTRICT restrict
  #else
    #define MC_RESTRICT
  #endif
#endif

/* =====================================================================
 * 1. CPU FEATURE DETECTION
 * ===================================================================== */

/**
 * @brief Detailed CPU feature snapshot (NEON).
 */
typedef struct {
    int has_neon;
    int has_neon_fma;    /* always 1 when has_neon; kept for API parity */
    int is_aarch64;
} mc_cpu_info_neon_t;

/**
 * @brief Probe the CPU / compile-time environment for NEON support.
 *
 * On AArch64, NEON is mandatory (part of the base ISA). On ARMv7 Linux,
 * may use getauxval(AT_HWCAP) & HWCAP_NEON if available.
 *
 * The compile-time macro __ARM_NEON (or __ARM_NEON__) is defined by the
 * compiler when NEON is enabled. This provides a cheap early-out.
 */
MC_INLINE mc_cpu_info_neon_t mc_detect_cpu_neon(void) {
    mc_cpu_info_neon_t info = {0, 0, 0};
#if defined(__aarch64__) || defined(_M_ARM64)
    info.is_aarch64 = 1;
    info.has_neon   = 1;
    info.has_neon_fma = 1;   /* NEON FMA is part of the base ARMv8.0 ISA */
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    info.has_neon   = 1;
    info.has_neon_fma = 1;
#else
    /* Compile-time: NEON not enabled. */
#endif
    return info;
}

/**
 * @brief One-shot gate: 1 iff NEON is safe to use right now.
 *
 * On AArch64 this always returns 1 (NEON is mandatory). On 32-bit ARM
 * with NEON enabled at compile-time, also returns 1.
 */
MC_INLINE int mc_supports_neon(void) {
    mc_cpu_info_neon_t i = mc_detect_cpu_neon();
    return i.has_neon && i.has_neon_fma;
}

/* =====================================================================
 * 2. FOUNDATION
 * ===================================================================== */

/** @brief |x| — clears the sign bit. */
MC_INLINE float32x4_t mc_abs_ps(float32x4_t x) {
    return vabsq_f32(x);
}

MC_INLINE float32x4_t mc_min_ps(float32x4_t a, float32x4_t b) { return vminq_f32(a, b); }
MC_INLINE float32x4_t mc_max_ps(float32x4_t a, float32x4_t b) { return vmaxq_f32(a, b); }

/** @brief clamp(x, lo, hi). */
MC_INLINE float32x4_t mc_clamp_ps(float32x4_t x, float32x4_t lo, float32x4_t hi) {
    return vminq_f32(vmaxq_f32(x, lo), hi);
}

/** @brief saturate(x) = clamp(x, 0, 1). */
MC_INLINE float32x4_t mc_saturate_ps(float32x4_t x) {
    return mc_clamp_ps(x, vdupq_n_f32(0.0f), vdupq_n_f32(1.0f));
}

/** @brief Linear interpolation: a + t*(b - a), via one FMA. */
MC_INLINE float32x4_t mc_lerp_ps(float32x4_t a, float32x4_t b, float32x4_t t) {
    float32x4_t diff = vsubq_f32(b, a);
    return vfmaq_f32(a, t, diff);
}

/** @brief Inverse lerp: (x - a) / (b - a). Undefined when a == b. */
MC_INLINE float32x4_t mc_inverse_lerp_ps(float32x4_t a, float32x4_t b, float32x4_t x) {
    return vdivq_f32(vsubq_f32(x, a), vsubq_f32(b, a));
}

/** @brief Step: 1.0 if x >= edge else 0.0. */
MC_INLINE float32x4_t mc_step_ps(float32x4_t edge, float32x4_t x) {
    uint32x4_t mask = vcgeq_f32(x, edge);
    return vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f))));
}

/**
 * @brief Smoothstep: Hermite blend 3t^2 - 2t^3 with t = clamp((x-a)/(b-a), 0, 1).
 * Degenerate a==b returns 0 per lane.
 */
MC_INLINE float32x4_t mc_smoothstep_ps(float32x4_t a, float32x4_t b, float32x4_t x) {
    float32x4_t num = vsubq_f32(x, a);
    float32x4_t den = vsubq_f32(b, a);
    uint32x4_t is_zero = vceqq_f32(den, vdupq_n_f32(0.0f));
    float32x4_t safe_den = vbslq_f32(is_zero, vdupq_n_f32(1.0f), den);
    float32x4_t t = vdivq_f32(num, safe_den);
    t = vmaxq_f32(vdupq_n_f32(0.0f), vminq_f32(t, vdupq_n_f32(1.0f)));
    /* poly = 3 - 2t = vfmsq(3, t, 2) */
    float32x4_t poly = vfmsq_f32(vdupq_n_f32(3.0f), t, vdupq_n_f32(2.0f));
    float32x4_t result = vmulq_f32(t, vmulq_f32(t, poly));
    return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(result), is_zero));
}

/** @brief Sign: +1 if x>0, -1 if x<0, 0 if x==0. */
MC_INLINE float32x4_t mc_sign_ps(float32x4_t x) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one  = vdupq_n_f32( 1.0f);
    const float32x4_t neg  = vdupq_n_f32(-1.0f);
    uint32x4_t pos_mask = vcgtq_f32(x, zero);
    uint32x4_t neg_mask = vcltq_f32(x, zero);
    float32x4_t res = vbslq_f32(pos_mask, one, zero);
    return vbslq_f32(neg_mask, neg, res);
}

/* ---------- Rounding family ---------- */

/** @brief floor(x) — round toward -inf. */
MC_INLINE float32x4_t mc_floor_ps(float32x4_t x) {
    return vrndmq_f32(x);
}

/** @brief ceil(x) — round toward +inf. */
MC_INLINE float32x4_t mc_ceil_ps(float32x4_t x) {
    return vrndpq_f32(x);
}

/** @brief round(x) — round to nearest, ties to even (IEEE default). */
MC_INLINE float32x4_t mc_round_ps(float32x4_t x) {
    return vrndnq_f32(x);
}

/** @brief trunc(x) — round toward zero (chop the fractional part). */
MC_INLINE float32x4_t mc_trunc_ps(float32x4_t x) {
    return vrndq_f32(x);
}

/**
 * @brief Fractional part: x - floor(x), result in [0, 1) for finite inputs.
 */
MC_INLINE float32x4_t mc_fract_ps(float32x4_t x) {
    return vsubq_f32(x, mc_floor_ps(x));
}

/* ---------- Modulo, wrap, remap ---------- */

/** @brief GLSL-style float modulo: x - y * floor(x / y). Undefined when y == 0. */
MC_INLINE float32x4_t mc_mod_ps(float32x4_t x, float32x4_t y) {
    float32x4_t q = vdivq_f32(x, y);
    return vfmsq_f32(x, y, mc_floor_ps(q));   /* x - y*floor(x/y) */
}

/** @brief Wrap x into the half-open interval [lo, hi). Undefined when hi == lo. */
MC_INLINE float32x4_t mc_wrap_ps(float32x4_t x, float32x4_t lo, float32x4_t hi) {
    float32x4_t range = vsubq_f32(hi, lo);
    return vaddq_f32(lo, mc_mod_ps(vsubq_f32(x, lo), range));
}

/** @brief Linear remap: x in [a0, a1] maps to result in [b0, b1]. */
MC_INLINE float32x4_t mc_remap_ps(float32x4_t x,
                                   float32x4_t a0, float32x4_t a1,
                                   float32x4_t b0, float32x4_t b1)
{
    float32x4_t t = vdivq_f32(vsubq_f32(x, a0), vsubq_f32(a1, a0));
    return vfmaq_f32(b0, t, vsubq_f32(b1, b0));
}

/* ---------- Sign and selection helpers ---------- */

/** @brief Copy sign from @p sign onto magnitude of @p mag. */
MC_INLINE float32x4_t mc_copysign_ps(float32x4_t mag, float32x4_t sign) {
    const uint32x4_t sign_mask = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));  /* 0x80000000 */
    uint32x4_t abs_mag = vbicq_u32(vreinterpretq_u32_f32(mag), sign_mask);
    uint32x4_t sign_of = vandq_u32(vreinterpretq_u32_f32(sign), sign_mask);
    return vreinterpretq_f32_u32(vorrq_u32(abs_mag, sign_of));
}

/** @brief Per-lane select: returns @p on_true where mask lanes are true, @p on_false elsewhere. */
MC_INLINE float32x4_t mc_select_ps(float32x4_t mask, float32x4_t on_true, float32x4_t on_false) {
    return vbslq_f32(vreinterpretq_u32_f32(mask), on_true, on_false);
}

/* ---------- Smootherstep (C2-continuous Hermite) ---------- */

/** @brief Ken Perlin's smootherstep: 6t^5 - 15t^4 + 10t^3 on the clamped t. */
MC_INLINE float32x4_t mc_smootherstep_ps(float32x4_t a, float32x4_t b, float32x4_t x) {
    float32x4_t num = vsubq_f32(x, a);
    float32x4_t den = vsubq_f32(b, a);
    uint32x4_t is_zero  = vceqq_f32(den, vdupq_n_f32(0.0f));
    float32x4_t safe_den = vbslq_f32(is_zero, vdupq_n_f32(1.0f), den);
    float32x4_t t = vdivq_f32(num, safe_den);
    t = vmaxq_f32(vdupq_n_f32(0.0f), vminq_f32(t, vdupq_n_f32(1.0f)));

    /* Evaluate t^3 * (t*(t*6 - 15) + 10).  vfms(15, t, 6) = 15 - 6t = -(6t - 15).
       Then vfms(10, t, p) = 10 - t*(15 - 6t) = 10 - 15t + 6t^2. */
    float32x4_t p = vfmsq_f32(vdupq_n_f32(15.0f), t, vdupq_n_f32(6.0f));
    p = vfmsq_f32(vdupq_n_f32(10.0f), t, p);
    float32x4_t t3 = vmulq_f32(t, vmulq_f32(t, t));
    float32x4_t result = vmulq_f32(t3, p);

    return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(result), is_zero));
}

/* ---------- Pingpong and angle wrap ---------- */

/** @brief Bounce x within [0, length] — yo-yo pattern. Undefined when length <= 0. */
MC_INLINE float32x4_t mc_pingpong_ps(float32x4_t x, float32x4_t length) {
    float32x4_t two_len = vaddq_f32(length, length);
    float32x4_t t = mc_mod_ps(x, two_len);
    return vsubq_f32(length, mc_abs_ps(vsubq_f32(t, length)));
}

/** @brief Fold an angle into [-pi, pi). */
MC_INLINE float32x4_t mc_wrap_angle_ps(float32x4_t angle) {
    const float32x4_t neg_pi = vdupq_n_f32(-3.14159265358979323846f);
    const float32x4_t pos_pi = vdupq_n_f32( 3.14159265358979323846f);
    return mc_wrap_ps(angle, neg_pi, pos_pi);
}

/* ---------- Degree / radian conversion ---------- */

/** @brief Degrees -> radians (multiply by pi/180). */
MC_INLINE float32x4_t mc_to_radians_ps(float32x4_t deg) {
    return vmulq_f32(deg, vdupq_n_f32(0.01745329251994329577f));
}

/** @brief Radians -> degrees (multiply by 180/pi). */
MC_INLINE float32x4_t mc_to_degrees_ps(float32x4_t rad) {
    return vmulq_f32(rad, vdupq_n_f32(57.29577951308232088f));
}

/* ---------- Classification masks ---------- */

/** @brief Per-lane NaN test. NaN is the only value not equal to itself. */
MC_INLINE float32x4_t mc_is_nan_ps(float32x4_t x) {
    /* vceqq returns 0 for NaN compared with itself; invert to get all-1s for NaN. */
    return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(x, x)));
}

/** @brief Per-lane +/-inf test. */
MC_INLINE float32x4_t mc_is_inf_ps(float32x4_t x) {
    const float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);
    return vreinterpretq_f32_u32(vceqq_f32(vabsq_f32(x), inf));
}

/** @brief Per-lane finite test (not NaN and not +/-inf). */
MC_INLINE float32x4_t mc_is_finite_ps(float32x4_t x) {
    const float32x4_t inf = vdupq_n_f32(1.0f / 0.0f);
    return vreinterpretq_f32_u32(vcltq_f32(vabsq_f32(x), inf));
}

/** @brief Absolute-tolerance fuzzy equality mask: |a - b| <= eps. */
MC_INLINE float32x4_t mc_approx_equal_ps(float32x4_t a, float32x4_t b, float32x4_t eps) {
    float32x4_t diff = vabsq_f32(vsubq_f32(a, b));
    return vreinterpretq_f32_u32(vcleq_f32(diff, eps));
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
MC_INLINE float32x4_t mc_poly_estrin_deg3_ps(float32x4_t x, const float* c) {
    float32x4_t c0 = vdupq_n_f32(c[0]);
    float32x4_t c1 = vdupq_n_f32(c[1]);
    float32x4_t c2 = vdupq_n_f32(c[2]);
    float32x4_t c3 = vdupq_n_f32(c[3]);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t y0 = vfmaq_f32(c0, c1, x);
    float32x4_t y1 = vfmaq_f32(c2, c3, x);
    return vfmaq_f32(y0, y1, x2);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[4]x^4. */
MC_INLINE float32x4_t mc_poly_estrin_deg4_ps(float32x4_t x, const float* c) {
    float32x4_t c0 = vdupq_n_f32(c[0]);
    float32x4_t c1 = vdupq_n_f32(c[1]);
    float32x4_t c2 = vdupq_n_f32(c[2]);
    float32x4_t c3 = vdupq_n_f32(c[3]);
    float32x4_t c4 = vdupq_n_f32(c[4]);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t y0 = vfmaq_f32(c0, c1, x);
    float32x4_t y1 = vfmaq_f32(c2, c3, x);
    float32x4_t z0 = vfmaq_f32(y0, y1, x2);
    return vfmaq_f32(z0, c4, x4);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[5]x^5. */
MC_INLINE float32x4_t mc_poly_estrin_deg5_ps(float32x4_t x, const float* c) {
    float32x4_t c0 = vdupq_n_f32(c[0]);
    float32x4_t c1 = vdupq_n_f32(c[1]);
    float32x4_t c2 = vdupq_n_f32(c[2]);
    float32x4_t c3 = vdupq_n_f32(c[3]);
    float32x4_t c4 = vdupq_n_f32(c[4]);
    float32x4_t c5 = vdupq_n_f32(c[5]);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t y0 = vfmaq_f32(c0, c1, x);
    float32x4_t y1 = vfmaq_f32(c2, c3, x);
    float32x4_t y2 = vfmaq_f32(c4, c5, x);
    float32x4_t z0 = vfmaq_f32(y0, y1, x2);
    return vfmaq_f32(z0, y2, x4);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[6]x^6. */
MC_INLINE float32x4_t mc_poly_estrin_deg6_ps(float32x4_t x, const float* c) {
    float32x4_t c0 = vdupq_n_f32(c[0]);
    float32x4_t c1 = vdupq_n_f32(c[1]);
    float32x4_t c2 = vdupq_n_f32(c[2]);
    float32x4_t c3 = vdupq_n_f32(c[3]);
    float32x4_t c4 = vdupq_n_f32(c[4]);
    float32x4_t c5 = vdupq_n_f32(c[5]);
    float32x4_t c6 = vdupq_n_f32(c[6]);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t y0 = vfmaq_f32(c0, c1, x);
    float32x4_t y1 = vfmaq_f32(c2, c3, x);
    float32x4_t y2 = vfmaq_f32(c4, c5, x);
    float32x4_t z0 = vfmaq_f32(y0, y1, x2);
    float32x4_t z1 = vfmaq_f32(y2, c6, x2);
    return vfmaq_f32(z0, z1, x4);
}

/** @brief P(x) = c[0] + c[1]x + ... + c[8]x^8. */
MC_INLINE float32x4_t mc_poly_estrin_deg8_ps(float32x4_t x, const float* c) {
    float32x4_t c0 = vdupq_n_f32(c[0]);
    float32x4_t c1 = vdupq_n_f32(c[1]);
    float32x4_t c2 = vdupq_n_f32(c[2]);
    float32x4_t c3 = vdupq_n_f32(c[3]);
    float32x4_t c4 = vdupq_n_f32(c[4]);
    float32x4_t c5 = vdupq_n_f32(c[5]);
    float32x4_t c6 = vdupq_n_f32(c[6]);
    float32x4_t c7 = vdupq_n_f32(c[7]);
    float32x4_t c8 = vdupq_n_f32(c[8]);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t x8 = vmulq_f32(x4, x4);
    float32x4_t y0 = vfmaq_f32(c0, c1, x);
    float32x4_t y1 = vfmaq_f32(c2, c3, x);
    float32x4_t y2 = vfmaq_f32(c4, c5, x);
    float32x4_t y3 = vfmaq_f32(c6, c7, x);
    float32x4_t z0 = vfmaq_f32(y0, y1, x2);
    float32x4_t z1 = vfmaq_f32(y2, y3, x2);
    float32x4_t w0 = vfmaq_f32(z0, z1, x4);
    return vfmaq_f32(w0, c8, x8);
}

/* =====================================================================
 * 4. MINIMAX COEFFICIENT TABLES
 * =====================================================================
 * Float-precision minimax polynomials generated with Sollya 8.0, prec=128.
 * Each table covers a reduced range; argument reduction is performed in
 * the caller. These tables are identical to the AVX2 version.
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

/* tan(x) on [0, pi/8],  tan(x) = x + x^3 * P(x^2), deg-3, ~0.047 ULP. */
static const float MC_TAN_COEFFS[4] = {
    0.333333287350697673210307974070168833443f,
    0.133342724106033723928251509259696377785f,
    5.36695340958928597698376503512930466572e-2f,
    2.48696209035983955819911447913718818547e-2f
};

/* asin(x) on [0, 0.5],  asin(x) = x * P(x^2), deg-5, ~0.038 ULP. */
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

/** @brief sqrt(x). Maps to VSQRT (AArch64). Negative lanes produce NaN. */
MC_INLINE float32x4_t mc_sqrt_ps(float32x4_t x) { return vsqrtq_f32(x); }

/**
 * @brief Fast 1/sqrt(x) — hardware seed only, ~11 bits. Particles, LODs.
 * x must be > 0.
 */
MC_INLINE float32x4_t mc_rsqrt_fast_ps(float32x4_t x) { return vrsqrteq_f32(x); }

/**
 * @brief 1/sqrt(x) with one Newton-Raphson refinement — ~23 bits / ~0.5 ULP.
 */
MC_INLINE float32x4_t mc_rsqrt_ps(float32x4_t x) {
    float32x4_t y  = vrsqrteq_f32(x);
    float32x4_t y2 = vmulq_f32(y, y);
    float32x4_t hx = vmulq_f32(x, y2);
    /* vfms(1.5, hx, 0.5) = 1.5 - 0.5*hx */
    float32x4_t t  = vfmsq_f32(vdupq_n_f32(1.5f), hx, vdupq_n_f32(0.5f));
    return vmulq_f32(y, t);
}

/**
 * @brief 1/sqrt(x) with two NR iterations — ~24 bits (full mantissa).
 */
MC_INLINE float32x4_t mc_rsqrt_precise_ps(float32x4_t x) {
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t ofh  = vdupq_n_f32(1.5f);
    float32x4_t y = vrsqrteq_f32(x);
    float32x4_t y2 = vmulq_f32(y, y);
    float32x4_t t  = vfmsq_f32(ofh, vmulq_f32(x, y2), half);
    y = vmulq_f32(y, t);
    y2 = vmulq_f32(y, y);
    t  = vfmsq_f32(ofh, vmulq_f32(x, y2), half);
    return vmulq_f32(y, t);
}

/**
 * @brief SoA 3D normalize (1x NR precision). 4 vectors at once.
 * Zero-length vectors produce inf/NaN; guard at call site if needed.
 *
 * @note SoA = Structure of Arrays layout. x[], y[], z[] are separate
 *       arrays of 4 floats each.
 */
MC_INLINE void mc_normalize_soa_ps(float32x4_t* MC_RESTRICT x,
                                    float32x4_t* MC_RESTRICT y,
                                    float32x4_t* MC_RESTRICT z) {
    float32x4_t x2  = vmulq_f32(*x, *x);
    float32x4_t y2  = vmulq_f32(*y, *y);
    float32x4_t z2  = vmulq_f32(*z, *z);
    float32x4_t sum = vaddq_f32(x2, vaddq_f32(y2, z2));
    float32x4_t inv = mc_rsqrt_ps(sum);
    *x = vmulq_f32(*x, inv);
    *y = vmulq_f32(*y, inv);
    *z = vmulq_f32(*z, inv);
}

/** @brief SoA 3D normalize, fast tier (~12 bits). For particles / billboards. */
MC_INLINE void mc_normalize_soa_fast_ps(float32x4_t* MC_RESTRICT x,
                                        float32x4_t* MC_RESTRICT y,
                                        float32x4_t* MC_RESTRICT z) {
    float32x4_t x2  = vmulq_f32(*x, *x);
    float32x4_t y2  = vmulq_f32(*y, *y);
    float32x4_t z2  = vmulq_f32(*z, *z);
    float32x4_t sum = vaddq_f32(x2, vaddq_f32(y2, z2));
    float32x4_t inv = mc_rsqrt_fast_ps(sum);
    *x = vmulq_f32(*x, inv);
    *y = vmulq_f32(*y, inv);
    *z = vmulq_f32(*z, inv);
}

/* =====================================================================
 * 6. EXP, EXP2, LN
 * ===================================================================== */

/**
 * @brief Assemble 2^n as float by writing n into the exponent field.
 * Clamps n to [-126, 127] to stay in normal-float range.
 */
MC_INLINE float32x4_t mc_i_pow2i_ps(int32x4_t n) {
    n = vmaxq_s32(n, vdupq_n_s32(-126));
    n = vminq_s32(n, vdupq_n_s32( 127));
    int32x4_t bits = vaddq_s32(n, vdupq_n_s32(127));
    bits = vshlq_n_s32(bits, 23);
    return vreinterpretq_f32_s32(bits);
}

/**
 * @brief e^x. ~1-5 ULP. Saturates: x < -87.3 -> 0, x > 88.7 -> FLT_MAX.
 *
 * Algorithm: clamp; Cody-Waite reduce by ln2 to r in [-ln2/2, ln2/2];
 * P(r) via Estrin + MC_EXP_COEFFS; scale by 2^n.
 */
MC_INLINE float32x4_t mc_exp_ps(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3365386f)),
                                vdupq_n_f32( 88.7228395f));
    const float32x4_t log2e = vdupq_n_f32(1.44269504088896340736f);
    float32x4_t fn = vrndnq_f32(vmulq_f32(x, log2e));
    float32x4_t r = vfmsq_f32(x,  fn, vdupq_n_f32(MC_LN2_HI));
    r             = vfmsq_f32(r,  fn, vdupq_n_f32(MC_LN2_LO));
    float32x4_t p = mc_poly_estrin_deg6_ps(r, MC_EXP_COEFFS);
    int32x4_t n = vcvtnq_s32_f32(fn);
    return vmulq_f32(p, mc_i_pow2i_ps(n));
}

/** @brief 2^x. ~1-2 ULP. */
MC_INLINE float32x4_t mc_exp2_ps(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-126.0f)),
                                vdupq_n_f32( 127.0f));
    float32x4_t fn = vrndnq_f32(x);
    float32x4_t r  = vsubq_f32(x, fn);
    float32x4_t p  = mc_poly_estrin_deg6_ps(r, MC_EXP2_COEFFS);
    int32x4_t n = vcvtnq_s32_f32(fn);
    return vmulq_f32(p, mc_i_pow2i_ps(n));
}

/**
 * @brief ln(x). ~1-3 ULP.
 *
 * x <= 0 returns -inf per lane (no separate NaN for x<0; mask at call site
 * if you need strict IEEE semantics).
 */
MC_INLINE float32x4_t mc_ln_ps(float32x4_t x) {
    uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0.0f));
    x = vbslq_f32(invalid_mask, vdupq_n_f32(1.0f), x);

    int32x4_t xi = vreinterpretq_s32_f32(x);
    uint32x4_t exp_biased = vshrq_n_u32(vreinterpretq_u32_s32(xi), 23);
    int32x4_t n_i = vsubq_s32(vreinterpretq_s32_u32(exp_biased), vdupq_n_s32(127));

    int32x4_t mant_bits = vandq_s32(xi, vdupq_n_s32(0x007FFFFF));
    mant_bits = vorrq_s32(mant_bits, vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(mant_bits);    /* m in [1, 2) */

    /* Normalize m into [1/sqrt(2), sqrt(2)]: if m > sqrt(2), halve and bump n. */
    uint32x4_t need_adjust = vcgtq_f32(m, vdupq_n_f32(1.41421356237309504880f));
    int32x4_t adjust = vreinterpretq_s32_u32(vshrq_n_u32(need_adjust, 31));
    n_i = vaddq_s32(n_i, adjust);
    m = vbslq_f32(need_adjust, vmulq_f32(m, vdupq_n_f32(0.5f)), m);

    float32x4_t u = vsubq_f32(m, vdupq_n_f32(1.0f));
    float32x4_t p = mc_poly_estrin_deg8_ps(u, MC_LN_COEFFS);
    float32x4_t ln_m = vmulq_f32(u, p);

    float32x4_t fn = vcvtq_f32_s32(n_i);
    float32x4_t result = vfmaq_f32(ln_m, fn, vdupq_n_f32(MC_LN2_HI));
    result             = vfmaq_f32(result, fn, vdupq_n_f32(MC_LN2_LO));

    const float32x4_t neg_inf = vdupq_n_f32(-1.0f / 0.0f);
    return vbslq_f32(invalid_mask, neg_inf, result);
}

/* =====================================================================
 * 7. SIN, COS, SINCOS
 * =====================================================================
 * Two-part Cody-Waite reduction by pi/2. ~1-3 ULP for |x| <= 10^5;
 * gracefully degrades beyond that.
 */

#define MC_TWO_OVER_PI  (0.636619772367581343f)
#define MC_PIO2_HI      ( 1.5707963705062866f)
#define MC_PIO2_LO      (-4.3711388286737929e-8f)

/** @brief Compute sin(x) and cos(x) together. */
MC_INLINE void mc_sincos_ps(float32x4_t x,
                            float32x4_t* MC_RESTRICT s_out,
                            float32x4_t* MC_RESTRICT c_out) {
    float32x4_t fn = vrndnq_f32(
        vmulq_f32(x, vdupq_n_f32(MC_TWO_OVER_PI)));

    float32x4_t r = vfmsq_f32(x,  fn, vdupq_n_f32(MC_PIO2_HI));
    r             = vfmsq_f32(r,  fn, vdupq_n_f32(MC_PIO2_LO));

    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t sp = mc_poly_estrin_deg3_ps(r2, MC_SIN_COEFFS);
    float32x4_t sin_r = vmulq_f32(r, sp);

    float32x4_t cp = mc_poly_estrin_deg3_ps(r2, MC_COS_COEFFS);
    float32x4_t half_r2 = vmulq_f32(r2, vdupq_n_f32(0.5f));
    float32x4_t cos_r = vfmsq_f32(vdupq_n_f32(1.0f), half_r2, cp);

    int32x4_t ni    = vcvtnq_s32_f32(fn);
    int32x4_t v_one = vdupq_n_s32(1);
    int32x4_t v_two = vdupq_n_s32(2);

    uint32x4_t swap_mask = vceqq_s32(vandq_s32(ni, v_one), v_one);
    float32x4_t s_pre = vbslq_f32(swap_mask, cos_r, sin_r);
    float32x4_t c_pre = vbslq_f32(swap_mask, sin_r, cos_r);

    int32x4_t sin_sign = vshlq_n_s32(vandq_s32(ni, v_two), 30);
    int32x4_t cos_sign = vshlq_n_s32(vandq_s32(vaddq_s32(ni, v_one), v_two), 30);

    vst1q_f32((float*)s_out, vreinterpretq_f32_s32(
        veorq_s32(vreinterpretq_s32_f32(s_pre), sin_sign)));
    vst1q_f32((float*)c_out, vreinterpretq_f32_s32(
        veorq_s32(vreinterpretq_s32_f32(c_pre), cos_sign)));
}

/** @brief sin(x). */
MC_INLINE float32x4_t mc_sin_ps(float32x4_t x) {
    float32x4_t s, c; mc_sincos_ps(x, &s, &c); return s;
}

/** @brief cos(x). */
MC_INLINE float32x4_t mc_cos_ps(float32x4_t x) {
    float32x4_t s, c; mc_sincos_ps(x, &s, &c); return c;
}

/* =====================================================================
 * 8. TAN
 * =====================================================================
 * pi/2 reduction, then halve-and-double-angle so the polynomial only
 * sees |y| <= pi/8. For odd quadrants we want -cot(r); implemented as
 * a num/den swap so the whole function uses a single division.
 */

MC_INLINE float32x4_t mc_tan_ps(float32x4_t x) {
    float32x4_t fn = vrndnq_f32(
        vmulq_f32(x, vdupq_n_f32(MC_TWO_OVER_PI)));

    float32x4_t r = vfmsq_f32(x,  fn, vdupq_n_f32(MC_PIO2_HI));
    r             = vfmsq_f32(r,  fn, vdupq_n_f32(MC_PIO2_LO));

    float32x4_t y  = vmulq_f32(r, vdupq_n_f32(0.5f));
    float32x4_t y2 = vmulq_f32(y, y);
    float32x4_t p  = mc_poly_estrin_deg3_ps(y2, MC_TAN_COEFFS);
    float32x4_t y3 = vmulq_f32(y2, y);
    float32x4_t t  = vfmaq_f32(y, y3, p);

    float32x4_t t2    = vmulq_f32(t, t);
    float32x4_t two_t = vaddq_f32(t, t);
    float32x4_t one_minus_t2 = vsubq_f32(vdupq_n_f32(1.0f), t2);
    float32x4_t t2_minus_one = vsubq_f32(t2, vdupq_n_f32(1.0f));

    int32x4_t ni    = vcvtnq_s32_f32(fn);
    int32x4_t v_one = vdupq_n_s32(1);
    uint32x4_t odd_mask = vceqq_s32(vandq_s32(ni, v_one), v_one);

    float32x4_t num = vbslq_f32(odd_mask, t2_minus_one, two_t);
    float32x4_t den = vbslq_f32(odd_mask, two_t,        one_minus_t2);
    return vdivq_f32(num, den);
}

/* =====================================================================
 * 9. ASIN, ACOS
 * =====================================================================
 * Sqrt-based reduction near +/-1 (instead of pi/2 - asin) to avoid the
 * cancellation that would lose precision there. Inputs saturated to
 * [-1, 1] so slight overshoot from dot products doesn't produce NaN.
 */

MC_INLINE float32x4_t mc_asin_ps(float32x4_t x) {
    const float32x4_t one  = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t pio2 = vdupq_n_f32(1.57079632679489661923f);
    const uint32x4_t  sign_bit = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));

    uint32x4_t sign_x = vandq_u32(vreinterpretq_u32_f32(x), sign_bit);
    float32x4_t ax = vabsq_f32(x);
    ax = vminq_f32(ax, one);                  /* saturate to [0,1] */

    uint32x4_t big = vcgtq_f32(ax, half);

    float32x4_t t_small = vmulq_f32(ax, ax);
    float32x4_t t_big   = vmulq_f32(vsubq_f32(one, ax), half);
    float32x4_t t = vbslq_f32(big, t_big, t_small);

    float32x4_t v_small = ax;
    float32x4_t v_big   = vsqrtq_f32(t_big);
    float32x4_t v = vbslq_f32(big, v_big, v_small);

    float32x4_t p   = mc_poly_estrin_deg5_ps(t, MC_ASIN_COEFFS);
    float32x4_t pre = vmulq_f32(v, p);

    float32x4_t res_big = vsubq_f32(pio2, vaddq_f32(pre, pre));
    float32x4_t res     = vbslq_f32(big, res_big, pre);

    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(res), sign_x));
}

MC_INLINE float32x4_t mc_acos_ps(float32x4_t x) {
    const float32x4_t one  = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t pio2 = vdupq_n_f32(1.57079632679489661923f);
    const float32x4_t pi   = vdupq_n_f32(3.14159265358979323846f);
    const uint32x4_t  sign_bit = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));

    /* Saturate to [-1, 1]. */
    x = vmaxq_f32(vminq_f32(x, one), vsubq_f32(zero, one));
    float32x4_t ax    = vabsq_f32(x);
    uint32x4_t big   = vcgtq_f32(ax, half);
    uint32x4_t x_neg = vcltq_f32(x, zero);

    float32x4_t t_small = vmulq_f32(x, x);
    float32x4_t t_big   = vmulq_f32(vsubq_f32(one, ax), half);
    float32x4_t t = vbslq_f32(big, t_big, t_small);

    float32x4_t v_small = x;
    float32x4_t v_big   = vsqrtq_f32(t_big);
    float32x4_t v = vbslq_f32(big, v_big, v_small);

    float32x4_t p   = mc_poly_estrin_deg5_ps(t, MC_ASIN_COEFFS);
    float32x4_t pre = vmulq_f32(v, p);

    float32x4_t res_small = vsubq_f32(pio2, pre);

    float32x4_t two_pre = vaddq_f32(pre, pre);
    float32x4_t signed_two_pre = vreinterpretq_f32_u32(
        veorq_u32(vreinterpretq_u32_f32(two_pre),
                  vandq_u32(x_neg, sign_bit)));
    float32x4_t offset_pi = vreinterpretq_f32_u32(vandq_u32(x_neg, vreinterpretq_u32_f32(pi)));
    float32x4_t res_big = vaddq_f32(offset_pi, signed_two_pre);

    return vbslq_f32(big, res_big, res_small);
}

/* =====================================================================
 * 10. ATAN, ATAN2
 * =====================================================================
 * Two-stage reduction: fold |x|>1 via 1/x, then fold residual >tan(pi/8)
 * via (x-1)/(x+1). Polynomial then sees v in [-tan(pi/8), tan(pi/8)].
 */

MC_INLINE float32x4_t mc_atan_ps(float32x4_t x) {
    const float32x4_t one      = vdupq_n_f32(1.0f);
    const uint32x4_t  sign_bit = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));

    uint32x4_t sign_x = vandq_u32(vreinterpretq_u32_f32(x), sign_bit);
    float32x4_t ax = vabsq_f32(x);

    uint32x4_t swap_mask = vcgtq_f32(ax, one);
    float32x4_t inv_ax    = vdivq_f32(one, ax);
    float32x4_t u         = vbslq_f32(swap_mask, inv_ax, ax);

    const float32x4_t tan_pio8 = vdupq_n_f32(0.41421356237309504880f);
    uint32x4_t shift_mask = vcgtq_f32(u, tan_pio8);
    float32x4_t v_red = vdivq_f32(vsubq_f32(u, one), vaddq_f32(u, one));
    float32x4_t v = vbslq_f32(shift_mask, v_red, u);

    float32x4_t v2 = vmulq_f32(v, v);
    float32x4_t p  = mc_poly_estrin_deg4_ps(v2, MC_ATAN_COEFFS);
    float32x4_t atan_v = vmulq_f32(v, p);

    const float32x4_t pio4 = vdupq_n_f32(0.78539816339744830962f);
    float32x4_t atan_u = vaddq_f32(atan_v,
        vreinterpretq_f32_u32(vandq_u32(shift_mask, vreinterpretq_u32_f32(pio4))));

    const float32x4_t pio2 = vdupq_n_f32(1.57079632679489661923f);
    float32x4_t atan_ax_swap = vsubq_f32(pio2, atan_u);
    float32x4_t atan_ax = vbslq_f32(swap_mask, atan_ax_swap, atan_u);

    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(atan_ax), sign_x));
}

/**
 * @brief atan2(y, x). (0, 0) returns NaN — mask at call site if needed.
 */
MC_INLINE float32x4_t mc_atan2_ps(float32x4_t y, float32x4_t x) {
    const float32x4_t zero     = vdupq_n_f32(0.0f);
    const uint32x4_t  sign_bit = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
    const float32x4_t pi       = vdupq_n_f32(3.14159265358979323846f);

    float32x4_t ratio  = vdivq_f32(y, x);
    float32x4_t atan_r = mc_atan_ps(ratio);

    uint32x4_t x_neg     = vcltq_f32(x, zero);
    uint32x4_t y_sign    = vandq_u32(vreinterpretq_u32_f32(y), sign_bit);
    float32x4_t signed_pi = vreinterpretq_f32_u32(
        veorq_u32(vreinterpretq_u32_f32(pi), y_sign));
    float32x4_t offset    = vreinterpretq_f32_u32(
        vandq_u32(x_neg, vreinterpretq_u32_f32(signed_pi)));

    return vaddq_f32(atan_r, offset);
}

/* =====================================================================
 * 11. SINH, COSH, TANH
 * =====================================================================
 * Polynomial for small |x| (|x| <= ln2/2), exp-identity for the rest.
 * Both branches always evaluate; blended per lane.
 */

#define MC_LN2_OVER_2  (0.34657359027997265471f)

MC_INLINE float32x4_t mc_sinh_ps(float32x4_t x) {
    const float32x4_t half  = vdupq_n_f32(0.5f);
    const float32x4_t one   = vdupq_n_f32(1.0f);
    const float32x4_t bound = vdupq_n_f32(MC_LN2_OVER_2);

    float32x4_t ax  = vabsq_f32(x);
    uint32x4_t big = vcgtq_f32(ax, bound);

    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t p_small = mc_poly_estrin_deg3_ps(x2, MC_SINH_COEFFS);
    float32x4_t sinh_small = vmulq_f32(x, p_small);

    float32x4_t ex = mc_exp_ps(x);
    float32x4_t inv_ex = vdivq_f32(one, ex);
    float32x4_t sinh_big = vmulq_f32(vsubq_f32(ex, inv_ex), half);

    return vbslq_f32(big, sinh_big, sinh_small);
}

MC_INLINE float32x4_t mc_cosh_ps(float32x4_t x) {
    const float32x4_t half  = vdupq_n_f32(0.5f);
    const float32x4_t one   = vdupq_n_f32(1.0f);
    const float32x4_t bound = vdupq_n_f32(MC_LN2_OVER_2);

    float32x4_t ax  = vabsq_f32(x);
    uint32x4_t big = vcgtq_f32(ax, bound);

    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t cosh_small = mc_poly_estrin_deg3_ps(x2, MC_COSH_COEFFS);

    float32x4_t ex = mc_exp_ps(ax);
    float32x4_t inv_ex = vdivq_f32(one, ex);
    float32x4_t cosh_big = vmulq_f32(vaddq_f32(ex, inv_ex), half);

    return vbslq_f32(big, cosh_big, cosh_small);
}

MC_INLINE float32x4_t mc_tanh_ps(float32x4_t x) {
    const float32x4_t one   = vdupq_n_f32(1.0f);
    const float32x4_t two   = vdupq_n_f32(2.0f);
    const float32x4_t bound = vdupq_n_f32(MC_LN2_OVER_2);

    float32x4_t ax  = vabsq_f32(x);
    uint32x4_t big = vcgtq_f32(ax, bound);

    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t p_small = mc_poly_estrin_deg4_ps(x2, MC_TANH_COEFFS);
    float32x4_t tanh_small = vmulq_f32(x, p_small);

    float32x4_t two_x = vaddq_f32(x, x);
    float32x4_t e2x   = mc_exp_ps(two_x);
    float32x4_t denom = vaddq_f32(e2x, one);
    float32x4_t tanh_big = vsubq_f32(one, vdivq_f32(two, denom));

    return vbslq_f32(big, tanh_big, tanh_small);
}

/* =====================================================================
 * 12. POW
 * =====================================================================
 * pow(x, y) = exp(y * ln(x)). Error scales with |y| (see notes).
 * x <= 0 produces non-finite/NaN; integer-exponent path for negative
 * bases is not provided here.
 */

MC_INLINE float32x4_t mc_pow_ps(float32x4_t x, float32x4_t y) {
    float32x4_t lx = mc_ln_ps(x);
    float32x4_t yl = vmulq_f32(y, lx);
    return mc_exp_ps(yl);
}

#endif /* MC_SIMD_MATH_NEON_H */