/**
 * @file SIMD_sqrt_ps.h
 * @brief Vectorized square root and reciprocal square root for AVX2 float (ps).
 *
 * Provides three precision tiers for 1/sqrt(x) and SoA vector normalization helpers.
 *
 * @par Precision tiers
 * | Function               | Iterations | ~Bits | Typical use                  |
 * |------------------------|------------|-------|------------------------------|
 * | mc_rsqrt_fast_ps       | 0 (seed)   | 12    | Particles, LOD weights       |
 * | mc_rsqrt_ps            | 1x NR      | 23    | Physics, lighting, normals   |
 * | mc_rsqrt_precise_ps    | 2x NR      | 24    | When last ULP matters        |
 *
 * @note Requires AVX2 and FMA support (compile with -mavx2 -mfma).
 */

#ifndef MC_SIMD_SQRT_PS_H
#define MC_SIMD_SQRT_PS_H
#include <immintrin.h>

/* ============================================================
 * Square Root
 * ============================================================ */

/**
 * @brief Vectorized square root.
 * @math \sqrt{x}
 * @note Maps directly to the hardware VSQRTPS instruction. IEEE 754 compliant,
 *       correctly rounded to 0.5 ULP.
 * @param x Input vector. Negative lanes produce NaN.
 */
static inline __m256 mc_sqrt_ps(__m256 x) {
    return _mm256_sqrt_ps(x);
}

/* ============================================================
 * Reciprocal Square Root  1 / sqrt(x)
 * ============================================================ */

/**
 * @brief Fast reciprocal square root — hardware seed only, no refinement.
 * @math 1 / \sqrt{x}
 * @details Uses VRSQRTPS which gives an ~11-bit approximation. Throughput is
 *          very high (1 cycle on most µarchs) but error is ~0.03%.
 * @note Suitable for particle systems, screen-space effects, coarse LOD.
 *       Do NOT use for physics, lighting math, or anything accumulating error.
 * @param x Input vector. x must be > 0; behaviour for x <= 0 is undefined.
 */
static inline __m256 mc_rsqrt_fast_ps(__m256 x) {
    return _mm256_rsqrt_ps(x);
}

/**
 * @brief Accurate reciprocal square root — 1x Newton-Raphson refinement.
 * @math y_{n+1} = y_n \cdot (1.5 - 0.5 \cdot x \cdot y_n^2)
 * @details Refines the hardware seed to ~23 bits (~0.5 ULP). One extra
 *          multiply-FMA over the fast version. The standard choice for
 *          game physics, per-fragment lighting, and normal map math.
 * @param x Input vector. x must be > 0.
 */
static inline __m256 mc_rsqrt_ps(__m256 x) {
    __m256 y  = _mm256_rsqrt_ps(x);
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 hx = _mm256_mul_ps(x, y2);
    // t = 1.5 - 0.5 * x * y^2
    __m256 t  = _mm256_fnmadd_ps(hx, _mm256_set1_ps(0.5f), _mm256_set1_ps(1.5f));
    return _mm256_mul_ps(y, t);
}

/**
 * @brief High-precision reciprocal square root — 2x Newton-Raphson refinement.
 * @details Applies two NR iterations for ~24-bit precision (full float mantissa).
 *          Costs two extra multiply-FMA pairs over @c mc_rsqrt_ps.
 *          Use only when the full ULP budget matters (e.g. computing exact
 *          unit quaternion lengths, normalizing bone transforms).
 * @param x Input vector. x must be > 0.
 */
static inline __m256 mc_rsqrt_precise_ps(__m256 x) {
    const __m256 half          = _mm256_set1_ps(0.5f);
    const __m256 one_point_five = _mm256_set1_ps(1.5f);

    __m256 y = _mm256_rsqrt_ps(x);

    // Iteration 1
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 t  = _mm256_fnmadd_ps(_mm256_mul_ps(x, y2), half, one_point_five);
    y = _mm256_mul_ps(y, t);

    // Iteration 2
    y2 = _mm256_mul_ps(y, y);
    t  = _mm256_fnmadd_ps(_mm256_mul_ps(x, y2), half, one_point_five);
    return _mm256_mul_ps(y, t);
}

/* ============================================================
 * SoA Vector Normalization
 * ============================================================ */

/**
 * @brief Normalize 3D vectors in SoA (Structure of Arrays) layout — 1x NR precision.
 * @math \hat{v} = v \cdot (1 / \sqrt{x^2 + y^2 + z^2})
 * @details Processes 8 vectors simultaneously. Uses @c mc_rsqrt_ps (~23 bits).
 *          Appropriate for mesh normals, lighting directions, physics contacts.
 * @param[in,out] x  Pointer to x-component lane; overwritten with normalized x.
 * @param[in,out] y  Pointer to y-component lane; overwritten with normalized y.
 * @param[in,out] z  Pointer to z-component lane; overwritten with normalized z.
 * @warning Zero-length vectors produce inf/NaN. Guard at call site if needed.
 */
static inline void mc_normalize_soa_ps(__m256* x, __m256* y, __m256* z) {
    __m256 x2  = _mm256_mul_ps(*x, *x);
    __m256 y2  = _mm256_mul_ps(*y, *y);
    __m256 z2  = _mm256_mul_ps(*z, *z);
    __m256 sum = _mm256_add_ps(x2, _mm256_add_ps(y2, z2));

    __m256 inv_len = mc_rsqrt_ps(sum);

    *x = _mm256_mul_ps(*x, inv_len);
    *y = _mm256_mul_ps(*y, inv_len);
    *z = _mm256_mul_ps(*z, inv_len);
}

/**
 * @brief Normalize 3D vectors in SoA layout — fast (~12-bit) approximation.
 * @details Uses the raw hardware seed with no NR refinement (~12 bits, ~0.03% error).
 *          Suitable for particle orientations, billboards, coarse screen-space effects
 *          where a slightly non-unit result is visually acceptable.
 * @param[in,out] x  Pointer to x-component lane.
 * @param[in,out] y  Pointer to y-component lane.
 * @param[in,out] z  Pointer to z-component lane.
 */
static inline void mc_normalize_soa_fast_ps(__m256* x, __m256* y, __m256* z) {
    __m256 x2  = _mm256_mul_ps(*x, *x);
    __m256 y2  = _mm256_mul_ps(*y, *y);
    __m256 z2  = _mm256_mul_ps(*z, *z);
    __m256 sum = _mm256_add_ps(x2, _mm256_add_ps(y2, z2));

    __m256 inv_len = mc_rsqrt_fast_ps(sum);

    *x = _mm256_mul_ps(*x, inv_len);
    *y = _mm256_mul_ps(*y, inv_len);
    *z = _mm256_mul_ps(*z, inv_len);
}

#endif // MC_SIMD_SQRT_PS_H
