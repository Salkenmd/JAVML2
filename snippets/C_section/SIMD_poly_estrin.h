/**
 * @file SIMD_poly_estrin.h
 * @brief Polynomial evaluation via Estrin's scheme for AVX2 float (ps).
 *
 * Estrin's scheme reduces sequential dependency chains by computing independent
 * sub-products that the CPU can execute in parallel, improving ILP over Horner.
 *
 * @par Naming convention
 *   `mc_poly_estrin_degN_ps` evaluates a degree-N polynomial with N+1 coefficients.
 *   The argument `x` is whatever the caller passes — it is NOT squared internally
 *   (except for the internal power precomputation). For trig/hyperbolic functions
 *   that use `x^2` as their polynomial variable, pass `x^2` from the call site.
 *
 * @par Coefficient layout
 *   All functions take `const float* c` where `c[0]` is the constant term
 *   and `c[N]` is the highest-degree coefficient. This matches the layout of
 *   the arrays in minimax_coeffs.h.
 */

#ifndef MC_SIMD_POLY_ESTRIN_H
#define MC_SIMD_POLY_ESTRIN_H
#include <immintrin.h>

/**
 * @brief Evaluate a degree-3 polynomial (4 coefficients) via Estrin's scheme.
 *
 * @math P(x) = c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3
 *
 * @param x  Input vector. For sin/cos/tan/sinh/cosh/tanh, pass @c x_orig^2
 *           (the polynomial variable is t = x^2; the caller multiplies by x
 *           outside this function to form the final x*P(x^2) result).
 * @param c  Array of 4 coefficients, c[0] = constant term.
 *
 * @par Callers (from minimax_coeffs.h)
 *   - @c SIN_COEFFS  — pass t=x^2, multiply result by x
 *   - @c COS_COEFFS  — pass t=x^2, result is P(t); caller does 1 - (x^2/2)*P(t)
 *   - @c TAN_COEFFS  — pass t=x^2, multiply result by x^3, add x
 *   - @c SINH_COEFFS — pass t=x^2, multiply result by x
 *   - @c COSH_COEFFS — pass t=x^2, result is P(t) directly (even polynomial)
 */
static inline __m256 mc_poly_estrin_deg3_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);

    __m256 x2 = _mm256_mul_ps(x, x);

    // Level 1: two independent linear pairs
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);   // c0 + c1*x
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);   // c2 + c3*x

    // Level 2: combine with x^2
    return _mm256_fmadd_ps(y1, x2, y0);        // y0 + y1*x^2
}

/**
 * @brief Evaluate a degree-4 polynomial (5 coefficients) via Estrin's scheme.
 *
 * @math P(x) = c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3 + c[4]*x^4
 *
 * @param x  Input vector. For atan/tanh, pass @c x_orig^2.
 * @param c  Array of 5 coefficients, c[0] = constant term.
 *
 * @par Callers (from minimax_coeffs.h)
 *   - @c ATAN_COEFFS  — pass t=x^2, multiply result by x (add range offset if reduced)
 *   - @c TANH_COEFFS  — pass t=x^2, multiply result by x
 */
static inline __m256 mc_poly_estrin_deg4_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);
    __m256 c4 = _mm256_set1_ps(c[4]);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);

    // Level 1: two independent linear pairs
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);

    // Level 2: combine with x^2
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);

    // Level 3: final term with x^4
    return _mm256_fmadd_ps(c4, x4, z0);
}

/**
 * @brief Evaluate a degree-6 polynomial (7 coefficients) via Estrin's scheme.
 *
 * @math P(x) = c[0] + c[1]*x + ... + c[6]*x^6
 *
 * @param x  Input vector. For exp/exp2, pass the reduced argument r directly.
 * @param c  Array of 7 coefficients, c[0] = constant term.
 *
 * @par Callers (from minimax_coeffs.h)
 *   - @c EXP_COEFFS  — pass reduced r in [-ln2/2, ln2/2]; scale result by 2^n
 *   - @c EXP2_COEFFS — pass reduced r in [-0.5, 0.5]; scale result by 2^n
 */
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

    // Level 1: three independent linear pairs
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);   // covers x^0, x^1
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);   // covers x^2, x^3
    __m256 y2 = _mm256_fmadd_ps(c5, x, c4);   // covers x^4, x^5

    // Level 2: combine pairs with x^2
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);  // covers x^0..x^3
    __m256 z1 = _mm256_fmadd_ps(c6, x2, y2);  // covers x^4..x^6

    // Level 3: combine halves with x^4
    return _mm256_fmadd_ps(z1, x4, z0);
}

/**
 * @brief Evaluate a degree-8 polynomial (9 coefficients) via Estrin's scheme.
 *
 * @math P(x) = c[0] + c[1]*x + ... + c[8]*x^8
 *
 * @param x  Input vector.
 * @param c  Array of 9 coefficients, c[0] = constant term.
 *
 * @par Callers (from minimax_coeffs.h)
 *   - @c LN_COEFFS  — pass reduced u = m-1 where m in [1/sqrt(2), sqrt(2)];
 *                     multiply result by u, then add n*ln2
 */
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

    // Level 1: four independent linear pairs
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);
    __m256 y2 = _mm256_fmadd_ps(c5, x, c4);
    __m256 y3 = _mm256_fmadd_ps(c7, x, c6);

    // Level 2: combine with x^2
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);
    __m256 z1 = _mm256_fmadd_ps(y3, x2, y2);

    // Level 3: combine with x^4
    __m256 w0 = _mm256_fmadd_ps(z1, x4, z0);

    // Level 4: final term with x^8
    return _mm256_fmadd_ps(c8, x8, w0);
}

/**
 * @brief Evaluate a degree-5 polynomial (6 coefficients) via Estrin's scheme.
 *
 * @math P(x) = c[0] + c[1]*x + ... + c[5]*x^5
 *
 * @param x  Input vector. For asin, pass @c x_orig^2; multiply result by x outside.
 * @param c  Array of 6 coefficients, c[0] = constant term.
 *
 * @par Callers (from minimax_coeffs.h)
 *   - @c ASIN_COEFFS — pass t=x^2, multiply result by x
 */
static inline __m256 mc_poly_estrin_deg5_ps(__m256 x, const float* c) {
    __m256 c0 = _mm256_set1_ps(c[0]);
    __m256 c1 = _mm256_set1_ps(c[1]);
    __m256 c2 = _mm256_set1_ps(c[2]);
    __m256 c3 = _mm256_set1_ps(c[3]);
    __m256 c4 = _mm256_set1_ps(c[4]);
    __m256 c5 = _mm256_set1_ps(c[5]);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);

    // Level 1: three independent pairs (c5 is odd term out)
    __m256 y0 = _mm256_fmadd_ps(c1, x, c0);
    __m256 y1 = _mm256_fmadd_ps(c3, x, c2);
    __m256 y2 = _mm256_fmadd_ps(c5, x, c4);

    // Level 2: combine with x^2
    __m256 z0 = _mm256_fmadd_ps(y1, x2, y0);  // covers x^0..x^3

    // Level 3: combine with x^4
    return _mm256_fmadd_ps(y2, x4, z0);
}

#endif // MC_SIMD_POLY_ESTRIN_H
