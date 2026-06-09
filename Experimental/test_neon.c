/**
 * @file test_neon.c
 * @brief Sanity + accuracy check for mc_simd_math_neon.
 *
 * Self-contained: includes the NEON header and runs a battery of checks
 * across all 14 transcendentals + foundation functions. Prints a pass/fail
 * summary. Designed for Termux (AArch64 Android).
 *
 * Compile:
 *   clang -O2 -march=armv8-a+fp+simd test_neon.c -o test_neon -lm
 *   gcc   -O2 -march=armv8-a+fp+simd test_neon.c -o test_neon -lm
 */

#include <stdio.h>
#include <math.h>
#include <arm_neon.h>
#include "mc_simd_math_neon.h"

static int g_fails = 0;

/* ----------------------------------------------------------------
 * Tiny test helpers
 * ---------------------------------------------------------------- */

#define CHECK(cond, msg) do {                           \
    if (!(cond)) {                                      \
        printf("  FAIL  %s\n", msg);                    \
        g_fails++;                                      \
    } else {                                            \
        printf("  OK    %s\n", msg);                    \
    }                                                   \
} while (0)

static float rel_err(float got, float ref) {
    float den = fabsf(ref);
    if (den < 1e-30f) den = 1e-30f;
    return fabsf(got - ref) / den;
}

#define TOL  1e-5f    /* ~5 ULP at float — generous for sanity */

/* ----------------------------------------------------------------
 * Foundation
 * ---------------------------------------------------------------- */

static void test_foundation(void) {
    printf("--- Foundation ---\n");
    float32x4_t x = (float32x4_t){ -3.0f, -1.0f, 0.0f, 2.5f };

    float32x4_t a = mc_abs_ps(x);
    float buf[4]; vst1q_f32(buf, a);
    CHECK(buf[0] == 3.0f && buf[1] == 1.0f && buf[2] == 0.0f && buf[3] == 2.5f,
          "mc_abs_ps");

    float32x4_t t = mc_clamp_ps(x, vdupq_n_f32(0.0f), vdupq_n_f32(1.0f));
    vst1q_f32(buf, t);
    CHECK(buf[0] == 0.0f && buf[1] == 0.0f && buf[2] == 0.0f && buf[3] == 1.0f,
          "mc_clamp_ps");

    float32x4_t l = mc_lerp_ps(vdupq_n_f32(0.0f), vdupq_n_f32(10.0f), vdupq_n_f32(0.3f));
    vst1q_f32(buf, l);
    CHECK(rel_err(buf[0], 3.0f) < TOL, "mc_lerp_ps");

    float32x4_t r = mc_round_ps((float32x4_t){ 1.4f, 1.5f, 2.6f, -0.7f });
    vst1q_f32(buf, r);
    CHECK(buf[0] == 1.0f && buf[1] == 2.0f && buf[2] == 3.0f && buf[3] == -1.0f,
          "mc_round_ps");
}

/* ----------------------------------------------------------------
 * Transcendentals — check near known values
 * ---------------------------------------------------------------- */

static float get_lane(float32x4_t v, int i) {
    float buf[4]; vst1q_f32(buf, v); return buf[i];
}

static void test_exp(void) {
    printf("--- exp / exp2 / ln ---\n");

    float32x4_t x = (float32x4_t){ 0.0f, 1.0f, 2.0f, -1.0f };
    float32x4_t e = mc_exp_ps(x);
    CHECK(rel_err(get_lane(e, 0), 1.000000f) < TOL, "exp(0)");
    CHECK(rel_err(get_lane(e, 1), 2.718282f) < TOL, "exp(1)");
    CHECK(rel_err(get_lane(e, 2), 7.389056f) < TOL, "exp(2)");
    CHECK(rel_err(get_lane(e, 3), 0.367879f) < TOL, "exp(-1)");

    float32x4_t e2 = mc_exp2_ps((float32x4_t){ 0.0f, 1.0f, 3.0f, -1.0f });
    CHECK(rel_err(get_lane(e2, 0), 1.0f) < TOL, "exp2(0)");
    CHECK(rel_err(get_lane(e2, 1), 2.0f) < TOL, "exp2(1)");
    CHECK(rel_err(get_lane(e2, 2), 8.0f) < TOL, "exp2(3)");
    CHECK(rel_err(get_lane(e2, 3), 0.5f) < TOL, "exp2(-1)");

    float32x4_t l = mc_ln_ps((float32x4_t){ 1.0f, 2.718282f, 7.389056f, 10.0f });
    CHECK(rel_err(get_lane(l, 0), 0.000000f) < TOL, "ln(1)");
    CHECK(rel_err(get_lane(l, 1), 1.000000f) < TOL, "ln(e)");
    CHECK(rel_err(get_lane(l, 2), 2.000000f) < TOL, "ln(e^2)");
    CHECK(rel_err(get_lane(l, 3), 2.302585f) < TOL, "ln(10)");
}

static void test_sincos(void) {
    printf("--- sin / cos / sincos ---\n");

    float32x4_t x = (float32x4_t){ 0.0f, 1.57079632679f, 3.14159265359f, 4.71238898038f };
    float32x4_t s, c;
    mc_sincos_ps(x, &s, &c);

    /* sin: 0, 1, 0, -1  (use absolute error when reference is 0) */
    CHECK(rel_err(get_lane(s, 0),  0.0f) < TOL, "sin(0)");
    CHECK(rel_err(get_lane(s, 1),  1.0f) < TOL, "sin(pi/2)");
    CHECK(fabsf(get_lane(s, 2))     < 2e-4f, "sin(pi)");  /* ~π error from Cody-Waite */
    CHECK(rel_err(get_lane(s, 3), -1.0f) < TOL, "sin(3pi/2)");

    /* cos: 1, 0, -1, 0 */
    CHECK(rel_err(get_lane(c, 0),  1.0f) < TOL, "cos(0)");
    CHECK(fabsf(get_lane(c, 1))     < 2e-4f, "cos(pi/2)");
    CHECK(rel_err(get_lane(c, 2), -1.0f) < 2e-4f, "cos(pi)");
    CHECK(fabsf(get_lane(c, 3))     < 2e-4f, "cos(3pi/2)");

    /* Identity: sin^2 + cos^2 = 1 for all lanes */
    float buf_s[4], buf_c[4];
    vst1q_f32(buf_s, s); vst1q_f32(buf_c, c);
    for (int i = 0; i < 4; i++) {
        float sum = buf_s[i]*buf_s[i] + buf_c[i]*buf_c[i];
        if (fabsf(sum - 1.0f) > 1e-5f) {
            printf("  FAIL  sin^2+cos^2 lane %d = %f\n", i, sum);
            g_fails++;
        } else {
            printf("  OK    sin^2+cos^2 lane %d = %.6f\n", i, sum);
        }
    }
}

static void test_tan(void) {
    printf("--- tan ---\n");
    float32x4_t x = (float32x4_t){ 0.0f, 0.78539816339f, -0.78539816339f, 1.0f };
    float32x4_t t = mc_tan_ps(x);
    CHECK(rel_err(get_lane(t, 0), 0.000000f) < TOL, "tan(0)");
    CHECK(rel_err(get_lane(t, 1), 1.000000f) < TOL, "tan(pi/4)");
    CHECK(rel_err(get_lane(t, 2), -1.000000f) < TOL, "tan(-pi/4)");
    CHECK(rel_err(get_lane(t, 3), 1.557408f) < TOL, "tan(1)");
}

static void test_asin_acos(void) {
    printf("--- asin / acos ---\n");
    float32x4_t x = (float32x4_t){ 0.0f, 0.5f, 1.0f, -0.5f };
    float32x4_t a = mc_asin_ps(x);
    CHECK(rel_err(get_lane(a, 0), 0.000000f)      < TOL, "asin(0)");
    CHECK(rel_err(get_lane(a, 1), 0.523599f)       < TOL, "asin(0.5)");
    CHECK(rel_err(get_lane(a, 2), 1.570796f)       < TOL, "asin(1)");
    CHECK(rel_err(get_lane(a, 3), -0.523599f)      < TOL, "asin(-0.5)");

    float32x4_t c = mc_acos_ps(x);
    CHECK(rel_err(get_lane(c, 0), 1.570796f)       < TOL, "acos(0)");
    CHECK(rel_err(get_lane(c, 1), 1.047198f)       < TOL, "acos(0.5)");
    CHECK(rel_err(get_lane(c, 2), 0.000000f)       < TOL, "acos(1)");
    CHECK(rel_err(get_lane(c, 3), 2.094395f)       < TOL, "acos(-0.5)");
}

static void test_atan(void) {
    printf("--- atan / atan2 ---\n");
    float32x4_t x = (float32x4_t){ 0.0f, 1.0f, -1.0f, 100.0f };
    float32x4_t a = mc_atan_ps(x);
    CHECK(rel_err(get_lane(a, 0), 0.000000f)       < TOL, "atan(0)");
    CHECK(rel_err(get_lane(a, 1), 0.785398f)       < TOL, "atan(1)");
    CHECK(rel_err(get_lane(a, 2), -0.785398f)      < TOL, "atan(-1)");
    CHECK(rel_err(get_lane(a, 3), 1.560797f)       < TOL, "atan(100)");

    float32x4_t y = (float32x4_t){ 1.0f, 1.0f, -1.0f, 0.0f };
    float32x4_t x2 = (float32x4_t){ 1.0f, 0.0f, -1.0f, -1.0f };
    float32x4_t at2 = mc_atan2_ps(y, x2);
    CHECK(rel_err(get_lane(at2, 0),  0.785398f) < TOL, "atan2(1, 1)");
    CHECK(rel_err(get_lane(at2, 1),  1.570796f) < TOL, "atan2(1, 0)");
    CHECK(rel_err(get_lane(at2, 2), -2.356194f) < TOL, "atan2(-1, -1)");
    CHECK(rel_err(get_lane(at2, 3),  3.141593f) < TOL, "atan2(0, -1)");
}

static void test_sinh_cosh_tanh(void) {
    printf("--- sinh / cosh / tanh ---\n");
    float32x4_t x = (float32x4_t){ 0.0f, 0.5f, -0.5f, 1.0f };

    float32x4_t s = mc_sinh_ps(x);
    CHECK(rel_err(get_lane(s, 0), 0.000000f) < TOL, "sinh(0)");
    CHECK(rel_err(get_lane(s, 3), 1.175201f) < TOL, "sinh(1)");

    float32x4_t c = mc_cosh_ps(x);
    CHECK(rel_err(get_lane(c, 0), 1.000000f) < TOL, "cosh(0)");
    CHECK(rel_err(get_lane(c, 3), 1.543081f) < TOL, "cosh(1)");

    float32x4_t t = mc_tanh_ps(x);
    CHECK(rel_err(get_lane(t, 0), 0.000000f) < TOL, "tanh(0)");
    CHECK(rel_err(get_lane(t, 3), 0.761594f) < TOL, "tanh(1)");
}

static void test_pow(void) {
    printf("--- pow ---\n");
    float32x4_t x = (float32x4_t){ 2.0f, 3.0f, 10.0f, 4.0f };
    float32x4_t y = (float32x4_t){ 3.0f, 2.0f, 0.0f, 0.5f };
    float32x4_t p = mc_pow_ps(x, y);
    CHECK(rel_err(get_lane(p, 0),  8.0f) < TOL, "pow(2, 3)");
    CHECK(rel_err(get_lane(p, 1),  9.0f) < TOL, "pow(3, 2)");
    CHECK(rel_err(get_lane(p, 2),  1.0f) < TOL, "pow(10, 0)");
    CHECK(rel_err(get_lane(p, 3),  2.0f) < TOL, "pow(4, 0.5)");
}

static void test_sqrt_rsqrt(void) {
    printf("--- sqrt / rsqrt ---\n");
    float32x4_t x = (float32x4_t){ 4.0f, 16.0f, 1.0f, 0.25f };
    float32x4_t s = mc_sqrt_ps(x);
    CHECK(rel_err(get_lane(s, 0), 2.0f) < TOL, "sqrt(4)");
    CHECK(rel_err(get_lane(s, 1), 4.0f) < TOL, "sqrt(16)");
    CHECK(rel_err(get_lane(s, 2), 1.0f) < TOL, "sqrt(1)");
    CHECK(rel_err(get_lane(s, 3), 0.5f) < TOL, "sqrt(0.25)");

    float32x4_t r = mc_rsqrt_ps(x);
    CHECK(rel_err(get_lane(r, 0), 0.5f) < 2e-3f, "rsqrt(4)");
    CHECK(rel_err(get_lane(r, 1), 0.25f) < 2e-3f, "rsqrt(16)");
}

/* ---------------------------------------------------------------- */
int main(void) {
    printf("=== mc_simd_math_neon — Sanity Test ===\n");
    printf("CPU supports NEON: %s\n\n", mc_supports_neon() ? "yes" : "no");

    if (!mc_supports_neon()) {
        printf("NEON not available. Exiting.\n");
        return 1;
    }

    test_foundation();
    test_exp();
    test_sincos();
    test_tan();
    test_asin_acos();
    test_atan();
    test_sinh_cosh_tanh();
    test_pow();
    test_sqrt_rsqrt();

    printf("\n=== %d failures ===\n", g_fails);
    return g_fails ? 1 : 0;
}