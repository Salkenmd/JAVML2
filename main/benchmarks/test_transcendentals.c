/**
 * @file test_transcendentals.c
 * @brief ULP-error harness for the mc_* AVX2 transcendentals.
 *
 * For each function, sweeps inputs over a "normal" working range and an
 * "extreme" stress range, comparing against double-precision libm cast
 * back to float. Reports max and mean ULP error per range, plus a small
 * battery of named edge cases.
 *
 * Compile (MinGW / Clang / GCC):
 *   gcc -O2 -mavx2 -mfma test_transcendentals.c -o test_transcendentals.exe -lm
 *
 * Compile (MSVC):
 *   cl /O2 /arch:AVX2 test_transcendentals.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <immintrin.h>

#include "SIMD_sincos_ps.h"
#include "SIMD_tan_ps.h"
#include "SIMD_atan_ps.h"
#include "SIMD_asin_acos_ps.h"
#include "SIMD_explog_ps.h"
#include "SIMD_hyperbolic_ps.h"
#include "SIMD_pow_ps.h"

/* ============================================================
 * ULP distance
 * ============================================================ */

typedef union { float f; int32_t i; uint32_t u; } f32u;

/**
 * Integer ULP distance between two float32 values.
 *
 * Returns INT64_MAX for any infinity/NaN disagreement (so it propagates
 * into the "max" but we exclude it from the mean).
 */
static int64_t ulp_dist(float a, float b) {
    if (isnan(a) && isnan(b)) return 0;
    if (isnan(a) || isnan(b)) return INT64_MAX;
    if (a == b)               return 0;
    if (isinf(a) || isinf(b)) {
        return (isinf(a) && isinf(b) && (a < 0) == (b < 0)) ? 0 : INT64_MAX;
    }
    f32u ua, ub; ua.f = a; ub.f = b;
    int64_t ai = ua.i, bi = ub.i;
    if (ai < 0) ai = (int64_t)0x80000000 - ai;
    if (bi < 0) bi = (int64_t)0x80000000 - bi;
    int64_t d = ai - bi;
    return d < 0 ? -d : d;
}

/* ============================================================
 * Wrappers — take/return float arrays to dodge MinGW's Windows-x64
 * ABI bug with passing __m256 by value across function pointers.
 * Each wrapper processes one 8-lane chunk.
 * ============================================================ */

typedef void (*fn1_t)(const float*, float*);
typedef void (*fn2_t)(const float*, const float*, float*);

#define WRAP1(name) static void w_##name(const float* x, float* r) { \
    _mm256_storeu_ps(r, mc_##name##_ps(_mm256_loadu_ps(x))); }
#define WRAP2(name) static void w_##name(const float* a, const float* b, float* r) { \
    _mm256_storeu_ps(r, mc_##name##_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b))); }

WRAP1(sin)   WRAP1(cos)   WRAP1(tan)
WRAP1(asin)  WRAP1(acos)  WRAP1(atan)
WRAP1(exp)   WRAP1(exp2)  WRAP1(ln)
WRAP1(sinh)  WRAP1(cosh)  WRAP1(tanh)
WRAP2(atan2) WRAP2(pow)

/* libm reference wrappers — use double for headroom, cast back at the end. */
static double r_sin (double x)            { return sin (x); }
static double r_cos (double x)            { return cos (x); }
static double r_tan (double x)            { return tan (x); }
static double r_asin(double x)            { return asin(x); }
static double r_acos(double x)            { return acos(x); }
static double r_atan(double x)            { return atan(x); }
static double r_exp (double x)            { return exp (x); }
static double r_exp2(double x)            { return exp2(x); }
static double r_ln  (double x)            { return log (x); }
static double r_sinh(double x)            { return sinh(x); }
static double r_cosh(double x)            { return cosh(x); }
static double r_tanh(double x)            { return tanh(x); }
static double r_atan2(double y, double x) { return atan2(y, x); }
static double r_pow  (double x, double y) { return pow(x, y); }

/* ============================================================
 * Sweep drivers
 * ============================================================ */

#define NSWEEP 65536  /* must be multiple of 8 */

static float  in_buf[NSWEEP];
static float  in2_buf[NSWEEP];
static float  out_buf[NSWEEP];
static float  ref_buf[NSWEEP];

static void linspace(float* a, int n, float lo, float hi) {
    for (int i = 0; i < n; i++) {
        a[i] = lo + (hi - lo) * ((float)i / (float)(n - 1));
    }
}

static void report(const char* name, const char* tag,
                   const float* x_in, const float* y_in, int n)
{
    int64_t max_ulp = 0;
    double  sum_ulp = 0.0;
    int     counted = 0;
    int     worst_i = 0;

    for (int i = 0; i < n; i++) {
        int64_t d = ulp_dist(out_buf[i], ref_buf[i]);
        if (d == INT64_MAX) continue;
        if (d > max_ulp) { max_ulp = d; worst_i = i; }
        sum_ulp += (double)d;
        counted++;
    }

    double mean = counted ? sum_ulp / (double)counted : 0.0;
    if (y_in) {
        printf("    %-10s %-10s  max=%6lld  mean=%5.2f  "
               "worst @(x=%+.6e, y=%+.6e): got %+.6e ref %+.6e\n",
               name, tag, (long long)max_ulp, mean,
               x_in[worst_i], y_in[worst_i],
               out_buf[worst_i], ref_buf[worst_i]);
    } else {
        printf("    %-10s %-10s  max=%6lld  mean=%5.2f  "
               "worst @x=%+.6e: got %+.6e ref %+.6e\n",
               name, tag, (long long)max_ulp, mean,
               x_in[worst_i], out_buf[worst_i], ref_buf[worst_i]);
    }
}

static void sweep_unary(const char* name, fn1_t f_simd,
                        double (*f_ref)(double),
                        const char* tag, float lo, float hi)
{
    linspace(in_buf, NSWEEP, lo, hi);
    for (int i = 0; i < NSWEEP; i += 8) {
        f_simd(in_buf + i, out_buf + i);
    }
    for (int i = 0; i < NSWEEP; i++) {
        ref_buf[i] = (float)f_ref((double)in_buf[i]);
    }
    report(name, tag, in_buf, NULL, NSWEEP);
}

static void sweep_binary(const char* name, fn2_t f_simd,
                         double (*f_ref)(double, double),
                         const char* tag,
                         float xlo, float xhi, float ylo, float yhi)
{
    for (int i = 0; i < NSWEEP; i++) {
        float tx = (float)i / (float)(NSWEEP - 1);
        float ty = (float)((i * 7919) % NSWEEP) / (float)(NSWEEP - 1);
        in_buf [i] = xlo + (xhi - xlo) * tx;
        in2_buf[i] = ylo + (yhi - ylo) * ty;
    }
    for (int i = 0; i < NSWEEP; i += 8) {
        f_simd(in_buf + i, in2_buf + i, out_buf + i);
    }
    for (int i = 0; i < NSWEEP; i++) {
        ref_buf[i] = (float)f_ref((double)in_buf[i], (double)in2_buf[i]);
    }
    report(name, tag, in_buf, in2_buf, NSWEEP);
}

/* ============================================================
 * Edge case spot checks
 * ============================================================ */

static void edge_unary(const char* name, fn1_t f_simd,
                       double (*f_ref)(double),
                       const float* xs, int n)
{
    printf("    %s edges:\n", name);
    float xbuf[8], obuf[8];
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 8; k++) xbuf[k] = xs[i];
        f_simd(xbuf, obuf);
        float got = obuf[0];
        float ref = (float)f_ref((double)xs[i]);
        int64_t d = ulp_dist(got, ref);
        const char* tag = (d == INT64_MAX) ? "INF/NAN" : (d <= 4 ? "OK" : "WIDE");
        printf("      %-7s f(%+.7e) = %+.8e  ref %+.8e  (%lld ULP)\n",
               tag, xs[i], got, ref, (long long)d);
    }
}

static void edge_binary(const char* name, fn2_t f_simd,
                        double (*f_ref)(double, double),
                        const float (*xy)[2], int n)
{
    printf("    %s edges:\n", name);
    float xbuf[8], ybuf[8], obuf[8];
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 8; k++) { xbuf[k] = xy[i][0]; ybuf[k] = xy[i][1]; }
        f_simd(xbuf, ybuf, obuf);
        float got = obuf[0];
        float ref = (float)f_ref((double)xy[i][0], (double)xy[i][1]);
        int64_t d = ulp_dist(got, ref);
        const char* tag = (d == INT64_MAX) ? "INF/NAN" : (d <= 4 ? "OK" : "WIDE");
        printf("      %-7s f(%+.7e, %+.7e) = %+.8e  ref %+.8e  (%lld ULP)\n",
               tag, xy[i][0], xy[i][1], got, ref, (long long)d);
    }
}

/* ============================================================
 * main
 * ============================================================ */

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: localize crashes */
    const float PI    = 3.14159265358979323846f;
    const float PI_2  = 1.57079632679489661923f;
    const float PI_4  = 0.78539816339744830962f;

    puts("=== Sweep tests (normal + extreme ranges) ===\n");

    puts("[sin]");
    sweep_unary("sin", w_sin, r_sin, "normal",   -PI,        PI);
    sweep_unary("sin", w_sin, r_sin, "moderate", -100.0f,    100.0f);
    sweep_unary("sin", w_sin, r_sin, "extreme",  -1.0e5f,    1.0e5f);

    puts("\n[cos]");
    sweep_unary("cos", w_cos, r_cos, "normal",   -PI,        PI);
    sweep_unary("cos", w_cos, r_cos, "moderate", -100.0f,    100.0f);
    sweep_unary("cos", w_cos, r_cos, "extreme",  -1.0e5f,    1.0e5f);

    puts("\n[tan]");
    /* Avoid sweeping straight through pi/2 singularities — bracket them. */
    sweep_unary("tan", w_tan, r_tan, "near0",    -1.0f,      1.0f);
    sweep_unary("tan", w_tan, r_tan, "below",    -PI_2*0.95f, PI_2*0.95f);
    sweep_unary("tan", w_tan, r_tan, "moderate", -10.0f,     10.0f);

    puts("\n[asin]");
    sweep_unary("asin", w_asin, r_asin, "core",  -1.0f,      1.0f);
    sweep_unary("asin", w_asin, r_asin, "small", -0.5f,      0.5f);
    sweep_unary("asin", w_asin, r_asin, "near1", -0.9999f,   0.9999f);

    puts("\n[acos]");
    sweep_unary("acos", w_acos, r_acos, "core",  -1.0f,      1.0f);
    sweep_unary("acos", w_acos, r_acos, "small", -0.5f,      0.5f);
    sweep_unary("acos", w_acos, r_acos, "near1", -0.9999f,   0.9999f);

    puts("\n[atan]");
    sweep_unary("atan", w_atan, r_atan, "near0",   -1.0f,    1.0f);
    sweep_unary("atan", w_atan, r_atan, "moderate",-100.0f,  100.0f);
    sweep_unary("atan", w_atan, r_atan, "extreme", -1e6f,    1e6f);

    puts("\n[atan2]");
    sweep_binary("atan2", w_atan2, r_atan2, "unit",     -1.0f, 1.0f, -1.0f, 1.0f);
    sweep_binary("atan2", w_atan2, r_atan2, "moderate", -50.0f, 50.0f, -50.0f, 50.0f);

    puts("\n[exp]");
    sweep_unary("exp", w_exp, r_exp, "core",     -10.0f,   10.0f);
    sweep_unary("exp", w_exp, r_exp, "moderate", -50.0f,   50.0f);
    sweep_unary("exp", w_exp, r_exp, "extreme",  -87.0f,   88.0f);

    puts("\n[exp2]");
    sweep_unary("exp2", w_exp2, r_exp2, "core",     -10.0f, 10.0f);
    sweep_unary("exp2", w_exp2, r_exp2, "moderate", -64.0f, 64.0f);
    sweep_unary("exp2", w_exp2, r_exp2, "extreme",  -125.0f, 126.0f);

    puts("\n[ln]");
    sweep_unary("ln", w_ln, r_ln, "near1",    0.5f,       2.0f);
    sweep_unary("ln", w_ln, r_ln, "moderate", 1e-3f,      1e3f);
    sweep_unary("ln", w_ln, r_ln, "extreme",  1e-30f,     1e30f);

    puts("\n[sinh]");
    sweep_unary("sinh", w_sinh, r_sinh, "near0",   -0.3f,   0.3f);
    sweep_unary("sinh", w_sinh, r_sinh, "moderate",-10.0f,  10.0f);
    sweep_unary("sinh", w_sinh, r_sinh, "extreme", -80.0f,  80.0f);

    puts("\n[cosh]");
    sweep_unary("cosh", w_cosh, r_cosh, "near0",   -0.3f,   0.3f);
    sweep_unary("cosh", w_cosh, r_cosh, "moderate",-10.0f,  10.0f);
    sweep_unary("cosh", w_cosh, r_cosh, "extreme", -80.0f,  80.0f);

    puts("\n[tanh]");
    sweep_unary("tanh", w_tanh, r_tanh, "near0",   -0.3f,   0.3f);
    sweep_unary("tanh", w_tanh, r_tanh, "moderate",-5.0f,   5.0f);
    sweep_unary("tanh", w_tanh, r_tanh, "extreme", -20.0f,  20.0f);

    puts("\n[pow]");
    sweep_binary("pow", w_pow, r_pow, "core",     0.1f,  10.0f, -3.0f, 3.0f);
    sweep_binary("pow", w_pow, r_pow, "moderate", 0.01f, 100.0f, -10.0f, 10.0f);

    /* ============================================================
     * Edge cases
     * ============================================================ */
    puts("\n=== Edge cases ===\n");

    {
        float xs[] = { 0.0f, PI_4, PI_2, PI, 2.0f*PI, -PI_2, 100.0f, 1.0e4f };
        edge_unary("sin", w_sin, r_sin, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { 0.0f, PI_4, PI_2, PI, 2.0f*PI, -PI_2, 100.0f, 1.0e4f };
        edge_unary("cos", w_cos, r_cos, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { 0.0f, PI_4, -PI_4, PI/8.0f, PI_2*0.99f, -PI_2*0.99f };
        edge_unary("tan", w_tan, r_tan, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f /* clamped */ };
        edge_unary("asin", w_asin, r_asin, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f /* clamped */ };
        edge_unary("acos", w_acos, r_acos, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { 0.0f, 1.0f, -1.0f, 1e6f, -1e6f, 0.4142f, -0.4142f };
        edge_unary("atan", w_atan, r_atan, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { 0.0f, 1.0f, -1.0f, 10.0f, -10.0f, 80.0f, -80.0f };
        edge_unary("exp", w_exp, r_exp, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { 1.0f, 2.0f, 0.5f, (float)M_E, 1e-10f, 1e10f };
        edge_unary("ln", w_ln, r_ln, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xs[] = { 0.0f, 0.5f, 1.0f, -1.0f, 5.0f, -5.0f, 50.0f };
        edge_unary("sinh", w_sinh, r_sinh, xs, (int)(sizeof xs / sizeof *xs));
        edge_unary("cosh", w_cosh, r_cosh, xs, (int)(sizeof xs / sizeof *xs));
        edge_unary("tanh", w_tanh, r_tanh, xs, (int)(sizeof xs / sizeof *xs));
    }
    {
        float xy[][2] = {
            { 1.0f,  1.0f }, { -1.0f,  1.0f }, { 1.0f, -1.0f }, { -1.0f, -1.0f },
            { 0.0f,  1.0f }, {  0.0f, -1.0f }, { 1.0f,  0.0f }, { -1.0f,  0.0f },
            { 1e-6f, 1.0f }, { 1.0f,   1e-6f }
        };
        edge_binary("atan2", w_atan2, r_atan2, xy, (int)(sizeof xy / sizeof *xy));
    }
    {
        float xy[][2] = {
            { 2.0f,  3.0f }, { 2.0f, -3.0f }, { 0.5f, 10.0f },
            { 10.0f, 0.5f }, { 1.0f, 100.0f }, { (float)M_E, 1.0f }
        };
        edge_binary("pow", w_pow, r_pow, xy, (int)(sizeof xy / sizeof *xy));
    }

    puts("\n=== Done ===");
    return 0;
}
