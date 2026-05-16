/**
 * @file bench_transcendentals.c
 * @brief Throughput benchmark: mc_*_ps (AVX2) vs scalar libm.
 *
 * Processes a large input buffer N times, reports nanoseconds per element
 * and millions of evaluations per second for each function. Uses a volatile
 * accumulator to defeat dead-code elimination.
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -fno-math-errno bench_transcendentals.c \
 *       -o bench_transcendentals.exe -lm
 *
 * @note -fno-math-errno lets the compiler vectorize libm calls in the
 *       reference loops where it legally can; this is fair because we
 *       want best-case scalar performance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#if defined(_WIN32)
  #include <windows.h>
#endif

#include "SIMD_sincos_ps.h"
#include "SIMD_tan_ps.h"
#include "SIMD_atan_ps.h"
#include "SIMD_asin_acos_ps.h"
#include "SIMD_explog_ps.h"
#include "SIMD_hyperbolic_ps.h"
#include "SIMD_pow_ps.h"

#define N_ELEMS  (1 << 16)    /* 65536 floats, fits in L2 */
#define N_REPS   2000         /* loop count for stable timing */

/* Static buffers to keep the harness off the stack and aligned to a
 * cache line (BSS gives 64-byte alignment for sufficiently sized arrays
 * on most toolchains; _mm256_loadu_ps tolerates any alignment anyway). */
static float in_a[N_ELEMS];
static float in_b[N_ELEMS];
static float out[N_ELEMS];

/* Volatile sink so the compiler can't drop the work. */
static volatile float g_sink;

/* ============================================================
 * High-resolution timer
 * ============================================================ */
static double now_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ============================================================
 * Report helper
 * ============================================================ */
static void report(const char* name, const char* impl,
                   double secs, uint64_t total_ops)
{
    double ns_per = secs * 1e9 / (double)total_ops;
    double mevs   = (double)total_ops / (secs * 1e6);
    printf("    %-8s %-7s  %8.3f ns/elem  %9.2f Meval/s\n",
           name, impl, ns_per, mevs);
}

/* ============================================================
 * Bench drivers — one per function. Macros keep it terse.
 * ============================================================ */

#define BENCH_UNARY(name, simd_call, scalar_call)                              \
    do {                                                                       \
        /* warmup */                                                           \
        for (int i = 0; i < N_ELEMS; i += 8)                                   \
            _mm256_storeu_ps(out + i, simd_call(_mm256_loadu_ps(in_a + i)));   \
        /* AVX2 */                                                             \
        double t0 = now_seconds();                                             \
        float acc_v = 0.0f;                                                    \
        __m256 vacc = _mm256_setzero_ps();                                     \
        for (int rep = 0; rep < N_REPS; rep++) {                               \
            for (int i = 0; i < N_ELEMS; i += 8) {                             \
                __m256 r = simd_call(_mm256_loadu_ps(in_a + i));               \
                vacc = _mm256_add_ps(vacc, r);                                 \
            }                                                                  \
        }                                                                      \
        float buf[8]; _mm256_storeu_ps(buf, vacc);                             \
        for (int k = 0; k < 8; k++) acc_v += buf[k];                           \
        g_sink = acc_v;                                                        \
        double t1 = now_seconds();                                             \
        /* scalar */                                                           \
        float acc_s = 0.0f;                                                    \
        for (int rep = 0; rep < N_REPS; rep++) {                               \
            for (int i = 0; i < N_ELEMS; i++) acc_s += scalar_call(in_a[i]);   \
        }                                                                      \
        g_sink = acc_s;                                                        \
        double t2 = now_seconds();                                             \
        uint64_t ops = (uint64_t)N_ELEMS * (uint64_t)N_REPS;                   \
        printf("  [%s]\n", name);                                              \
        report(name, "AVX2",   t1 - t0, ops);                                  \
        report(name, "scalar", t2 - t1, ops);                                  \
        printf("    %-8s speedup  %.2fx\n", name,                              \
               (t2 - t1) / (t1 - t0));                                         \
    } while (0)

#define BENCH_BINARY(name, simd_call, scalar_call)                             \
    do {                                                                       \
        for (int i = 0; i < N_ELEMS; i += 8)                                   \
            _mm256_storeu_ps(out + i,                                          \
                simd_call(_mm256_loadu_ps(in_a + i),                           \
                          _mm256_loadu_ps(in_b + i)));                         \
        double t0 = now_seconds();                                             \
        __m256 vacc = _mm256_setzero_ps();                                     \
        for (int rep = 0; rep < N_REPS; rep++) {                               \
            for (int i = 0; i < N_ELEMS; i += 8) {                             \
                __m256 r = simd_call(_mm256_loadu_ps(in_a + i),                \
                                     _mm256_loadu_ps(in_b + i));               \
                vacc = _mm256_add_ps(vacc, r);                                 \
            }                                                                  \
        }                                                                      \
        float buf[8]; _mm256_storeu_ps(buf, vacc);                             \
        float acc_v = 0; for (int k = 0; k < 8; k++) acc_v += buf[k];          \
        g_sink = acc_v;                                                        \
        double t1 = now_seconds();                                             \
        float acc_s = 0.0f;                                                    \
        for (int rep = 0; rep < N_REPS; rep++) {                               \
            for (int i = 0; i < N_ELEMS; i++)                                  \
                acc_s += scalar_call(in_a[i], in_b[i]);                        \
        }                                                                      \
        g_sink = acc_s;                                                        \
        double t2 = now_seconds();                                             \
        uint64_t ops = (uint64_t)N_ELEMS * (uint64_t)N_REPS;                   \
        printf("  [%s]\n", name);                                              \
        report(name, "AVX2",   t1 - t0, ops);                                  \
        report(name, "scalar", t2 - t1, ops);                                  \
        printf("    %-8s speedup  %.2fx\n", name,                              \
               (t2 - t1) / (t1 - t0));                                         \
    } while (0)

int main(void) {
    /* Reasonable working ranges — keep inputs in domain so neither
     * implementation does NaN/inf fast-path work. */
    srand(12345);
    for (int i = 0; i < N_ELEMS; i++) {
        float u = (float)rand() / (float)RAND_MAX;     /* [0, 1]  */
        float s = u * 2.0f - 1.0f;                     /* [-1, 1] */
        in_a[i] = s * 3.0f;                            /* trig-friendly */
        in_b[i] = 0.5f + u * 9.5f;                     /* pow base > 0 */
    }

    printf("=== Throughput bench (N=%d elems, %d reps) ===\n\n",
           N_ELEMS, N_REPS);

    BENCH_UNARY("sin",  mc_sin_ps,  sinf);
    BENCH_UNARY("cos",  mc_cos_ps,  cosf);
    BENCH_UNARY("tan",  mc_tan_ps,  tanf);
    BENCH_UNARY("asin", mc_asin_ps, asinf);   /* in_a in [-3,3]; clamp inside lib */
    BENCH_UNARY("acos", mc_acos_ps, acosf);
    BENCH_UNARY("atan", mc_atan_ps, atanf);
    BENCH_UNARY("exp",  mc_exp_ps,  expf);
    BENCH_UNARY("exp2", mc_exp2_ps, exp2f);
    BENCH_UNARY("ln",   mc_ln_ps,   logf);    /* in_a has negatives; lib returns -inf */
    BENCH_UNARY("sinh", mc_sinh_ps, sinhf);
    BENCH_UNARY("cosh", mc_cosh_ps, coshf);
    BENCH_UNARY("tanh", mc_tanh_ps, tanhf);

    /* Refresh in_a to log-positive for pow / ln fairness */
    for (int i = 0; i < N_ELEMS; i++) in_a[i] = 0.5f + 9.5f * ((float)rand() / RAND_MAX);

    BENCH_BINARY("atan2", mc_atan2_ps, atan2f);
    BENCH_BINARY("pow",   mc_pow_ps,   powf);

    printf("\n(sink = %g, prevents DCE)\n", (double)g_sink);
    return 0;
}
