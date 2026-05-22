/**
 * @file example.c
 * @brief Minimal usage example for mc_simd_math.
 *
 * Builds a buffer of angles, computes sin and cos for all of them in batches
 * of 8, and prints the first few results. Demonstrates the typical access
 * pattern: load -> compute -> store, with the CPU gate at startup.
 *
 * Compile (any of):
 *   gcc   -O2 -mavx2 -mfma example.c -o example -lm
 *   clang -O2 -mavx2 -mfma example.c -o example -lm
 *   cl    /O2 /arch:AVX2   example.c
 */

#include <stdio.h>
#include "mc_simd_math.h"

#define N 64    /* must be a multiple of 8 */

int main(void) {
    if (!mc_supports_avx2_fma()) {
        fputs("This program requires a CPU with AVX2 + FMA support.\n", stderr);
        return 1;
    }

    /* Build N evenly-spaced angles across [0, 2*pi). */
    float angles[N], sin_out[N], cos_out[N];
    for (int i = 0; i < N; i++) {
        angles[i] = (float)i * (6.28318530718f / (float)N);
    }

    /* Vectorised sweep: 8 lanes per iteration. */
    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_loadu_ps(angles + i);
        __m256 s, c;
        mc_sincos_ps(a, &s, &c);
        _mm256_storeu_ps(sin_out + i, s);
        _mm256_storeu_ps(cos_out + i, c);
    }

    /* Spot-check the first quarter turn. */
    printf("  i    angle      sin       cos     sin^2+cos^2\n");
    for (int i = 0; i < 16; i++) {
        float s = sin_out[i], c = cos_out[i];
        printf("  %2d  %+6.3f   %+6.3f   %+6.3f   %.6f\n",
               i, angles[i], s, c, s*s + c*c);
    }

    return 0;
}
