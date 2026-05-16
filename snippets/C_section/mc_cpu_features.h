/**
 * @file mc_cpu_features.h
 * @brief Runtime CPU feature detection for AVX2 + FMA gating.
 *
 * Before calling any mc_*_ps function, verify the host CPU and OS support
 * the instruction set. Calling AVX2/FMA intrinsics on an unsupported CPU
 * raises SIGILL; running them on an OS that doesn't preserve YMM state
 * across context switches silently corrupts results.
 *
 * @par Typical use
 * @code
 *   if (!mc_supports_avx2_fma()) {
 *       fputs("This build requires AVX2 + FMA.\n", stderr);
 *       return 1;
 *   }
 *   // ... safe to call mc_sin_ps, mc_exp_ps, etc.
 * @endcode
 *
 * @note Header-only. No allocations, no globals, no caching. The check
 *       is ~50 cycles and only needs to run once at program start.
 */

#ifndef MC_CPU_FEATURES_H
#define MC_CPU_FEATURES_H

#include <stdint.h>

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
      /* `xgetbv` opcode: 0F 01 D0. Inline-encoded so the file compiles
       * even when the toolchain is not invoked with -mxsave. */
      __asm__ volatile (".byte 0x0f, 0x01, 0xd0"
                        : "=a"(eax), "=d"(edx)
                        : "c"(idx));
      return ((uint64_t)edx << 32) | (uint64_t)eax;
  }
#endif

/**
 * @brief Detailed feature flags. Each field is 0 or 1.
 */
typedef struct {
    int has_osxsave;       /**< OS uses XSAVE for extended state (precondition). */
    int os_ymm_enabled;    /**< OS has enabled YMM state preservation (XCR0 bits 1+2). */
    int has_sse2;
    int has_sse4_1;
    int has_avx;
    int has_avx2;
    int has_fma;
} mc_cpu_info_t;

/**
 * @brief Populate a full feature snapshot.
 *
 * Safe to call on any x86-64 CPU. Probes CPUID leaves 1 and 7.
 */
static inline mc_cpu_info_t mc_detect_cpu(void) {
    mc_cpu_info_t info = {0};
    int r[4];

    /* Leaf 0: max supported leaf in EAX. */
    mc_i_cpuid(0, 0, r);
    int max_leaf = r[0];
    if (max_leaf < 1) return info;

    /* Leaf 1: classic feature flags. */
    mc_i_cpuid(1, 0, r);
    int ecx = r[2], edx = r[3];
    info.has_sse2     = (edx >> 26) & 1;
    info.has_sse4_1   = (ecx >> 19) & 1;
    info.has_fma      = (ecx >> 12) & 1;
    info.has_osxsave  = (ecx >> 27) & 1;
    info.has_avx      = (ecx >> 28) & 1;

    /* OS must actually be saving YMM state, else AVX faults on context switch. */
    if (info.has_osxsave) {
        uint64_t xcr0 = mc_i_xgetbv(0);
        info.os_ymm_enabled = ((xcr0 & 0x6) == 0x6);   /* SSE (bit 1) + AVX (bit 2) */
    }

    /* Leaf 7 sub 0: extended features incl. AVX2. */
    if (max_leaf >= 7) {
        mc_i_cpuid(7, 0, r);
        info.has_avx2 = (r[1] >> 5) & 1;
    }

    return info;
}

/**
 * @brief One-shot gate: returns 1 iff AVX2 + FMA are safe to use.
 *
 * Checks the full chain: CPU exposes AVX, AVX2, FMA, OSXSAVE; and the OS
 * has enabled YMM state preservation. Failing any of these means calling
 * an mc_*_ps function will crash or silently corrupt.
 */
static inline int mc_supports_avx2_fma(void) {
    mc_cpu_info_t i = mc_detect_cpu();
    return i.has_avx
        && i.has_avx2
        && i.has_fma
        && i.has_osxsave
        && i.os_ymm_enabled;
}

#endif /* MC_CPU_FEATURES_H */
