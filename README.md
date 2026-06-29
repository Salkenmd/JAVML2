# JAVML2 — Just Another Vectorized Math Library 2

Single-header AVX2 + FMA math for C, aimed at gamedev.
Game-grade precision (~1-5 ULP). See benchmarks for speed. This is not exactly the best library you would want. More of a mini project ig. Please don't slap this anywhere you want. Review and refine to your tastes.

## What's in it

- **14 transcendentals**: `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
  `exp`, `exp2`, `ln`, `sinh`, `cosh`, `tanh`, `pow`.
- **Sqrt family**: `sqrt`, three precision tiers of `rsqrt`, plus SoA-batch `normalize`.
- **27 foundation utilities**: `abs`, `min`/`max`, `clamp`, `saturate`, `lerp`,
  `inverse_lerp`, `step`, `smoothstep`, `smootherstep`, `sign`, `floor`, `ceil`,
  `round`, `trunc`, `fract`, `mod`, `wrap`, `remap`, `pingpong`, `wrap_angle`,
  `to_radians`, `to_degrees`, `copysign`, `select`, `is_nan`, `is_inf`, `is_finite`,
  `approx_equal`.
- **CPU feature gate**: `mc_supports_avx2_fma()` — checks CPU + OS-level AVX state
  so you can bail safely on unsupported machines.

All functions operate on `__m256` (8-wide float).
Polynomials are minimax fits from Sollya 8.0; see the header for fit accuracy per function.

## Quick start

```c
#include "mc_simd_math.h"

int main(void) {
    if (!mc_supports_avx2_fma()) {
        /* CPU or OS doesn't support AVX2 + FMA — bail out. */
        return 1;
    }

    float in[8]  = {0, 1, 2, 3, 4, 5, 6, 7};
    float out[8];

    __m256 v = _mm256_loadu_ps(in);
    _mm256_storeu_ps(out, mc_sin_ps(v));
    return 0;
}
```

## Build

```
gcc   -O2 -mavx2 -mfma  your_file.c -lm
clang -O2 -mavx2 -mfma  your_file.c -lm
cl    /O2 /arch:AVX2    your_file.c
```

**Important:** compile any TU that calls `mc_supports_avx2_fma()` (and only that
function) with baseline flags — *without* `-mavx2`. Otherwise the compiler can
emit AVX instructions before the check runs, and the program faults on
unsupported hardware. Put the gate in a separate `main.c` from the rest.

## Accuracy (measured ULP error vs double libm)

| Function | Normal | Moderate | Extreme |
|---|---:|---:|---:|
| sin, cos | 2 | 2 | 34-110 @ \|x\| ≥ 10⁵ |
| tan | 2-3 | 3 | 3 |
| asin, acos | 3-4 | 2-3 | 3-4 |
| atan, atan2 | 1-3 | 2-3 | 1 |
| exp, exp2 | 1-2 | 2-3 | 2-5 |
| ln | 3 | 2 | 1 |
| sinh, cosh, tanh | 1-2 | 2-5 | 4-5 |
| pow | 10 (\|y\| ≤ 3) | 66 (\|y\| ≤ 10) | scales with \|y\| |

`sin`/`cos` precision drops at huge angles because the two-part Cody-Waite reduction
loses bits; the polynomial is fine. If you need correct behaviour at `|x| > 10⁵`,
swap in Payne-Hanek reduction — the rest of the code stays the same.

See [`test_transcendentals.c`](test_transcendentals.c) for the full report.

## Throughput

Measured on Intel i7-11800H, GCC 13.2, `-O3 -mavx2 -mfma`:

| Function | ns/elem | Meval/s | vs scalar libm |
|---|---:|---:|---:|
| sin / cos | 0.4 | ~2 500 | ~95× |
| exp | 0.36 | 2 800 | ~130× |
| pow | 1.1 | 907 | ~67× |

Full benchmark in [`BENCHMARK.md`](BENCHMARK.md).
Most functions land at ~1.5 cycles per element — load/store bandwidth bound, not
compute bound.

## Layout

- `mc_simd_math.h` — the whole library, one file. Vendor it however you like.
- `test_transcendentals.c` — ULP report harness.
- `bench_transcendentals.c` — throughput benchmark.
- `BENCHMARK.md` — benchmark writeup.

## Caveats

- **Float only.** No `double` path.
- **`pow(x, y)`** error scales with `|y|` — fine for typical material/lighting
  exponents (≤10), degrades for huge `y`. See header notes.
- **`x ≤ 0` for `ln`/`pow`** returns `-inf` / non-finite, not NaN. Mask at the
  call site if you need strict IEEE.
- **Windows MinGW** had a `__m256`-by-value ABI quirk during development —
  the library uses unaligned stores in `mc_sincos_ps` to side-step it.

## License

MIT. See `LICENSE`.

This is "just another" library — built for curiosity and to learn how the math
is actually done under the hood. It works, it's tested, and it's faster than
scalar `libm`. Use it if you find it useful; don't take it too seriously.
