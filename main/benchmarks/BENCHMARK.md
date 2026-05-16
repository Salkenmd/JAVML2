# `mc_simd_math` — Throughput Benchmark

AVX2 implementation vs scalar libm (`sinf`, `cosf`, …) for the full transcendental set.

## Host

| | |
|---|---|
| CPU | Intel Core i7-11800H (Tiger Lake, 11th gen) |
| Base clock | 2.30 GHz (boost higher under load) |
| Compiler | MinGW-W64 GCC 13.2.0 (UCRT, SEH) |
| Flags | `-O3 -mavx2 -mfma -fno-math-errno` |
| OS | Windows 11 |

## Method

- 65 536 floats per pass, 2 000 passes per function (~131 M evaluations total).
- Inputs randomised once and held constant across both implementations.
- Each loop accumulates results into a volatile sink to defeat dead-code elimination.
- Per-element nanoseconds = total wallclock / total elements; Meval/s = the inverse expressed in millions.
- Single-threaded. No SMT pinning; representative of one-core throughput.

Source: [bench_transcendentals.c](bench_transcendentals.c).

## Results

| Function | AVX2 ns/elem | AVX2 Meval/s | Scalar ns/elem | Scalar Meval/s | Speedup |
|---|---:|---:|---:|---:|---:|
| `sin`   | 0.383 | 2 612 | 37.23 |   26.9 |  **97×** |
| `cos`   | 0.418 | 2 391 | 37.58 |   26.6 |  **90×** |
| `tan`   | 0.385 | 2 597 | 12.75 |   78.5 |  **33×** |
| `asin`  | 0.396 | 2 526 | 147.83 |  6.8 | **373×** |
| `acos`  | 0.436 | 2 291 | 147.33 |  6.8 | **338×** |
| `atan`  | 0.418 | 2 392 | 34.93 |   28.6 |  **84×** |
| `atan2` | 0.668 | 1 497 | 36.96 |   27.1 |  **55×** |
| `exp`   | 0.357 | 2 804 | 45.87 |   21.8 | **129×** |
| `exp2`  | 0.319 | 3 139 | 32.02 |   31.2 | **101×** |
| `ln`    | 0.463 | 2 161 | 25.03 |   40.0 |  **54×** |
| `sinh`  | 0.591 | 1 692 |  9.58 |  104.4 |  **16×** |
| `cosh`  | 0.551 | 1 814 |  8.43 |  118.7 |  **15×** |
| `tanh`  | 0.576 | 1 735 | 13.43 |   74.5 |  **23×** |
| `pow`   | 1.103 |   907 | 73.60 |   13.6 |  **67×** |

## Interpretation

Two effects compose to produce these speedups:

1. **Lane width (~8×).** AVX2 processes 8 floats per instruction; scalar processes 1.
2. **Algorithm complexity (variable).** The library here targets ~1-5 ULP game-grade
   precision via short minimax polynomials. Scalar `libm` is typically correctly-rounded
   double-precision internally with the result cast back to float, plus errno/domain
   handling — significantly heavier work per element.

The cleanest single-number reads of the lane-width win are `sinh`/`cosh` at ~15×,
because there `libm` is itself just `(exp(x) ± exp(-x))/2` — the algorithm gap is
narrow and almost all of the speedup is SIMD width plus stripped overhead.

The `asin`/`acos` numbers (~340×) are eye-watering because MinGW's scalar `asinf` is
unusually slow — it appears to call full double-precision `asin` and cast. Most of
that factor is *libm being slow*, not *AVX2 being fast*; treat it as illustrative
rather than canonical.

`pow` at 67× is in line with `exp(y · ln(x))` composition cost.

## Sanity check: absolute throughput

Most AVX2 results land at **0.3–0.5 ns/elem**, which at ~4 GHz boost is roughly
1.2–2.0 cycles per element. That's load/store-bandwidth-bound rather than
compute-bound on this hardware — the FMA pipeline is being fed faster than the
polynomial chain itself can retire results. Inlining at the call site
(no separate input/output buffer round-trip) would shave a touch more.

## What about SSE?

A clean port of the same algorithms to `__m128` (4 lanes) would run at roughly
**55 % of the AVX2 throughput** on Skylake/Tiger Lake/Zen, because FMA throughput
is identical at both widths (2 FMAs / cycle / port) — the only loss is lane count.
Projected, an SSE port would still beat scalar `libm` by ~50–200×; the AVX2-over-SSE
margin is therefore on the order of 1.8×, not the kind of gap that justifies an
SSE rewrite if AVX2 is available.

## Caveats

- Accuracy and speed coexist: see [test ULP report](test_transcendentals.c) for
  the precision side of the picture.
- Numbers will shift on other µarchs. AMD Zen 3/4 should be within 10–15 % of
  these. Older Haswell/Broadwell will be lower due to FMA throughput limits.
- Re-running on a thermally throttled or battery-power laptop will look slower;
  pin to performance mode for repeatable numbers.
