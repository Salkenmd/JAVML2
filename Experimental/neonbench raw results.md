=== Throughput bench (N=65536 elems, 2000 reps) ===

  [sin]
    sin      NEON        5.468 ns/elem     182.87 Meval/s
    sin      scalar     22.924 ns/elem      43.62 Meval/s
    sin      speedup  4.19x
  [cos]
    cos      NEON        2.614 ns/elem     382.57 Meval/s
    cos      scalar     15.686 ns/elem      63.75 Meval/s
    cos      speedup  6.00x
  [tan]
    tan      NEON        3.333 ns/elem     300.06 Meval/s
    tan      scalar     17.115 ns/elem      58.43 Meval/s
    tan      speedup  5.14x
  [asin]
    asin     NEON        2.291 ns/elem     436.52 Meval/s
    asin     scalar     12.411 ns/elem      80.57 Meval/s
    asin     speedup  5.42x
  [acos]
    acos     NEON        2.721 ns/elem     367.48 Meval/s
    acos     scalar     12.456 ns/elem      80.29 Meval/s
    acos     speedup  4.58x
  [atan]
    atan     NEON        3.045 ns/elem     328.41 Meval/s
    atan     scalar     19.613 ns/elem      50.99 Meval/s
    atan     speedup  6.44x
  [exp]
    exp      NEON        2.242 ns/elem     445.98 Meval/s
    exp      scalar      8.100 ns/elem     123.46 Meval/s
    exp      speedup  3.61x
  [exp2]
    exp2     NEON        1.931 ns/elem     517.93 Meval/s
    exp2     scalar      6.891 ns/elem     145.13 Meval/s
    exp2     speedup  3.57x
  [ln]
    ln       NEON        3.177 ns/elem     314.72 Meval/s
    ln       scalar     11.488 ns/elem      87.05 Meval/s
    ln       speedup  3.62x
  [sinh]
    sinh     NEON        3.397 ns/elem     294.38 Meval/s
    sinh     scalar     34.600 ns/elem      28.90 Meval/s
    sinh     speedup  10.19x
  [cosh]
    cosh     NEON        3.426 ns/elem     291.85 Meval/s
    cosh     scalar     13.613 ns/elem      73.46 Meval/s
    cosh     speedup  3.97x
  [tanh]
    tanh     NEON        3.633 ns/elem     275.26 Meval/s
    tanh     scalar     37.261 ns/elem      26.84 Meval/s
    tanh     speedup  10.26x
  [atan2]
    atan2    NEON        4.015 ns/elem     249.06 Meval/s
    atan2    scalar     32.370 ns/elem      30.89 Meval/s
    atan2    speedup  8.06x
  [pow]
    pow      NEON        6.790 ns/elem     147.29 Meval/s
    pow      scalar     13.660 ns/elem      73.21 Meval/s
    pow      speedup  2.01x

(sink = 5.65492e+15, prevents DCE)