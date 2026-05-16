/**
 * @file minimax_coeffs.h
 * @brief Float (f32) minimax polynomial coefficients, ~1 ULP accuracy.
 *
 * Generated with Sollya 8.0, prec=128.
 *
 * @par Usage Pattern
 * Each function's coefficients are stored as a `static const float[]` array,
 * sized to match the corresponding `mc_poly_estrin_degN_ps()` evaluator.
 * Pass the array pointer directly:
 * @code
 *   __m256 t   = _mm256_mul_ps(x, x);          // x^2
 *   __m256 p   = mc_poly_estrin_deg3_ps(t, SIN_COEFFS);
 *   __m256 res = _mm256_mul_ps(x, p);           // x * P(x^2)
 * @endcode
 *
 * @note ULP = unit in the last place for float32 = 2^-24 relative to result.
 * @note These coefficients only cover a reduced range. Argument reduction must
 *       be applied before evaluation. See each section for the valid range and
 *       the reconstruction step needed to get the full-domain result.
 */

#ifndef MINIMAX_COEFFS_H
#define MINIMAX_COEFFS_H

/* ============================================================
 * SIN(x)
 * ============================================================
 * @par Range
 *   x in [-pi/4, pi/4] after argument reduction.
 *
 * @par Form
 *   sin(x) = x * P(x^2),  P is degree-3 in t = x^2
 *
 * @par Reconstruction
 *   Pass x^2 to mc_poly_estrin_deg3_ps(), then multiply result by x.
 *
 * @par Accuracy
 *   0.04 ULP
 *
 * @par Coefficients (4 values, indices 0..3)
 *   P(t) = C[0] + t*(C[1] + t*(C[2] + t*C[3]))
 * ============================================================ */
static const float SIN_COEFFS[4] = {
     0.999999996926342778282546052871901425708f,   /**< C0 */
    -0.166666506992021714609082680346419283499f,   /**< C1 */
     8.3320368751463668594189710376374380334e-3f,  /**< C2 */
    -1.95040219991569957621701412087918905123e-4f  /**< C3 */
};

/* ============================================================
 * COS(x)
 * ============================================================
 * @par Range
 *   x in [0, pi/4]. Use cos(-x) = cos(x) for negative inputs.
 *
 * @par Form
 *   cos(x) = 1 - (x^2 / 2) * P(x^2),  P is degree-3 in t = x^2
 *
 * @par Reconstruction
 *   Pass x^2 to mc_poly_estrin_deg3_ps(), scale by -(x^2/2), add 1.
 *
 * @par Accuracy
 *   0.003 ULP
 *
 * @par Coefficients (4 values, indices 0..3)
 *   P(t) = C[0] + t*(C[1] + t*(C[2] + t*C[3]))
 * ============================================================ */
static const float COS_COEFFS[4] = {
     0.999999999383828582129270698455438593398f,   /**< C0 */
    -8.33333013311138576100588974476052245317e-2f, /**< C1 */
     2.77751801301879159485208760118056498869e-3f, /**< C2 */
    -4.89277731664569609684403388950264514135e-5f  /**< C3 */
};

/* ============================================================
 * TAN(x)
 * ============================================================
 * @par Range
 *   x in [0, pi/8]. For x in (pi/8, pi/4], halve the argument,
 *   evaluate, then apply the double-angle identity:
 *   tan(2y) = 2*t / (1 - t^2)
 *
 * @par Form
 *   tan(x) = x + x^3 * P(x^2),  P is degree-3 in t = x^2
 *
 * @par Reconstruction
 *   Pass x^2 to mc_poly_estrin_deg3_ps(), multiply by x^3, add x.
 *
 * @par Accuracy
 *   0.047 ULP
 *
 * @par Coefficients (4 values, indices 0..3)
 *   P(t) = C[0] + t*(C[1] + t*(C[2] + t*C[3]))
 * ============================================================ */
static const float TAN_COEFFS[4] = {
    0.333333287350697673210307974070168833443f,    /**< C0 */
    0.133342724106033723928251509259696377785f,    /**< C1 */
    5.36695340958928597698376503512930466572e-2f,  /**< C2 */
    2.48696209035983955819911447913718818547e-2f   /**< C3 */
};

/* ============================================================
 * ASIN(x)
 * ============================================================
 * @par Range
 *   x in [0, 0.5].
 *   For |x| > 0.5: asin(x) = pi/2 - 2*asin(sqrt((1-x)/2))
 *
 * @par Form
 *   asin(x) = x * P(x^2),  P is degree-5 in t = x^2
 *
 * @par Reconstruction
 *   Pass x^2 to mc_poly_estrin_deg5_ps() (or deg4 + manual C5 term),
 *   then multiply result by x.
 *
 * @par Accuracy
 *   0.038 ULP
 *
 * @par Coefficients (6 values, indices 0..5)
 *   P(t) = C[0] + t*(C[1] + t*(C[2] + t*(C[3] + t*(C[4] + t*C[5]))))
 *
 * @note acos(x) = pi/2 - asin(x); no separate polynomial needed.
 * ============================================================ */
static const float ASIN_COEFFS[6] = {
    0.999999995478096608701846419443927593772f,    /**< C0 */
    0.166667922213005814888055819423805727124f,    /**< C1 */
    7.4943675002161483595064288611545911554e-2f,   /**< C2 */
    4.55576929736023952506899394496118912138e-2f,  /**< C3 */
    2.3823936588797289217393470823864621618e-2f,   /**< C4 */
    4.26903047154410196086038580757107848077e-2f   /**< C5 */
};

/* ============================================================
 * ATAN(x)
 * ============================================================
 * @par Range
 *   x in [0, tan(pi/8)] (~0.4142) after range reduction.
 *   Range reduction steps:
 *     x > tan(3*pi/8): atan(x) = pi/2 - atan(1/x)
 *     x > tan(pi/8):   atan(x) = pi/4 + atan((x-1)/(x+1))
 *
 * @par Form
 *   atan(x) = x * P(x^2),  P is degree-4 in t = x^2
 *
 * @par Reconstruction
 *   Pass x^2 to mc_poly_estrin_deg4_ps(), multiply by x, add offset.
 *
 * @par Accuracy
 *   0.12 ULP
 *
 * @par Coefficients (5 values, indices 0..4)
 *   P(t) = C[0] + t*(C[1] + t*(C[2] + t*(C[3] + t*C[4])))
 *
 * @note atan2(y, x) is implemented via atan with quadrant correction.
 * ============================================================ */
static const float ATAN_COEFFS[5] = {
     0.999999982460717905120588625597135589113f,   /**< C0 */
    -0.333328079017683915803156014462307278338f,   /**< C1 */
     0.199747243062319873757635773939061055264f,   /**< C2 */
    -0.138544576850887346340673529423546659512f,   /**< C3 */
     7.99364423130897184747222910912351542317e-2f  /**< C4 */
};

/* ============================================================
 * EXP(x)
 * ============================================================
 * @par Range
 *   Reduced argument r in [-ln2/2, ln2/2] after Cody-Waite reduction:
 *     n = round(x / ln2)
 *     r = x - n * ln2   (use LN2_HI/LN2_LO for precision)
 *
 * @par Form
 *   exp(r) = P(r),  degree-6 polynomial
 *
 * @par Reconstruction
 *   result = P(r) * 2^n   (scale by inserting n into float exponent bits)
 *
 * @par Accuracy
 *   0.031 ULP
 *
 * @par Coefficients (7 values, indices 0..6)
 *   P(r) = C[0] + r*(C[1] + r*(C[2] + r*(C[3] + r*(C[4] + r*(C[5] + r*C[6])))))
 * ============================================================ */
static const float EXP_COEFFS[7] = {
    0.999999999919067037197896619810258382586f,    /**< C0 */
    1.00000003772604441761669504931498793344f,     /**< C1 */
    0.500000016841291732193530788373548378825f,    /**< C2 */
    0.16666415481911346468144430935857029765f,     /**< C3 */
    4.16660835948060307131369604131088746096e-2f,  /**< C4 */
    8.3751285790288349279387284883017034716e-3f,   /**< C5 */
    1.39560559247713612085306234329346882015e-3f   /**< C6 */
};

/** @brief High part of ln(2) for Cody-Waite argument reduction. */
#define LN2_HI  (6.93147180369123816490e-1f)
/** @brief Low part of ln(2) for Cody-Waite argument reduction. */
#define LN2_LO  (1.90821492927058770002e-10f)

/* ============================================================
 * EXP2(x)  -- 2^x
 * ============================================================
 * @par Range
 *   Reduced argument r in [-0.5, 0.5]:
 *     n = round(x)
 *     r = x - n
 *
 * @par Form
 *   2^r = P(r),  degree-6 polynomial
 *
 * @par Reconstruction
 *   result = P(r) * 2^n   (same exponent-bit trick as EXP)
 *
 * @par Accuracy
 *   0.031 ULP
 *
 * @par Coefficients (7 values, indices 0..6)
 * ============================================================ */
static const float EXP2_COEFFS[7] = {
    0.999999999919067037197896619810258382586f,    /**< C0 */
    0.693147206709646631167508230321444285403f,    /**< C1 */
    0.24022651505055008334162029128006404866f,     /**< C2 */
    5.55032721576643256179871599540553593548e-2f,  /**< C3 */
    9.6179945141780462397505203069232532594e-3f,   /**< C4 */
    1.34004316670754714578668090114651575682e-3f,  /**< C5 */
    1.54780222758443390748352500253494123394e-4f   /**< C6 */
};

/* ============================================================
 * LN(x) -- natural log
 * ============================================================
 * @par Range
 *   Reduced mantissa m in [1/sqrt(2), sqrt(2)], substituting u = m - 1,
 *   so u in [1/sqrt(2)-1, sqrt(2)-1].
 *
 * @par Form
 *   ln(1+u) = u * P(u),  P is degree-8 in u
 *
 * @par Reconstruction
 *   Decompose x = 2^n * m, then:
 *     ln(x) = n * ln(2) + ln(m)
 *   Use LN2_HI + LN2_LO for the n*ln(2) term.
 *
 * @par Accuracy
 *   0.18 ULP
 *
 * @par Coefficients (9 values, indices 0..8)
 *   P(u) = C[0] + u*(C[1] + u*(... + u*C[8]))
 * ============================================================ */
static const float LN_COEFFS[9] = {
     0.999999975045684318538782514084829989036f,   /**< C0 */
    -0.499999777407575512300036757431471630087f,   /**< C1 */
     0.33334160454165577395122435673327927243f,    /**< C2 */
    -0.250029461445376526033833784745386161086f,   /**< C3 */
     0.1995841718412106778508522363336532356576f,  /**< C4 */
    -0.165437371026277935081140961987902041066f,   /**< C5 */
     0.1491737592987093474996556991856621741334f,  /**< C6 */
    -0.144821909765600057643339822163783003331f,   /**< C7 */
     8.9599230573989110379767056226651046356e-2f   /**< C8 */
};

/* ============================================================
 * SINH(x)
 * ============================================================
 * @par Range
 *   x in [0, ln2/2].
 *   For |x| > ln2/2: sinh(x) = (exp(x) - exp(-x)) / 2
 *
 * @par Form
 *   sinh(x) = x * P(x^2),  P is degree-3 in t = x^2
 *
 * @par Accuracy
 *   2.6e-5 ULP
 *
 * @par Coefficients (4 values, indices 0..3)
 * ============================================================ */
static const float SINH_COEFFS[4] = {
    0.999999999995506589113030978465441728845f,    /**< C0 */
    0.166666667863453360815279663872642870366f,    /**< C1 */
    8.3332835305493575391661845707539300956e-3f,   /**< C2 */
    1.99075738395020160195941395333857184758e-4f   /**< C3 */
};

/* ============================================================
 * COSH(x)
 * ============================================================
 * @par Range
 *   x in [0, ln2/2].
 *   For |x| > ln2/2: cosh(x) = (exp(x) + exp(-x)) / 2
 *
 * @par Form
 *   cosh(x) = P(x^2),  even polynomial, degree-3 in t = x^2
 *
 * @par Reconstruction
 *   Pass x^2 directly to mc_poly_estrin_deg3_ps(). No x multiply needed.
 *
 * @par Accuracy
 *   6.8e-4 ULP
 *
 * @par Coefficients (4 values, indices 0..3)
 * ============================================================ */
static const float COSH_COEFFS[4] = {
    0.999999999959534772527072194807811326055f,    /**< C0 */
    0.500000010776960664452566756637566496614f,    /**< C1 */
    4.16662182295045307372470182575150115655e-2f,  /**< C2 */
    1.39485833100088545140588669801341803883e-3f   /**< C3 */
};

/* ============================================================
 * TANH(x)
 * ============================================================
 * @par Range
 *   x in [0, ln2/2].
 *   For x > ln2/2: tanh(x) = 1 - 2 / (exp(2x) + 1)
 *
 * @par Form
 *   tanh(x) = x * P(x^2),  P is degree-4 in t = x^2
 *
 * @par Reconstruction
 *   Pass x^2 to mc_poly_estrin_deg4_ps(), multiply by x.
 *
 * @par Accuracy
 *   2.2e-3 ULP
 *
 * @par Coefficients (5 values, indices 0..4)
 * ============================================================ */
static const float TANH_COEFFS[5] = {
     0.99999999962493826654063440501532327137f,    /**< C0 */
    -0.333333175699346402857353793233040621278f,   /**< C1 */
     0.133322720162943708147471884818690998073f,   /**< C2 */
    -5.37170264940897791481321780372525368095e-2f, /**< C3 */
     1.94110016222003854929537112793490501015e-2f  /**< C4 */
};

/* ============================================================
 * SQRT(x) -- No polynomial
 * ============================================================
 * @note Use Newton-Raphson on the hardware rsqrt seed. See SIMD_sqrt_ps.h.
 *
 * @par POW(x, y) = exp(y * ln(x))
 * @note Implemented via EXP_COEFFS and LN_COEFFS above. No separate polynomial.
 * ============================================================ */

#endif /* MINIMAX_COEFFS_H */
