#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define LYA_HD __host__ __device__
#else
#define LYA_HD
#endif

// All integrals use H(a,x), whose integral over the real line is sqrt(pi).
// The first/second order expansion is Almualla et al. (2026), Eqs. 10-12.
// Callers validate finite inputs, nonnegative damping and nonnegative drift.
namespace lya { namespace math {
constexpr double sqrt_pi = 1.7724538509055160273;
constexpr double inv_sqrt_pi = 0.56418958354775628695;
constexpr double machine_epsilon = 2.22044604925031308085e-16;

// The relative-width quadrature shortcut assumes an inverse-square wing.
// Require its Gaussian contribution to be below roundoff before using it.
// At x>=8, ordinary Ly-alpha damping satisfies this without an exponential.
LYA_HD inline bool damping_wing_dominates(double x, double a) {
    return a > 0.0 && x >= 8.0 && (a >= 1.e-8 || x >= 27.0
        || sqrt_pi * exp(-x * x) * x * x < a * machine_epsilon);
}

// The supplied gpu/hypergeo_approx.cpp FP64 coefficients are unchanged.
LYA_HD inline double hypergeo(double x) {
  if (x < 0.) x = -x;

  if (x > 20.) {
    // Region 4: (x > 20) Asymptotic expansion
    const double z = 1. / x;
    const double z2 = z * z;
    return 0.55389595193643551 + 0.56418958354775629 * log(x) +
      exp(-x * x) * z * (0.5 + z2 * (-0.25 + z2 * (0.375 + z2 * (-0.9375 + z2 * (3.28125 - 14.765625 * z2))))) +
      z2 * (-0.14104739588693907 + z2 * (-0.1057855469152043 + z2 * (-0.17630924485867384 +
      z2 * (-0.46281176775401883 + z2 * (-1.6661223639144678 - 7.6363941679413107 * z2)))));
  } else if (x > 5.15) {
    // Region 3: (5.15 < x < 20) Minimax rational approximation
    return
      (0.43682128011845283 + x * (1.1350393218808412 + x * (-2.6886477055342774 +
       x * (0.38197597791453486 + x * (0.50175054988846498 + x * (-0.038945006088876756 +
       x * (-0.01960980532428994 + x * (-0.0012352801187522119 +
       x * (-0.000021929464819640861 - 8.9740333784715440e-8 * x))))))))) /
      (1. + x * (-1.4655446738737176 + x * (-1.4032687362246785 + x * (0.88017831100711925 +
       x * (0.20315464152004808 + x * (-0.050541821629539143 + x * (-0.010129700645083951 +
       x * (-0.00047171453699007742 + x * (-6.5774668524211879e-6 +
       x * (-2.0308831722073883e-8 + 3.0992253950009977e-12 * x))))))))));
  } else if (x > 2.14) {
    // Region 2: (2.14 < x < 5.15) Minimax rational approximation
    return
      (-0.042894679076630557 + x * (0.22824496819198929 + x * (0.025188478146879472 +
       x * (-0.073680556438870441 + x * (0.015478791578754775 + x * (0.03578546465825914 +
       x * (-0.027136284977882523 + x * (0.00988058128216986 + x * (-0.0018266571476397632 +
       x * (0.00015895087049937769 + 2.2012003108278176e-6 * x)))))))))) /
      (1. + x * (-1.4228913485269134 + x * (1.422586587092723 + x * (-0.90352979918845972 +
       x * (0.40905729430981564 + x * (-0.11897768600690815 + x * (0.020260960808591361 +
       x * (-0.00051357699758117836 + x * (-0.00042088721875418447 +
       x * (0.000070884265471591177 + 4.4276369978816760e-7 * x))))))))));
  } else {
    // Region 1: (0 < x < 2.14) Padé approximant at x = 0
    const double x2 = x * x;
    return
      (x2 * (0.56418958354775629 + x2 * (0.085729533651389712 + x2 * (0.022086832488217919 +
       x2 * (0.0017036307776357589 + x2 * (0.00018771767997732845 + x2 * (8.5231641525552708e-6 +
       x2 * (5.0427978836558623e-7 + x2 * (1.3331831289246041e-8 + x2 * (4.1502346771379203e-10 +
       x2 * (5.1601562894552471e-12 + 5.7441158232281185e-14 * x2))))))))))) /
      (1. + x2 * (0.48528497539007305 + x2 * (0.11202066087251571 + x2 * (0.016271004013903894 +
       x2 * (0.0016562768063564546 + x2 * (0.00012438873114514399 + x2 * (7.0533651782670942e-6 +
       x2 * (3.0296362830876420e-7 + x2 * (9.6956364133569300e-9 + x2 * (2.2108600436326525e-10 +
       x2 * (3.2481690234488706e-12 + 2.3420543014955824e-14 * x2)))))))))));
  }
}

template <int N>
LYA_HD inline double chebyshev(double x, const double (&c)[N]) {
    double b1 = 0.0, b2 = 0.0;
    for (int k = N - 1; k > 0; --k) {
        const double b0 = 2.0 * x * b1 - b2 + c[k];
        b2 = b1;
        b1 = b0;
    }
    return x * b1 - b2 + c[0];
}

// Independent Chebyshev expansions of Dawson's integral on [0,4] and [4,8].
// Coefficients were formed by cosine projection at 128 Chebyshev nodes.
// The tail is its asymptotic expansion; stopping early saves work in the wings.
LYA_HD inline double dawson(double x) {
    const double y = fabs(x);
    double result;
    if (y <= 4.0) {
        const double c[] = {
            0.22155672794739226, -0.31811346996168133, 0.20873845413642239,
            -0.12475409913779133, 0.06786930518667679, -0.033659144895270943,
            0.015260781271987971, -0.006348370962596211, 0.0024326740920748457,
            -0.00086219541491064301, 0.00028376573336321082, -8.7057549874165586e-05,
            2.4986849985476786e-05, -6.7319286764096295e-06, 1.7078578785488122e-06,
            -4.0917551225614058e-07, 9.2828292210308874e-08, -1.9991403602732621e-08,
            4.0963490568730556e-09, -8.0032408803125204e-10, 1.4938502225928852e-10,
            -2.6687993005450718e-11, 4.5712145582742312e-12, -7.5186313745019204e-13,
            1.1891737459112858e-13, -1.8102669564106291e-14, 2.647069439727992e-15,
            -3.6201849352469895e-16, 3.7235568361299043e-17
        };
        result = y * chebyshev(y * y / 8.0 - 1.0, c);
    } else if (y < 8.0) {
        const double c[] = {
            0.090155249144386898, -0.032059455771640578, 0.0058078885883300737,
            -0.0010735254547985886, 0.00020280497605197072, -3.9242280169813132e-05,
            7.7994926613860751e-06, -1.5982548796484828e-06, 3.3930229518951286e-07,
            -7.5030051457093472e-08, 1.7352873029697477e-08, -4.1935782703766181e-09,
            1.0478212837482119e-09, -2.6442154441563036e-10, 6.4860420617836844e-11,
            -1.4553196214292435e-11, 2.6342018214013698e-12, -2.1326148439627566e-13,
            -1.0549817804425413e-13, 7.2220636250313296e-14, -2.8066504046390548e-14,
            8.1728832652028151e-15, -1.7856731430288637e-15, 2.3847238542055484e-16,
            1.1827756317222236e-17
        };
        result = chebyshev((y - 6.0) * 0.5, c);
    } else {
        const double inverse = 1.0 / y;
        const double q = 0.5 * inverse * inverse;
        double term = 1.0, sum = 1.0;
        for (int n = 1; n <= 16; ++n) {
            term *= (2.0 * n - 1.0) * q;
            sum += term;
            if (term < machine_epsilon * sum) break;
        }
        result = 0.5 * inverse * sum;
    }
    return copysign(result, x);
}

LYA_HD inline double voigt_h(double x, double a, bool second_order = true) {
    const double y = fabs(x);
    const double x2 = y * y;
    // Avoid cancellation in 2*x*D(x)-1, including at enormous wing offsets.
    double wing;
    if (y >= 8.0) {
        const double inverse = 1.0 / y;
        const double q = 0.5 * inverse * inverse;
        double term = q;
        wing = term;
        for (int n = 2; n <= 17; ++n) {
            term *= (2.0 * n - 1.0) * q;
            wing += term;
            if (term < machine_epsilon * wing) break;
        }
    } else {
        wing = 2.0 * y * dawson(y) - 1.0;
    }
    const double gaussian = y < 27.0
        ? exp(-x2) * (1.0 + (second_order ? a * a * (1.0 - 2.0 * x2) : 0.0))
        : 0.0;
    return gaussian + 2.0 * a * inv_sqrt_pi * wing;
}

LYA_HD inline double upsilon(double x, double a, bool second_order = true) {
    const double gaussian_correction = fabs(x) < 27.0 && second_order
        ? a * a * x * exp(-x * x) : 0.0;
    return 0.5 * sqrt_pi * erf(x) - 2.0 * a * inv_sqrt_pi * dawson(x)
        + gaussian_correction;
}

// Residual of Eq. 12 after subtracting sqrt(pi)*abs(x)/2. Treating the
// subtracted piece as an overlap length removes large linear cancellations.
LYA_HD inline double band_primitive_residual(double x, double a, bool second_order) {
    const double y = fabs(x);
    const double gaussian = y < 27.0
        ? 0.5 * ((1.0 - (second_order ? a * a : 0.0)) * exp(-y * y)
            - sqrt_pi * y * erfc(y))
        : 0.0;
    return gaussian - (a == 0.0 ? 0.0 : a * hypergeo(y));
}

// Eight point Gauss-Legendre evaluation is used only for short, smooth
// intervals whose antiderivative subtraction loses significant digits.
LYA_HD inline double short_profile_integral(double lo, double hi, double a, bool second_order) {
    const double nodes[] = {0.18343464249564980494, 0.52553240991632898582,
                            0.79666647741362673959, 0.96028985649753623168};
    const double weights[] = {0.36268378337836198297, 0.31370664587788728734,
                              0.22238103445337447054, 0.10122853629037625915};
    const double half = 0.5 * (hi - lo), mid = lo + half;
    double sum = 0.0;
    for (int n = 0; n < 4; ++n)
        sum += weights[n] * (voigt_h(mid - half * nodes[n], a, second_order)
                            + voigt_h(mid + half * nodes[n], a, second_order));
    return half * sum;
}

LYA_HD inline double voigt_integral(double lo, double hi, double a, bool second_order = true) {
    if (hi == lo) return 0.0;
    if (hi < lo || a < 0.0) return nan("");
    if (hi <= 0.0) { const double old_lo = lo; lo = -hi; hi = -old_lo; }
    const double width = hi - lo;
    const double nearest = lo > 0.0 ? lo : 0.0;
    if (width < 1.e-3 * (1.0 + nearest) && nearest < 8.0)
        return short_profile_integral(lo, hi, a, second_order);
    if (nearest >= 8.0 && width * nearest < 1.0 && !damping_wing_dominates(nearest, a))
        return short_profile_integral(lo, hi, a, second_order);
    if (lo < 0.0)
        return upsilon(hi, a, second_order) + upsilon(-lo, a, second_order);
    double delta_dawson;
    if (lo >= 8.0) {
        // Evaluate D(lo)-D(hi) term by term with expm1(log1p(...)).
        const double inverse = 1.0 / lo, q = 0.5 * inverse * inverse;
        const double ratio_log = log1p(width / lo);
        double term = 0.5 * inverse;
        delta_dawson = term * -expm1(-ratio_log);
        for (int n = 1; n <= 16; ++n) {
            term *= (2.0 * n - 1.0) * q;
            const double change = term * -expm1(-(2.0 * n + 1.0) * ratio_log);
            delta_dawson += change;
            if (change < machine_epsilon * delta_dawson) break;
        }
    } else {
        delta_dawson = dawson(lo) - dawson(hi);
    }
    const double gaussian = 0.5 * sqrt_pi * (erfc(lo) - erfc(hi));
    const double second = second_order
        ? a * a * ((hi < 27.0 ? hi * exp(-hi * hi) : 0.0)
                 - (lo < 27.0 ? lo * exp(-lo * lo) : 0.0)) : 0.0;
    return fmax(0.0, gaussian + 2.0 * a * inv_sqrt_pi * delta_dawson + second);
}

// Integrate over the shorter rectangle dimension. The other dimension is
// evaluated analytically, so a large drift can cross resonance safely.
LYA_HD inline double short_band_integral(double lo, double hi, double drift,
                                        double a, bool second_order, bool very_smooth = false) {
    const double nodes[] = {0.18343464249564980494, 0.52553240991632898582,
                            0.79666647741362673959, 0.96028985649753623168};
    const double weights[] = {0.36268378337836198297, 0.31370664587788728734,
                              0.22238103445337447054, 0.10122853629037625915};
    const double width = hi - lo;
    const bool integrate_x = width < drift;
    const double half = 0.5 * (integrate_x ? width : drift);
    const double mid = integrate_x ? lo + half : half;
    // In a wing with relative extent below 0.01, three Gauss nodes have
    // ample accuracy and substantially reduce special-function calls.
    if (very_smooth) {
        const double offset = half * 0.77459666924148337704;
        const double left = mid - offset, right = mid + offset;
        const double center = integrate_x ? voigt_integral(mid - drift, mid, a, second_order)
            : voigt_integral(lo - mid, hi - mid, a, second_order);
        const double sides = integrate_x
            ? voigt_integral(left - drift, left, a, second_order)
              + voigt_integral(right - drift, right, a, second_order)
            : voigt_integral(lo - left, hi - left, a, second_order)
              + voigt_integral(lo - right, hi - right, a, second_order);
        return half * ((8.0 / 9.0) * center + (5.0 / 9.0) * sides);
    }
    double sum = 0.0;
    for (int n = 0; n < 4; ++n) {
        const double left = mid - half * nodes[n], right = mid + half * nodes[n];
        sum += weights[n] * (integrate_x
            ? voigt_integral(left - drift, left, a, second_order)
              + voigt_integral(right - drift, right, a, second_order)
            : voigt_integral(lo - left, hi - left, a, second_order)
              + voigt_integral(lo - right, hi - right, a, second_order));
    }
    return half * sum;
}

LYA_HD inline double far_band_integral(double nearest, double width, double drift,
                                      double a, bool second_order) {
    // The leading inverse-square wing has an exact stable double integral.
    const double w = width / nearest, d = drift / nearest;
    double sum = log1p((w / (1.0 + w + d)) * d);
    const double inv2 = 1.0 / (nearest * nearest);
    double power = inv2, h_coefficient = 1.5;
    for (int m = 1; m <= 15; ++m) {
        const double p = 2.0 * m;
        const double mixed = 1.0 - exp(-p * log1p(w)) - exp(-p * log1p(d))
            + exp(-p * log1p(w + d));
        const double change = h_coefficient * power * mixed / (p * (p + 1.0));
        sum += change;
        if (fabs(change) < machine_epsilon * fabs(sum)) break;
        h_coefficient *= (2.0 * m + 3.0) * 0.5;
        power *= inv2;
    }
    // Keep the Gaussian part even when damping is zero.
    const double gauss = band_primitive_residual(nearest, 0.0, false)
        - band_primitive_residual(nearest + width, 0.0, false)
        - band_primitive_residual(nearest + drift, 0.0, false)
        + band_primitive_residual(nearest + width + drift, 0.0, false);
    double second = 0.0;
    if (second_order && nearest < 27.0) {
        const double corners[] = {nearest, nearest + width, nearest + drift,
                                   nearest + width + drift};
        for (int n = 0; n < 4; ++n)
            if (corners[n] < 27.0)
                second -= 0.5 * a * a * ((n == 0 || n == 3) ? 1.0 : -1.0)
                    * exp(-corners[n] * corners[n]);
    }
    return fmax(0.0, a * inv_sqrt_pi * sum + gauss + second);
}

// Returns the double integral, BEFORE division by the spectral-bin width:
// integral_lo^hi integral_0^drift H(x-t) dt dx. Multiply by k0/(K*(hi-lo)).
LYA_HD inline double band_voigt_integral(double lo, double hi, double drift,
                                         double a, bool second_order = true) {
    if (hi == lo || drift == 0.0) return 0.0;
    if (hi < lo || drift < 0.0 || a < 0.0)
        return nan("");
    const double width = hi - lo, shorter = fmin(width, drift);
    const double nearest = lo > drift ? lo - drift : (hi < 0.0 ? -hi : 0.0);
    if (nearest >= 8.0 && !damping_wing_dominates(nearest, a)) {
        // Gaussian variation scales as x*dx, unlike the dx/x wing criterion.
        // Short smooth intervals use eight nodes. For larger variation the
        // Gaussian primitive differences are well conditioned and exact.
        return shorter * nearest < 1.0
            ? short_band_integral(lo, hi, drift, a, second_order)
            : far_band_integral(nearest, width, drift, a, second_order);
    }
    if (shorter < 0.1 || (nearest >= 8.0 && shorter < 0.25 * nearest))
        return short_band_integral(lo, hi, drift, a, second_order,
                                   nearest >= 8.0 && shorter < 0.01 * nearest);
    if (nearest >= 8.0)
        return far_band_integral(nearest, width, drift, a, second_order);
    const double linear = sqrt_pi * fmax(0.0, fmin(hi, drift) - fmax(lo, 0.0));
    const double b0 = band_primitive_residual(hi, a, second_order);
    const double b1 = band_primitive_residual(lo, a, second_order);
    const double b2 = band_primitive_residual(hi - drift, a, second_order);
    const double b3 = band_primitive_residual(lo - drift, a, second_order);
    const double result = linear + (b0 - b1) - (b2 - b3);
    // This condition also protects tiny damping in almost Gaussian wings.
    if (shorter < 1.0 && result < 1.e-7 * (fabs(b0) + fabs(b1) + fabs(b2) + fabs(b3)))
        return short_band_integral(lo, hi, drift, a, second_order);
    return fmax(0.0, result);
}

LYA_HD inline double delta_cell_tau(double x, double drift, double weight) {
    return drift > 0.0 && x > 0.0 && x <= drift ? sqrt_pi * weight : 0.0;
}

LYA_HD inline double band_delta_tau(double lo, double hi, double drift, double weight) {
    if (drift <= 0.0 || weight == 0.0) return 0.0;
    if (hi <= lo || weight < 0.0) return nan("");
    const double fraction = fmin(1.0, fmax(0.0, fmin(hi, drift) - fmax(lo, 0.0)) / (hi - lo));
    if (fraction == 0.0) return 0.0;
    const double tau = sqrt_pi * weight;
    // Preserve finite opaque optical depths when the entire bin resonates.
    if (fraction == 1.0) return tau;
    if (tau < 0.5) return -log1p(fraction * expm1(-tau));
    const double unabsorbed_log = log1p(-fraction);
    const double absorbed_log = log(fraction) - tau;
    const double larger = fmax(unabsorbed_log, absorbed_log);
    return -(larger + log1p(exp(fmin(unabsorbed_log, absorbed_log) - larger)));
}

}} // namespace lya::math
