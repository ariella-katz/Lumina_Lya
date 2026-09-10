#include "lya/numerics.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace {
void close(double actual, double expected, double tolerance, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected))) {
        std::cerr << label << ": got " << actual << ", expected " << expected << '\n';
        std::exit(1);
    }
}
template <typename F> double simpson(F f, double lo, double hi, int intervals = 2048) {
    const double dx = (hi - lo) / intervals;
    double sum = f(lo) + f(hi);
    for (int n = 1; n < intervals; ++n) sum += (n % 2 ? 4.0 : 2.0) * f(lo + n * dx);
    return sum * dx / 3.0;
}
}

int main() {
    using namespace lya::math;
    for (double x : {0.0, 0.01, 0.7, 2.14, 4.0, 5.15, 7.0, 8.0}) {
        const double reference = x * simpson([=](double t) { return std::exp(-x * x * (1.0 - t * t)); }, 0.0, 1.0, 4096);
        close(dawson(x), reference, 2.e-8, "Dawson quadrature");
        close(dawson(-x), -dawson(x), 1.e-15, "Dawson odd symmetry");
    }
    close(hypergeo(0.0), 0.0, 0.0, "hypergeometric origin");
    for (double boundary : {2.14, 5.15, 20.0})
        close(hypergeo(std::nextafter(boundary, INFINITY)), hypergeo(boundary), 2.e-10,
              "hypergeometric branch continuity");
    for (double a : {0.0, 4.7e-4, 0.05}) {
        for (bool second : {false, true}) {
            close(voigt_h(0.0, a, second), 1.0 - 2.0 * a * inv_sqrt_pi + (second ? a * a : 0.0),
                  1.e-14, "line centre");
            for (const auto bounds : {std::pair<double,double>{-2.0, 3.0}, {1.2, 1.200001}, {5.0, 7.0}, {8.0, 20.0}, {100.0, 100.001}}) {
                const double lo = bounds.first, hi = bounds.second;
                const double reference = simpson([=](double x) { return voigt_h(x, a, second); }, lo, hi);
                close(voigt_integral(lo, hi, a, second), reference, 2.e-10, "comb quadrature");
                close(voigt_integral(-hi, -lo, a, second), reference, 2.e-10, "comb even symmetry");
            }
            for (const auto bounds : {std::pair<double,double>{-2.0, 3.0}, {1.2, 1.200001}, {5.0, 7.0}, {100.0, 160.0}}) {
                const double lo = bounds.first, hi = bounds.second, drift = 1.3;
                const double reference = simpson([=](double x) { return voigt_integral(x - drift, x, a, second); }, lo, hi);
                close(band_voigt_integral(lo, hi, drift, a, second), reference, 2.e-10, "band quadrature");
                close(band_voigt_integral(drift - hi, drift - lo, drift, a, second), reference, 2.e-10,
                      "band reflection symmetry");
            }
        }
    }
    close(delta_cell_tau(0.0, 1.0, 3.0), 0.0, 0.0, "delta upstream endpoint");
    close(delta_cell_tau(1.0, 1.0, 3.0), 3.0 * sqrt_pi, 0.0, "delta downstream endpoint");
    close(delta_cell_tau(1.0, 0.0, 3.0), 0.0, 0.0, "zero path");
    close(band_delta_tau(-1.0, 1.0, 1.0, 1.e8), std::log(2.0), 1.e-15, "partly saturated delta");
    close(band_delta_tau(0.0, 1.0, 1.0, 1.e8), sqrt_pi * 1.e8, 1.e-15, "fully saturated delta");
    close(band_delta_tau(-1.0, 1.0, 1.0, 1.e-12), 0.5 * sqrt_pi * 1.e-12, 1.e-23, "transparent delta");
    close(band_voigt_integral(-1.0, 1.0, 0.0, 4.7e-4), 0.0, 0.0, "zero band path");
    std::cout << "Numerical primitive checks passed.\n";
}
