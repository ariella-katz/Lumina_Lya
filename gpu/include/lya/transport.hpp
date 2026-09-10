#pragma once

#include "numerics.hpp"
#include "types.hpp"
#include <cmath>

#if defined(__CUDACC__)
#define LYA_TRANSPORT_HD __host__ __device__
#else
#define LYA_TRANSPORT_HD
#endif

namespace lya {
// Density and velocity retain their scale-factor-independent code conversion.
// A single preprocessing pass is reused by every source and frequency.
struct GasCell {
    double neutral_a1, inverse_thermal_speed, outward_velocity_a1;
};

LYA_TRANSPORT_HD inline bool voigt_expansion_valid(const GasCell& gas,
    const BackendSettings& settings) {
    return settings.profile != Profile::Voigt || gas.neutral_a1 == 0.
        || natural_width * c_light / (2. * nu_0) * gas.inverse_thermal_speed <= .1;
}

LYA_TRANSPORT_HD inline GasCell prepare_gas(double density, double ionized,
    double temperature, double vx, double vy, double vz, std::uint64_t ray,
    const Region& region, const Header& header) {
    const auto ny = region.y1 - region.y0;
    const double x = static_cast<double>(region.x0 + ray / ny) + .5;
    const double y = static_cast<double>(region.y0 + ray % ny) + .5;
    const double np = static_cast<double>(header.pixels);
    const double tx = ::tan((x - .5 * np) * header.opening_angle / np);
    const double ty = ::tan((y - .5 * np) * header.opening_angle / np);
    const double norm = 1. / ::sqrt(1. + tx * tx + ty * ty);
    const double length = header.unit_length / header.hubble_param;
    const double density_unit = (header.unit_mass / header.hubble_param)
        / (length * length * length);
    return {.76 * density * density_unit / m_h * (1. - ionized),
        ::sqrt(m_h / (2. * k_b * temperature)),
        (vx * tx + vy * ty + vz) * norm * header.unit_velocity};
}

// x decreases along observerward propagation. Cosmological frequencies are
// evaluated at each subsegment midpoint; the analytic operator then uses the
// paper's constant linear frequency drift across that subsegment.
LYA_TRANSPORT_HD inline double segment_tau(const GasCell& gas,
    const Segment& segment, double source_redshift, const Frequency& frequency,
    const BackendSettings& settings) {
    if (gas.neutral_a1 == 0. || segment.length == 0.) return 0.;
    const double zp1 = 1. + segment.redshift;
    const double neutral = gas.neutral_a1 * zp1 * zp1 * zp1;
    const double inverse_b = gas.inverse_thermal_speed;
    const double velocity = gas.outward_velocity_a1 / ::sqrt(zp1);
    if (!(::fabs(velocity) < c_light)) return ::nan("");
    const double drift = segment.hubble * inverse_b * segment.length;
    // k0/K cancels thermal speed analytically, including in Delta mode.
    constexpr double sqrt_pi = 1.77245385090551602729816748334114518;
    const double weight = neutral * oscillator_strength * sqrt_pi
        * electron_charge * electron_charge
        / (electron_mass * nu_0 * segment.hubble);
    const double damping = natural_width * c_light / (2. * nu_0) * inverse_b;
    double lo, hi;
    if (settings.legacy) {
        const double offset = c_light * ((1. + frequency.velocity / c_light)
            * (1. + source_redshift) / zp1 - 1.);
        lo = hi = -(offset + velocity) * inverse_b;
    } else {
        // Photons travel along -n_out, giving nu_gas = nu*(1+v_out/c).
        const double factor = zp1 / (1. + source_redshift)
            * (1. + velocity / c_light);
        lo = (frequency.q_lo * factor - 1.) * c_light * inverse_b + .5 * drift;
        hi = (frequency.q_hi * factor - 1.) * c_light * inverse_b + .5 * drift;
    }
    if (settings.profile == Profile::Delta) {
        return settings.sampling == Sampling::Comb
            ? math::delta_cell_tau(lo, drift, weight)
            : math::band_delta_tau(lo, hi, drift, weight);
    }
    const bool second_order = !settings.legacy;
    return weight * (settings.sampling == Sampling::Comb
        ? math::voigt_integral(lo - drift, lo, damping, second_order)
        : math::band_voigt_integral(lo, hi, drift, damping, second_order) / (hi - lo));
}
} // namespace lya

#undef LYA_TRANSPORT_HD
