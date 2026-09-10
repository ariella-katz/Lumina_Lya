#include "lya/backend.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
        std::abs(actual - expected) <= 1.e-12 * (1. + std::abs(expected)), message);
}
template<class Function> void rejects(Function function, const char* message) {
    try { function(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

lya::Header header() {
    lya::Header h;
    h.pixels = 3; h.depth = 3;
    h.hubble_param = .7; h.omega_m = .3; h.omega_b = .048;
    h.unit_length = 1.e21; h.unit_mass = 1.e33;
    h.unit_velocity = 1.e5; h.opening_angle = .0628;
    return h;
}

lya::Spectrum spectrum(lya::Sampling sampling) {
    lya::Spectrum f;
    for (unsigned i = 0; i < 2; ++i) {
        lya::Frequency frequency;
        const double center = i ? 1.000004 : 1.;
        const double half_width = sampling == lya::Sampling::Band ? 1.e-6 : 0.;
        frequency.q_lo = center - half_width;
        frequency.q_hi = center + half_width;
        frequency.velocity = (1. / center - 1.) * lya::c_light;
        for (double& weight : frequency.weights) weight = i ? .7 : .3;
        f.frequencies.push_back(frequency);
    }
    return f;
}

lya::RawSlab raw(std::uint64_t z0, std::uint64_t depth, lya::Region region = {1, 2, 1, 2}) {
    lya::RawSlab slab;
    slab.region = region; slab.z0 = z0; slab.depth = depth;
    for (std::uint64_t ray = 0; ray < region.rays(); ++ray)
        for (std::uint64_t z = z0; z < z0 + depth; ++z) {
            // Global coordinates make a ray identical when spatially retiled.
            const auto x = region.x0 + ray / (region.y1 - region.y0);
            const auto y = region.y0 + ray % (region.y1 - region.y0);
            slab.density.push_back(1. + .1 * (x + y + z));
            slab.ionized.push_back(.999999);
            slab.temperature.push_back(1.e4 + z * 500.);
            slab.velocity.push_back(.1 * x);
            slab.velocity.push_back(-.2 * y);
            slab.velocity.push_back(.3 * z);
        }
    return slab;
}

lya::Batch batch(std::uint64_t z0, std::uint64_t depth) {
    lya::Batch result;
    result.offsets.push_back(0);
    for (unsigned source = 0; source < 2; ++source) {
        for (std::uint64_t z = z0; z < z0 + depth; ++z) {
            // Source one ends before the last cell, testing absent reports.
            if (source == 1 && z == 2) continue;
            result.segments.push_back({z - z0, 7. - 1.e-6 * z,
                1.e21 * (1. + .1 * z), 1.e-17, 1});
        }
        result.offsets.push_back(result.segments.size());
    }
    return result;
}

void mode(lya::Sampling sampling, lya::Profile profile) {
    const auto h = header();
    const auto f = spectrum(sampling);
    const std::vector<lya::Source> sources = {{7.000001, 7.000001, 0, {}},
                                             {7.000002, 7.000002, 0, {}}};
    lya::BackendSettings settings;
    settings.sampling = sampling; settings.profile = profile;
    auto normal = lya::make_cpu_backend(h, f, sources, settings);
    normal->begin({1, 2, 1, 2}, 3);
    normal->submit(raw(0, 3), batch(0, 3));
    const auto expected = normal->optical_depths();
    require(expected.size() == 4, "Expected [2 sources, 1 ray, 2 frequencies]");
    for (double tau : expected) require(tau >= 0. && std::isfinite(tau), "Invalid normal tau");

    settings.cumulative = true;
    auto cumulative = lya::make_cpu_backend(h, f, sources, settings);
    cumulative->begin({1, 2, 1, 2}, 2);
    cumulative->submit(raw(0, 1), batch(0, 1));
    cumulative->submit(raw(1, 2), batch(1, 2));
    cumulative->finish();
    const auto actual = cumulative->optical_depths();
    const auto sums = cumulative->cumulative_sums();
    require(sums.size() == 2 * 5 * 3, "Wrong cumulative shape");
    for (std::size_t i = 0; i < expected.size(); ++i)
        close(actual[i], expected[i], "Cumulative path or slab split changed optical depths");
    for (unsigned source = 0; source < 2; ++source)
        for (unsigned band = 0; band < 5; ++band) {
            const unsigned end = source == 0 ? 2 : 1;
            const double transmission = .3 * std::exp(-expected[2 * source])
                + .7 * std::exp(-expected[2 * source + 1]);
            close(sums[(source * 5 + band) * 3 + end], transmission,
                "Final cumulative value differs from final band transmission");
            for (unsigned z = 0; z <= end; ++z) {
                const double value = sums[(source * 5 + band) * 3 + z];
                require(value >= 0. && value <= 1. + 1.e-14, "Cumulative normalization outside [0,1]");
                if (z) require(value <= sums[(source * 5 + band) * 3 + z - 1] + 1.e-14,
                    "Cumulative absorption increased transmission");
            }
            if (source == 1) close(sums[(source * 5 + band) * 3 + 2], 0., "Absent cell has cumulative contribution");
        }

    // A source on the observer endpoint has no traversed segments.
    cumulative->begin({1, 2, 1, 2}, 1);
    lya::Batch empty;
    empty.offsets = {0, 0, 0};
    cumulative->submit(raw(0, 1), empty);
    for (double tau : cumulative->optical_depths()) close(tau, 0., "Empty source tau is not zero");
    for (double sum : cumulative->cumulative_sums()) close(sum, 0., "Empty source has report contributions");

    auto transparent = raw(0, 1);
    transparent.ionized[0] = 1.;
    cumulative->submit(transparent, batch(0, 1));
    for (double tau : cumulative->optical_depths()) close(tau, 0., "Fully ionized gas absorbs");
    const auto transparent_sums = cumulative->cumulative_sums();
    for (unsigned source = 0; source < 2; ++source)
        for (unsigned band = 0; band < 5; ++band)
            close(transparent_sums[(source * 5 + band) * 3], 1., "Transparent band is not normalized");

    // Four rays exercise [source,ray,frequency] indexing and absolute ray geometry.
    const lya::Region region{1, 3, 0, 2};
    normal->begin(region, 3);
    normal->submit(raw(0, 3, region), batch(0, 3));
    const auto all = normal->optical_depths();
    for (std::uint64_t ray = 0; ray < region.rays(); ++ray) {
        const auto x = region.x0 + ray / 2, y = region.y0 + ray % 2;
        const lya::Region one{x, x + 1, y, y + 1};
        normal->begin(one, 3);
        normal->submit(raw(0, 3, one), batch(0, 3));
        const auto separate = normal->optical_depths();
        for (unsigned source = 0; source < 2; ++source)
            for (unsigned frequency = 0; frequency < 2; ++frequency)
                close(separate[source * 2 + frequency], all[(source * 4 + ray) * 2 + frequency],
                    "Spatial tiling changed a ray's optical depths");
    }

    auto bad = raw(0, 1);
    bad.density[0] = std::numeric_limits<double>::infinity();
    rejects([&] { cumulative->submit(bad, batch(0, 1)); }, "Infinite gas was accepted");
    bad = raw(0, 1); bad.temperature[0] = 0.;
    rejects([&] { cumulative->submit(bad, batch(0, 1)); }, "Zero temperature was accepted");
    bad = raw(0, 1); bad.ionized[0] = 1.01;
    rejects([&] { cumulative->submit(bad, batch(0, 1)); }, "Ionization above unity was accepted");
    auto invalid_batch = batch(0, 1); invalid_batch.segments[0].hubble = 0.;
    rejects([&] { cumulative->submit(raw(0, 1), invalid_batch); }, "Zero Hubble rate was accepted");
    invalid_batch = batch(0, 1); invalid_batch.segments[0].cell = 1;
    rejects([&] { cumulative->submit(raw(0, 1), invalid_batch); }, "Out-of-slab cell was accepted");
    invalid_batch = batch(0, 1); invalid_batch.offsets = {0, 2, 1};
    rejects([&] { cumulative->submit(raw(0, 1), invalid_batch); }, "Invalid offsets were accepted");
    auto cold = raw(0, 1); cold.temperature[0] = 1.e-4;
    cumulative->begin(cold.region, 1);
    if (profile == lya::Profile::Voigt)
        rejects([&] { cumulative->submit(cold, batch(0, 1)); }, "Voigt expansion outside damping domain was accepted");
    else cumulative->submit(cold, batch(0, 1));
    cumulative->begin(cold.region, 1);
    auto fast = raw(0, 1); fast.velocity[2] = 1.e7;
    rejects([&] { cumulative->submit(fast, batch(0, 1)); }, "Superluminal outward velocity was accepted");
    fast.velocity[2] = -1.e7;
    rejects([&] { cumulative->submit(fast, batch(0, 1)); }, "Superluminal inward velocity was accepted");
}
} // namespace

int main() {
    try {
        for (auto sampling : {lya::Sampling::Comb, lya::Sampling::Band})
            for (auto profile : {lya::Profile::Delta, lya::Profile::Voigt}) mode(sampling, profile);
        std::cout << "CPU backend: four modes, cumulative normalization, slab/spatial tiling, empty sources and validation passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
