#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lya {
constexpr double c_light = 2.99792458e10;
constexpr double km = 1.e5;
constexpr double m_h = 1.6735327e-24;
constexpr double k_b = 1.380648813e-16;
constexpr double nu_0 = 2.466e15;
constexpr double electron_charge = 4.80320451e-10;
constexpr double electron_mass = 9.109382917e-28;
constexpr double oscillator_strength = .4162;
constexpr double natural_width = 9.936e7;

enum class Sampling : int { Band, Comb };
enum class Profile : int { Voigt, Delta };

struct Header {
    std::uint64_t pixels = 0, depth = 0;
    double hubble_param = 0, omega_m = 0, omega_b = 0;
    double unit_length = 0, unit_mass = 0, unit_velocity = 0, opening_angle = 0;
};

struct Region {
    std::uint64_t x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    std::uint64_t rays() const { return (x1 - x0) * (y1 - y0); }
};

// Source-frame frequencies are stored as nu / nu_Lya. The velocity coordinate
// remains a red-positive wavelength offset in cm/s for existing consumers.
struct Frequency {
    double velocity = 0, q_lo = 0, q_hi = 0;
    double weights[5] = {};
};

struct Spectrum {
    std::vector<Frequency> frequencies;
    std::vector<double> velocity_edges;
    std::vector<double> band_edges;
};

struct Source {
    double requested = 0, actual = 0;
    std::uint64_t start_cell = 0;
    std::vector<double> spread;
};

// A segment references a cell in the current raw slab. Its geometry and unit
// factors are independent of ray and frequency and are prepared on the host.
struct Segment {
    std::uint64_t cell = 0;
    double redshift = 0, length = 0, hubble = 0;
    int report = 0; // Last subsegment of this cell, for cumulative products.
};

struct RawSlab {
    Region region;
    std::uint64_t z0 = 0, depth = 0;
    // HDF5 order is [ray, depth], and velocities add a final component axis.
    std::vector<double> density, ionized, temperature, velocity;
};

struct Batch {
    std::vector<Segment> segments;
    std::vector<std::uint64_t> offsets; // source_count + 1
};

struct BackendSettings {
    Sampling sampling = Sampling::Band;
    Profile profile = Profile::Voigt;
    bool legacy = false, cumulative = false;
    int device = 0;
};

struct BackendTiming {
    double transfer_seconds = 0, kernel_seconds = 0;
};
} // namespace lya
