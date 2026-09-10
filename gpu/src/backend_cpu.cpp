#include "lya/backend.hpp"
#include "lya/transport.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace lya {
namespace {
std::size_t checked_product(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b / sizeof(double))
        throw std::overflow_error("Backend allocation size overflow");
    return static_cast<std::size_t>(a * b);
}

class CpuBackend final : public Backend {
    Header header_;
    Spectrum spectrum_;
    std::vector<Source> sources_;
    BackendSettings settings_;
    Region region_;
    std::uint64_t max_depth_ = 0;
    bool initialized_ = false;
    std::vector<double> tau_, cumulative_;
    BackendTiming timing_;

public:
    CpuBackend(const Header& h, const Spectrum& f, const std::vector<Source>& s,
        const BackendSettings& o) : header_(h), spectrum_(f), sources_(s), settings_(o) {
        if (f.frequencies.empty() || s.empty())
            throw std::invalid_argument("Backend requires at least one source and frequency");
    }

    void begin(const Region& region, std::uint64_t max_depth) override {
        if (region.x1 <= region.x0 || region.y1 <= region.y0 ||
            region.x1 > header_.pixels || region.y1 > header_.pixels || !max_depth)
            throw std::invalid_argument("Invalid backend tile or slab depth");
        region_ = region;
        max_depth_ = max_depth;
        tau_.assign(checked_product(checked_product(region.rays(), sources_.size()),
            spectrum_.frequencies.size()), 0.);
        if (settings_.cumulative)
            cumulative_.assign(checked_product(sources_.size() * 5, header_.depth), 0.);
        else cumulative_.clear();
        initialized_ = true;
    }

    void submit(const RawSlab& slab, const Batch& batch) override {
        if (!initialized_ || slab.region.x0 != region_.x0 || slab.region.x1 != region_.x1 ||
            slab.region.y0 != region_.y0 || slab.region.y1 != region_.y1 ||
            !slab.depth || slab.depth > max_depth_ || slab.z0 > header_.depth ||
            slab.depth > header_.depth - slab.z0)
            throw std::invalid_argument("Raw slab does not match active backend tile");
        const auto cells = checked_product(region_.rays(), slab.depth);
        if (slab.density.size() != cells || slab.ionized.size() != cells ||
            slab.temperature.size() != cells || slab.velocity.size() != 3 * cells)
            throw std::invalid_argument("Raw slab array dimensions do not agree");
        if (batch.offsets.size() != sources_.size() + 1 || batch.offsets.front() != 0 ||
            batch.offsets.back() != batch.segments.size() ||
            !std::is_sorted(batch.offsets.begin(), batch.offsets.end()))
            throw std::invalid_argument("Invalid source segment offsets");
        for (const auto& segment : batch.segments)
            if (segment.cell >= slab.depth || !(segment.redshift > -1.) ||
                !std::isfinite(segment.redshift) || !(segment.hubble > 0.) ||
                !std::isfinite(segment.hubble) || !(segment.length >= 0.) ||
                !std::isfinite(segment.length))
                throw std::invalid_argument("Invalid segment geometry");
        const auto start = std::chrono::steady_clock::now();
        std::vector<GasCell> gas(cells);
        for (std::size_t i = 0; i < cells; ++i) {
            if (!(slab.density[i] >= 0.) || !std::isfinite(slab.density[i]) ||
                !(slab.ionized[i] >= 0. && slab.ionized[i] <= 1.) ||
                !(slab.temperature[i] > 0.) || !std::isfinite(slab.temperature[i]) ||
                !std::isfinite(slab.velocity[3 * i]) ||
                !std::isfinite(slab.velocity[3 * i + 1]) ||
                !std::isfinite(slab.velocity[3 * i + 2]))
                throw std::runtime_error("Nonphysical or nonfinite input gas at slab cell "
                    + std::to_string(i));
            gas[i] = prepare_gas(slab.density[i], slab.ionized[i], slab.temperature[i],
                slab.velocity[3 * i], slab.velocity[3 * i + 1], slab.velocity[3 * i + 2],
                i / slab.depth, region_, header_);
            if (!std::isfinite(gas[i].neutral_a1) || !std::isfinite(gas[i].inverse_thermal_speed) ||
                !std::isfinite(gas[i].outward_velocity_a1))
                throw std::runtime_error("Gas unit conversion overflow at slab cell " + std::to_string(i));
            if (!voigt_expansion_valid(gas[i], settings_))
                throw std::runtime_error("Voigt expansion requires damping a <= 0.1; lower-temperature input needs a full-profile solver");
        }
        const auto nf = spectrum_.frequencies.size();
        const auto nr = region_.rays();
        for (std::size_t source = 0; source < sources_.size(); ++source)
            for (std::uint64_t ray = 0; ray < nr; ++ray) {
                double* tau = tau_.data() + (source * nr + ray) * nf;
                for (auto part = batch.offsets[source]; part < batch.offsets[source + 1]; ++part) {
                    const auto& segment = batch.segments[part];
                    const auto& cell = gas[ray * slab.depth + segment.cell];
                    for (std::size_t f = 0; f < nf; ++f) {
                        const double increment = segment_tau(cell, segment, sources_[source].actual,
                            spectrum_.frequencies[f], settings_);
                        if (!(increment >= 0.) || !std::isfinite(increment))
                            throw std::runtime_error("Invalid optical depth from absorption operator");
                        tau[f] += increment;
                        if (!std::isfinite(tau[f]))
                            throw std::runtime_error("Accumulated optical depth overflow");
                    }
                    if (settings_.cumulative && segment.report)
                        for (std::size_t f = 0; f < nf; ++f) {
                            const double transmission = std::exp(-tau[f]);
                            for (std::size_t band = 0; band < 5; ++band)
                                cumulative_[(source * 5 + band) * header_.depth
                                    + slab.z0 + segment.cell] += transmission
                                    * spectrum_.frequencies[f].weights[band];
                        }
                }
            }
        timing_.kernel_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }

    void finish() override {}
    std::vector<double> optical_depths() override { return tau_; }
    std::vector<double> cumulative_sums() override { return cumulative_; }
    BackendTiming timing() const override { return timing_; }
};
} // namespace

std::unique_ptr<Backend> make_cpu_backend(const Header& h, const Spectrum& f,
    const std::vector<Source>& s, const BackendSettings& o) {
    return std::make_unique<CpuBackend>(h, f, s, o);
}
} // namespace lya
