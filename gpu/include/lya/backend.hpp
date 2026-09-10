#pragma once

#include "types.hpp"
#include <memory>

namespace lya {
// The backend keeps only [source, ray, frequency] optical depths between slabs.
// Two raw/derived slots allow host reading and transfers to overlap execution.
class Backend {
public:
    virtual ~Backend() = default;
    virtual void begin(const Region&, std::uint64_t max_depth) = 0;
    virtual void submit(const RawSlab&, const Batch&) = 0;
    virtual void finish() = 0;
    virtual std::vector<double> optical_depths() = 0;
    // Flattened [source, 5, total_input_depth], spatial SUM of transmission.
    // Only observerward cells belonging to a source contain contributions.
    virtual std::vector<double> cumulative_sums() = 0;
    virtual BackendTiming timing() const = 0;
};

std::unique_ptr<Backend> make_cpu_backend(const Header&, const Spectrum&,
    const std::vector<Source>&, const BackendSettings&);
std::unique_ptr<Backend> make_cuda_backend(const Header&, const Spectrum&,
    const std::vector<Source>&, const BackendSettings&);
std::uint64_t cuda_free_memory(int device);
} // namespace lya
