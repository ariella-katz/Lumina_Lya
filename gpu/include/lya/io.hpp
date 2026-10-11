#pragma once
#include "config.hpp"
#include "hdf5.hpp"
#include <memory>

namespace lya {
class Input {
public:
    Header header;
    std::vector<double> redshifts, distances;
    std::vector<hsize_t> storage_chunks;
    std::string identity;
    std::filesystem::path path;
    h5::Handle file, density, ionized, temperature, velocity;
    explicit Input(const Config&);
    RawSlab read_slab(const Region&, std::uint64_t z0, std::uint64_t depth);
    void describe() const;
};

class Product {
    std::filesystem::path target_, temporary_, lock_;
    bool committed_ = false;
    h5::Handle file_, maps_, spectra_, binned_;
    Config config_;
    Region region_;
    Source source_;
    std::uint64_t depth_ = 0;
    std::vector<double> total_transmission_;
    // Spatial binning (inactive when bins_per_side_ == 0)
    std::uint64_t bins_per_side_ = 0, bin_size_ = 0;
    std::vector<double> bin_sum_;             // [bin][frequency]: running sum of exp(-tau)
    std::vector<std::uint64_t> bin_count_;    // [bin]: pixels accumulated so far
public:
    Product(const std::filesystem::path&, const Input&, const Config&, const Region&,
            const Source&, const Spectrum&);
    ~Product();
    static bool matches(const std::filesystem::path&, const std::string& signature);
    static std::string signature(const Input&, const Config&, const Region&, const Source&, bool region);
    void write_tile(const Region&, const Spectrum&, const double* tau);
    void complete(const std::vector<double>& cumulative_sum, double read_seconds,
                  double transfer_seconds, double kernel_seconds, double write_seconds);
};
void merge_products(const Config&);
} // namespace lya
