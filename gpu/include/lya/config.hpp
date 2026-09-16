#pragma once
#include "types.hpp"
#include <filesystem>
#include <map>

namespace lya {
struct Config {
    std::string command, backend = "cuda", input, source_file, output_dir, output;
    std::vector<std::string> merge_inputs;
    Sampling sampling = Sampling::Band;
    Profile profile = Profile::Voigt;
    double dv_step = 5, max_dln = 1e-3;
    std::uint64_t tile_size = 128, depth_slab = 128;
    double host_mib = 1024, device_mib = 0;
    int device = 0;
    std::int64_t chunk = -1;
    bool legacy = false, spectra = false, cumulative = false;
    bool resume = false, overwrite = false, dry_run = false, has_region = false;
    Region region;
};
Config parse_config(int argc, char** argv);
void print_help();
Spectrum make_spectrum(const Config&);
std::vector<Source> read_sources(const Config&, const std::vector<double>& redshifts);
Region select_region(const Config&, const Header&);
Batch make_batch(const Config&, const Header&, const std::vector<double>& redshifts,
                 const std::vector<double>& distances_cgs, const std::vector<Source>&,
                 std::uint64_t z0, std::uint64_t depth);
std::string number(double);
std::string configuration(const Config&, const Region&, bool with_region);
} // namespace lya
