#include "lya/config.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace lya {
namespace {
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::stringstream input(s);
    std::vector<std::string> out;
    std::string part;
    while (std::getline(input, part, delimiter)) out.push_back(part);
    return out;
}
double real(const std::string& value) {
    std::size_t used = 0;
    double x = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(x)) throw std::runtime_error("Invalid number: " + value);
    return x;
}
std::uint64_t integer(const std::string& value) {
    if (value.empty() || value[0] == '-') throw std::runtime_error("Expected nonnegative integer: " + value);
    std::size_t used = 0;
    auto x = std::stoull(value, &used);
    if (used != value.size()) throw std::runtime_error("Invalid integer: " + value);
    return x;
}
} // namespace

std::string number(double x) {
    std::ostringstream out;
    out << std::setprecision(17) << x;
    return out.str();
}

void print_help() {
    std::cout <<
        "lumina_lya inspect --input LIGHTCONE [--z0-file SOURCES] [run options]\n"
        "lumina_lya run --input LIGHTCONE --z0-file SOURCES --output-dir DIRECTORY [options]\n"
        "lumina_lya merge --output FILE TILE_FILE...\n\n"
        "Options:\n"
        "  --sampling band|comb       Default band\n"
        "  --profile voigt|delta      Default voigt\n"
        "  --dv-step KM/S             Maximum spectral spacing (default 5)\n"
        "  --max-dln VALUE            Maximum ln(1+z) interval (default 0.001)\n"
        "  --products maps[,spectra][,cumulative]\n"
        "  --region X0,X1,Y0,Y1       Half-open spatial bounds\n"
        "  --chunk INDEX              Historical spatial chunk selection\n"
        "  --tile-size N --depth-slab N   Maximum internal buffer dimensions (128)\n"
        "  --host-memory-mib N        Host working-buffer budget (default 1024)\n"
        "  --device-memory-mib N      Device budget (default 80% free memory)\n"
        "  --device INDEX --backend cuda|cpu\n"
        "  --legacy                  Historical working Python physics/sampling\n"
        "  --resume | --overwrite    Explicit handling of existing products\n"
        "  --dry-run                 Metadata and memory planning only; no GPU calls\n"
        "The CPU backend is intended for tiny validation fixtures. No GPU work is\n"
        "performed by inspect, dry-run, or compilation. Run CUDA jobs via Slurm.\n";
}

Config parse_config(int argc, char** argv) {
    Config cfg;
    if (argc < 2) { cfg.command = "help"; return cfg; }
    cfg.command = argv[1];
    if (cfg.command == "--help" || cfg.command == "-h") cfg.command = "help";
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") { cfg.command = "help"; return cfg; }
        if (key == "--legacy") { cfg.legacy = true; continue; }
        if (key == "--resume") { cfg.resume = true; continue; }
        if (key == "--overwrite") { cfg.overwrite = true; continue; }
        if (key == "--dry-run") { cfg.dry_run = true; continue; }
        if (cfg.command == "merge" && key.rfind("--", 0) != 0) { cfg.merge_inputs.push_back(key); continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        std::string value = argv[++i];
        if (key == "--input") cfg.input = value;
        else if (key == "--z0-file") cfg.source_file = value;
        else if (key == "--output-dir") cfg.output_dir = value;
        else if (key == "--output") cfg.output = value;
        else if (key == "--backend") cfg.backend = value;
        else if (key == "--sampling") {
            if (value != "band" && value != "comb") throw std::runtime_error("sampling must be band or comb");
            cfg.sampling = value == "band" ? Sampling::Band : Sampling::Comb;
        } else if (key == "--profile") {
            if (value != "delta" && value != "voigt") throw std::runtime_error("profile must be voigt or delta");
            cfg.profile = value == "voigt" ? Profile::Voigt : Profile::Delta;
        } else if (key == "--dv-step") cfg.dv_step = real(value);
        else if (key == "--max-dln") cfg.max_dln = real(value);
        else if (key == "--tile-size") cfg.tile_size = integer(value);
        else if (key == "--depth-slab") cfg.depth_slab = integer(value);
        else if (key == "--host-memory-mib") cfg.host_mib = real(value);
        else if (key == "--device-memory-mib") cfg.device_mib = real(value);
        else if (key == "--device") {
            auto n = integer(value);
            if (n > 1024) throw std::runtime_error("Invalid device index");
            cfg.device = static_cast<int>(n);
        } else if (key == "--chunk") {
            auto n = integer(value);
            if (n > static_cast<std::uint64_t>(INT64_MAX)) throw std::runtime_error("Chunk index overflow");
            cfg.chunk = static_cast<std::int64_t>(n);
        } else if (key == "--region") {
            auto parts = split(value, ',');
            if (parts.size() != 4) throw std::runtime_error("region requires X0,X1,Y0,Y1");
            cfg.region = {integer(parts[0]), integer(parts[1]), integer(parts[2]), integer(parts[3])};
            cfg.has_region = true;
        } else if (key == "--products") {
            for (const auto& p : split(value, ',')) {
                if (p == "spectra") cfg.spectra = true;
                else if (p == "cumulative") cfg.cumulative = true;
                else if (p != "maps") throw std::runtime_error("Unknown product: " + p);
            }
        } else throw std::runtime_error("Unknown option: " + key);
    }
    if (cfg.command == "help") return cfg;
    if (cfg.command != "run" && cfg.command != "inspect" && cfg.command != "merge") throw std::runtime_error("Unknown command");
    if (cfg.backend != "cpu" && cfg.backend != "cuda") throw std::runtime_error("backend must be cuda or cpu");
    if (!(cfg.dv_step > 0) || !(cfg.max_dln > 0) || !cfg.tile_size || !cfg.depth_slab ||
        !(cfg.host_mib > 0) || cfg.device_mib < 0) throw std::runtime_error("Spacing and memory limits must be positive");
    if (cfg.dv_step < 1e-4 || cfg.max_dln < 1e-8) throw std::runtime_error("Requested resolution exceeds supported planning bounds");
    if (cfg.resume && cfg.overwrite) throw std::runtime_error("Choose resume or overwrite, not both");
    if (cfg.has_region && cfg.chunk >= 0) throw std::runtime_error("Choose region or chunk, not both");
    if (cfg.command == "merge") {
        if (cfg.output.empty() || cfg.merge_inputs.empty()) throw std::runtime_error("merge requires --output and tile files");
    } else {
        if (cfg.input.empty()) throw std::runtime_error("--input is required");
        if (cfg.command == "run" && (cfg.source_file.empty() || cfg.output_dir.empty()))
            throw std::runtime_error("run requires --z0-file and --output-dir");
    }
    if (cfg.legacy) { cfg.sampling = Sampling::Comb; cfg.profile = Profile::Voigt; }
    return cfg;
}

Spectrum make_spectrum(const Config& cfg) {
    Spectrum grid;
    grid.band_edges = cfg.legacy ? std::vector<double>{-2000,-500,-100,101,501,2001}
                                 : std::vector<double>{-2000,-500,-100,100,500,2000};
    long double estimated_channels=cfg.legacy?801.L:1.L;
    if (!cfg.legacy) for (int b=0;b<5;++b)
        estimated_channels+=std::ceil((grid.band_edges[b+1]-grid.band_edges[b])/cfg.dv_step);
    if (estimated_channels*(sizeof(Frequency)+sizeof(double))*4.L>cfg.host_mib*1024.L*1024.L)
        throw std::runtime_error("Spectral grid exceeds host memory budget; increase --dv-step or memory limit");
    grid.frequencies.reserve(static_cast<std::size_t>(estimated_channels));
    if (cfg.legacy) {
        constexpr int indices[] = {0,300,380,421,501,801};
        for (int i = 0; i <= 800; ++i) {
            Frequency f;
            f.velocity = (-2000. + 5. * i) * km;
            f.q_lo = f.q_hi = 1. / (1. + f.velocity / c_light);
            for (int b = 0; b < 5; ++b)
                if (i >= indices[b] && i < indices[b+1]) f.weights[b] = 1. / (indices[b+1]-indices[b]);
            grid.frequencies.push_back(f);
        }
        return grid;
    }
    for (int b = 0; b < 5; ++b) {
        double left = grid.band_edges[b], right = grid.band_edges[b+1];
        auto n = static_cast<std::size_t>(std::ceil((right-left) / cfg.dv_step));
        double total = 1./(1.+left*km/c_light) - 1./(1.+right*km/c_light);
        for (std::size_t j = 0; j < n; ++j) {
            double u0 = (left + (right-left)*static_cast<double>(j)/n)*km;
            double u1 = (left + (right-left)*static_cast<double>(j+1)/n)*km;
            double q0 = 1./(1.+u0/c_light), q1 = 1./(1.+u1/c_light);
            if (grid.velocity_edges.empty()) grid.velocity_edges.push_back(u0);
            grid.velocity_edges.push_back(u1);
            if (cfg.sampling == Sampling::Band) {
                Frequency f;
                f.velocity = .5*(u0+u1); f.q_lo = q1; f.q_hi = q0;
                f.weights[b] = (q0-q1)/total;
                grid.frequencies.push_back(f);
            } else {
                if (grid.frequencies.empty()) {
                    Frequency f;
                    f.velocity=u0; f.q_lo=f.q_hi=q0;
                    grid.frequencies.push_back(f);
                }
                grid.frequencies.back().weights[b] += .5*(q0-q1)/total;
                Frequency f;
                f.velocity=u1; f.q_lo=f.q_hi=q1; f.weights[b]=.5*(q0-q1)/total;
                grid.frequencies.push_back(f);
            }
        }
    }
    return grid;
}

std::vector<Source> read_sources(const Config& cfg, const std::vector<double>& zs) {
    std::ifstream input(cfg.source_file);
    if (!input) throw std::runtime_error("Cannot read source list: " + cfg.source_file);
    std::vector<Source> sources;
    std::string line;
    while (std::getline(input, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream values(line);
        Source source;
        std::string value;
        while (values >> value) source.spread.push_back(real(value));
        if (source.spread.empty()) continue;
        source.requested = source.actual = source.spread.front();
        if (source.requested > zs.front() || source.requested < zs.back())
            throw std::runtime_error("Source outside lightcone: " + number(source.requested));
        if (source.requested == zs.back()) source.start_cell = zs.size()-1;
        else {
            auto it = std::upper_bound(zs.begin(), zs.end(), source.requested, std::greater<double>());
            source.start_cell = static_cast<std::uint64_t>(it-zs.begin()-1);
            if (cfg.legacy) source.actual = zs[source.start_cell];
        }
        for (const auto& previous : sources)
            if (previous.requested == source.requested) throw std::runtime_error("Duplicate source redshift");
        sources.push_back(source);
    }
    if (sources.empty()) throw std::runtime_error("Empty source list");
    std::sort(sources.begin(), sources.end(), [](const Source& a, const Source& b) { return a.requested > b.requested; });
    return sources;
}

Region select_region(const Config& cfg, const Header& header) {
    Region r = cfg.has_region ? cfg.region : Region{0,header.pixels,0,header.pixels};
    if (cfg.chunk >= 0) {
        auto n = std::min(header.pixels, std::max<std::uint64_t>(header.pixels/128, 4));
        if (static_cast<std::uint64_t>(cfg.chunk) >= n*n) throw std::runtime_error("Chunk index out of range");
        auto ix=cfg.chunk/n, iy=cfg.chunk%n;
        r={ix*header.pixels/n,(ix+1)*header.pixels/n,iy*header.pixels/n,(iy+1)*header.pixels/n};
    }
    if (r.x0 >= r.x1 || r.y0 >= r.y1 || r.x1 > header.pixels || r.y1 > header.pixels)
        throw std::runtime_error("Invalid spatial region");
    return r;
}

Batch make_batch(const Config& cfg, const Header& h, const std::vector<double>& zs,
                 const std::vector<double>& chi, const std::vector<Source>& sources,
                 std::uint64_t z0, std::uint64_t depth) {
    Batch batch;
    for (const auto& source : sources) {
        batch.offsets.push_back(batch.segments.size());
        for (auto j=std::max(z0,source.start_cell); j<z0+depth; ++j) {
            double high=std::min(zs[j],source.actual), low=zs[j+1];
            if (!(high>low)) continue;
            if (cfg.legacy) {
                double H=h.hubble_param*100.*km/3.085677581467192e24*std::sqrt(h.omega_m)*std::pow(1.+zs[j],1.5);
                batch.segments.push_back({j-z0,zs[j],c_light*(zs[j]-zs[j+1])/(H*(1.+zs[j])),H,1});
            } else {
                double slope=(chi[j]-chi[j+1])/(zs[j]-zs[j+1]);
                double width=std::log1p((high-low)/(1.+low));
                auto n=static_cast<std::uint64_t>(std::ceil(width/cfg.max_dln));
                if ((static_cast<long double>(batch.segments.size())+n)*sizeof(Segment)*4.L>cfg.host_mib*1024.L*1024.L)
                    throw std::runtime_error("Segment geometry exceeds host budget; increase --max-dln or reduce --depth-slab");
                double lnlow=std::log1p(low), step=width/n;
                for (std::uint64_t k=0;k<n;++k) {
                    double midpoint=std::expm1(lnlow+width-(k+.5)*step);
                    batch.segments.push_back({j-z0,midpoint,slope*step,c_light/slope,k+1==n});
                }
            }
        }
    }
    batch.offsets.push_back(batch.segments.size());
    return batch;
}

std::string configuration(const Config& c, const Region& r, bool with_region) {
    std::ostringstream out;
    out << "lumina-lya-v1;sampling=" << (c.sampling==Sampling::Band?"band":"comb")
        << ";profile=" << (c.profile==Profile::Voigt?"voigt":"delta") << ";legacy=" << c.legacy
        << ";dv_step=" << number(c.dv_step) << ";max_dln=" << number(c.max_dln)
        << ";spectra=" << c.spectra << ";cumulative=" << c.cumulative;
    if (with_region) out << ";region=" << r.x0 << ',' << r.x1 << ',' << r.y0 << ',' << r.y1;
    return out.str();
}
} // namespace lya
