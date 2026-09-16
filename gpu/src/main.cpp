#include "lya/backend.hpp"
#include "lya/config.hpp"
#include "lya/io.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace lya {
namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
constexpr long double mib=1024.L*1024.L;
struct MemoryPlan {
    std::uint64_t tile=1, depth=1;
    long double host=0, device=0;
};
MemoryPlan plan_memory(const Config& c,const Input& input,const Region& region,std::size_t sources,std::size_t nf,
                       long double device_budget) {
    MemoryPlan p;
    p.tile=std::min(c.tile_size,std::max(region.x1-region.x0,region.y1-region.y0));
    p.depth=std::min(c.depth_slab,input.header.depth);
    double max_width=0;
    for (std::size_t j=0;j+1<input.redshifts.size();++j)
        max_width=std::max(max_width,std::log1p((input.redshifts[j]-input.redshifts[j+1])/(1.+input.redshifts[j+1])));
    long double subdivisions=c.legacy?1.L:std::ceil(max_width/c.max_dln);
    auto estimate=[&]() {
        long double rays=static_cast<long double>(std::min(p.tile,region.x1-region.x0))*std::min(p.tile,region.y1-region.y0);
        long double cells=rays*p.depth;
        long double tau=8.L*sources*rays*nf;
        long double cumulative=c.cumulative?40.L*sources*input.header.depth:0;
        long double segments=sources*p.depth*subdivisions*sizeof(Segment);
        long double fixed=8.L*mib+4.L*nf*sizeof(Frequency)+sources*sizeof(Source)*4.L;
        // Includes caller vectors, two pinned slots, backend state, returned
        // spectra/cumulative copies, segment arrays and per-tile band maps.
        p.host=144.L*cells+2.L*tau+4.L*cumulative+5.L*segments+40.L*rays+fixed
            +16.L*input.redshifts.size();
        p.device=144.L*cells+tau+cumulative+2.L*segments+fixed;
    };
    estimate();
    while (p.host>c.host_mib*mib || p.device>device_budget) {
        if (p.tile>1) p.tile=(p.tile+1)/2;
        else if (p.depth>1) p.depth=(p.depth+1)/2;
        else throw std::runtime_error("Memory budget cannot fit one ray/slab and all source states; increase budgets or split the source list");
        estimate();
    }
    std::cout<<"Working buffers: host <= "<<static_cast<double>(p.host/mib)<<" MiB, device <= "
        <<static_cast<double>(p.device/mib)<<" MiB\nInternal tile: "<<p.tile<<"; depth slab: "<<p.depth<<'\n';
    std::cout<<"Budgets cover working buffers; reserve additional memory for HDF5 and runtime overhead.\n";
    return p;
}

std::filesystem::path output_path(const Config& c,const Source& source,const Region& region,const Header& h) {
    auto z=number(source.requested);
    std::string suffix=c.chunk>=0?std::to_string(c.chunk):"0";
    if (c.has_region && !(region.x0==0 && region.y0==0 && region.x1==h.pixels && region.y1==h.pixels))
        suffix="region_"+std::to_string(region.x0)+"_"+std::to_string(region.x1)+"_"+
            std::to_string(region.y0)+"_"+std::to_string(region.y1);
    return std::filesystem::path(c.output_dir)/("z0="+z)/("tau_map_"+z+"_"+suffix+".hdf5");
}

void run(const Config& config) {
    auto wall_start=Clock::now();
    Input input(config);
    input.describe();
    auto region=select_region(config,input.header);
    auto grid=make_spectrum(config);
    std::vector<Source> sources;
    if (!config.source_file.empty()) sources=read_sources(config,input.redshifts);
    bool inspect=config.command=="inspect" || config.dry_run;
    std::cout<<"Spectral channels: "<<grid.frequencies.size()<<"; requested sources: "<<sources.size()<<'\n';
    if (!inspect && config.resume) {
        sources.erase(std::remove_if(sources.begin(),sources.end(),[&](const Source& s) {
            auto path=output_path(config,s,region,input.header);
            if (!std::filesystem::exists(path)) return false;
            if (!Product::matches(path,Product::signature(input,config,region,s,true)))
                throw std::runtime_error("Resume configuration does not match completed output: "+path.string());
            std::cout<<"Resume: already complete "<<path<<'\n'; return true;
        }),sources.end());
        if (sources.empty()) return;
    }
    long double device_budget=config.device_mib?config.device_mib*mib:1024.L*mib;
    if (!inspect && config.backend=="cuda") {
        auto free=cuda_free_memory(config.device);
        device_budget=std::min(device_budget,static_cast<long double>(free)*.8L);
        if (!config.device_mib) device_budget=static_cast<long double>(free)*.8L;
    }
    if (config.backend=="cpu") device_budget=std::numeric_limits<long double>::infinity();
    auto plan=plan_memory(config,input,region,std::max<std::size_t>(1,sources.size()),grid.frequencies.size(),device_budget);
    long double bytes=8.L*sources.size()*region.rays()*(5+(config.spectra?grid.frequencies.size():0));
    std::cout<<"Map/spectral dataset storage: "<<static_cast<double>(bytes/mib/1024)<<" GiB (before metadata)\n";
    if (inspect) {
        std::cout<<"Metadata-only inspection complete; no gas fields or GPU context were loaded.\n";
        return;
    }
    std::vector<std::unique_ptr<Product>> products;
    for (const auto& source:sources)
        products.push_back(std::make_unique<Product>(output_path(config,source,region,input.header),input,config,region,source,grid));
    BackendSettings settings{config.sampling,config.profile,config.legacy,config.cumulative,config.device};
    auto backend=config.backend=="cpu"?make_cpu_backend(input.header,grid,sources,settings)
                                     :make_cuda_backend(input.header,grid,sources,settings);
    std::vector<double> cumulative;
    if (config.cumulative) cumulative.resize(sources.size()*5*input.header.depth,0.);
    double reading=0,writing=0;
    auto first_cell=input.header.depth;
    for (const auto& source:sources) first_cell=std::min(first_cell,source.start_cell);
    std::uint64_t tiles=0;
    for (auto x=region.x0;x<region.x1;x+=plan.tile) for (auto y=region.y0;y<region.y1;y+=plan.tile) {
        Region tile{x,std::min(x+plan.tile,region.x1),y,std::min(y+plan.tile,region.y1)};
        backend->begin(tile,plan.depth);
        // Clip the first read at the earliest source, then end slabs on storage
        // boundaries. Unused upstream gas is neither loaded nor validated.
        for (auto z=first_cell;z<input.header.depth;) {
            auto depth=std::min(plan.depth-z%plan.depth,input.header.depth-z);
            auto batch=make_batch(config,input.header,input.redshifts,input.distances,sources,z,depth);
            auto t=Clock::now();
            auto raw=input.read_slab(tile,z,depth);
            reading+=seconds(t);
            backend->submit(raw,batch);
            z+=depth;
        }
        backend->finish();
        auto tau=backend->optical_depths();
        if (config.cumulative) {
            auto sums=backend->cumulative_sums();
            for (std::size_t j=0;j<sums.size();++j) cumulative[j]+=sums[j];
        }
        auto t=Clock::now();
        for (std::size_t s=0;s<sources.size();++s)
            products[s]->write_tile(tile,grid,tau.data()+s*tile.rays()*grid.frequencies.size());
        writing+=seconds(t);
        ++tiles;
        std::cout<<"Completed tile "<<tiles<<": x=["<<tile.x0<<','<<tile.x1<<"), y=["<<tile.y0<<','<<tile.y1<<")\n";
    }
    auto timings=backend->timing();
    for (std::size_t s=0;s<sources.size();++s) {
        std::vector<double> source_sum;
        if (config.cumulative) source_sum.assign(cumulative.begin()+s*5*input.header.depth,cumulative.begin()+(s+1)*5*input.header.depth);
        products[s]->complete(source_sum,reading,timings.transfer_seconds,timings.kernel_seconds,writing);
    }
    std::cout<<"Timing seconds: read="<<reading<<" transfer="<<timings.transfer_seconds
        <<" kernel="<<timings.kernel_seconds<<" write="<<writing<<" wall="<<seconds(wall_start)<<'\n';
    std::cout<<"Timing covers this region and source batch; extrapolate only from representative compute-node measurements.\n";
}
} // namespace
} // namespace lya

int main(int argc,char** argv) {
    H5Eset_auto2(H5E_DEFAULT,nullptr,nullptr);
    try {
        auto config=lya::parse_config(argc,argv);
        if (config.command=="help") { lya::print_help(); return 0; }
        if (config.command=="merge") lya::merge_products(config);
        else lya::run(config);
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"lumina_lya: "<<e.what()<<'\n';
        return 1;
    }
}
