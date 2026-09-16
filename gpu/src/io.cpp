#include "lya/io.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unistd.h>

namespace lya {
namespace fs = std::filesystem;
namespace {
std::string file_identity(const fs::path& path) {
    auto p=fs::canonical(path);
    return p.string()+":"+std::to_string(fs::file_size(p))+":"+
        std::to_string(fs::last_write_time(p).time_since_epoch().count());
}
std::string hash_string(const std::string& s) {
    std::uint64_t hash=14695981039346656037ULL;
    for (unsigned char ch:s) { hash^=ch; hash*=1099511628211ULL; }
    std::ostringstream out; out<<std::hex<<hash; return out.str();
}
void validate_numeric(hid_t ds, const std::string& name) {
    h5::Handle type(H5Dget_type(ds), H5Tclose);
    if (H5Tget_class(type)!=H5T_FLOAT || (H5Tget_size(type)!=4 && H5Tget_size(type)!=8))
        throw std::runtime_error(name+" must contain float32 or float64 data");
}
void validate_virtual(hid_t ds, const fs::path& parent, std::string& identity,
                      std::vector<hsize_t>* storage_chunks = nullptr) {
    h5::Handle plist(H5Dget_create_plist(ds), H5Pclose);
    auto layout=H5Pget_layout(plist);
    if (layout==H5D_CHUNKED && storage_chunks) {
        storage_chunks->resize(h5::shape(ds).size());
        h5::check(H5Pget_chunk(plist,static_cast<int>(storage_chunks->size()),storage_chunks->data()),"reading chunk dimensions");
    }
    if (layout!=H5D_VIRTUAL) return;
    std::size_t n=0;
    h5::check(H5Pget_virtual_count(plist,&n),"reading virtual mappings");
    for (std::size_t i=0;i<n;++i) {
        auto length=H5Pget_virtual_filename(plist,i,nullptr,0);
        auto dlength=H5Pget_virtual_dsetname(plist,i,nullptr,0);
        if (length<0 || dlength<0) throw std::runtime_error("Unreadable virtual mapping");
        std::vector<char> fname(length+1), dname(dlength+1);
        H5Pget_virtual_filename(plist,i,fname.data(),fname.size());
        H5Pget_virtual_dsetname(plist,i,dname.data(),dname.size());
        fs::path source(fname.data());
        if (source.string().find('%')!=std::string::npos || std::string(dname.data()).find('%')!=std::string::npos)
            throw std::runtime_error("Printf-style virtual sources require an explicit finite VDS mapping");
        h5::Handle owner;
        if (source==".") {
            owner=h5::Handle(H5Iget_file_id(ds),H5Fclose);
        } else {
            if (source.is_relative()) source=parent/source;
            if (!fs::is_regular_file(source)) throw std::runtime_error("Missing virtual source: "+source.string());
            identity+=';'+file_identity(source);
            owner=h5::open(source);
        }
        auto linked=h5::dataset(owner,dname.data());
        validate_numeric(linked,dname.data());
        // Lumina uses one-level virtual mappings. Reject nested VDS rather than
        // silently reading fill values from missing nested sources.
        h5::Handle sp(H5Dget_create_plist(linked),H5Pclose);
        if (H5Pget_layout(sp)==H5D_VIRTUAL) throw std::runtime_error("Nested VDS sources are not supported: "+source.string());
        auto actual_dims=h5::shape(linked);
        h5::Handle source_space(H5Pget_virtual_srcspace(plist,i),H5Sclose);
        h5::Handle virtual_space(H5Pget_virtual_vspace(plist,i),H5Sclose);
        auto virtual_points=H5Sget_select_npoints(virtual_space);
        auto source_points=H5Sget_select_npoints(source_space);
        if (virtual_points<0 || source_points<0)
            throw std::runtime_error("Virtual mappings must use finite selections");
        if (H5Sget_select_type(source_space)==H5S_SEL_ALL) {
            // HDF5 represents a whole-source selection using a scalar ALL
            // token. Its apparent rank/point count are not the source shape.
            if (h5::count(actual_dims)!=static_cast<std::uint64_t>(virtual_points))
                throw std::runtime_error("Whole VDS source dimensions disagree with mapping: "+source.string());
        } else if (source_points) {
            if (H5Sget_simple_extent_ndims(source_space)!=static_cast<int>(actual_dims.size()) || source_points!=virtual_points)
                throw std::runtime_error("VDS source selection dimensions disagree with mapping: "+source.string());
            std::vector<hsize_t> lower(actual_dims.size()),upper(actual_dims.size());
            h5::check(H5Sget_select_bounds(source_space,lower.data(),upper.data()),"reading VDS source selection");
            for (std::size_t axis=0;axis<actual_dims.size();++axis)
                if (upper[axis]>=actual_dims[axis])
                    throw std::runtime_error("VDS source selection exceeds dataset extent: "+source.string());
        } else if (virtual_points) {
            throw std::runtime_error("Empty VDS source selection has a nonempty destination");
        }
        if (storage_chunks && H5Pget_layout(sp)==H5D_CHUNKED) {
            storage_chunks->resize(h5::shape(linked).size());
            H5Pget_chunk(sp,static_cast<int>(storage_chunks->size()),storage_chunks->data());
        }
    }
}
void common_attributes(hid_t f, const Header& h, const Region& r, const Source& s, const Config& c) {
    h5::put_attr(f,"HubbleParam",h.hubble_param); h5::put_attr(f,"Omega0",h.omega_m);
    h5::put_attr(f,"OmegaBaryon",h.omega_b); h5::put_index(f,"NumPixels",h.pixels);
    h5::put_attr(f,"Redshift",s.actual); h5::put_attr(f,"SourceRedshift",s.actual);
    h5::put_attr(f,"SourceRedshiftRequested",s.requested); h5::put_index(f,"SourceCell",s.start_cell);
    h5::put_index(f,"Chunk",c.chunk>=0?c.chunk:0); h5::put_index(f,"ChunkSize",r.x1-r.x0);
    h5::put_index(f,"X0",r.x0); h5::put_index(f,"X1",r.x1); h5::put_index(f,"Y0",r.y0); h5::put_index(f,"Y1",r.y1);
    h5::put_index(f,"PixelCount",r.rays()); h5::put_attr(f,"Dv_min",-2000); h5::put_attr(f,"Dv_max",2000);
    h5::put_attr(f,"Dv_local",0);
}
} // namespace

Input::Input(const Config& cfg) : path(fs::canonical(cfg.input)), file(h5::open(path)),
    density(h5::dataset(file,"Density")), ionized(h5::dataset(file,"HII_Fraction")),
    temperature(h5::dataset(file,"Temperature")), velocity(h5::dataset(file,"Velocities")) {
    h5::Handle hg(H5Gopen2(file,"Header",H5P_DEFAULT),H5Gclose);
    auto dims=h5::shape(density);
    if (dims.size()!=3 || !dims[0] || dims[0]!=dims[1] || !dims[2])
        throw std::runtime_error("Expected nonempty square Density[x,y,depth]");
    (void)h5::count(dims); // Validate full-grid indexing without reading gas.
    if (dims[2]==UINT64_MAX) throw std::runtime_error("Depth edge count overflows 64-bit indexing");
    header.pixels=dims[0]; header.depth=dims[2];
    if (h5::attr(hg,"NumPixels")!=header.pixels) throw std::runtime_error("NumPixels disagrees with dataset shape");
    if (H5Aexists(hg,"NumDepth")>0 && h5::attr(hg,"NumDepth")!=header.depth)
        throw std::runtime_error("NumDepth disagrees with dataset shape");
    header.hubble_param=h5::attr(hg,"HubbleParam"); header.omega_m=h5::attr(hg,"Omega0");
    header.omega_b=h5::attr(hg,"OmegaBaryon"); header.opening_angle=h5::attr(hg,"OpeningAngle");
    header.unit_length=h5::attr(hg,"UnitLength_in_cm"); header.unit_mass=h5::attr(hg,"UnitMass_in_g");
    header.unit_velocity=h5::attr(hg,"UnitVelocity_in_cm_per_s");
    for (double v:{header.hubble_param,header.omega_m,header.omega_b,header.unit_length,header.unit_mass,header.unit_velocity,header.opening_angle})
        if (!std::isfinite(v) || v<=0) throw std::runtime_error("Invalid Header physical parameter");
    if (header.opening_angle>=3.141592653589793) throw std::runtime_error("OpeningAngle must be in radians and less than pi");
    if (h5::shape(ionized)!=dims || h5::shape(temperature)!=dims ||
        h5::shape(velocity)!=std::vector<hsize_t>{dims[0],dims[1],dims[2],3})
        throw std::runtime_error("Lightcone gas field shapes disagree");
    std::string fingerprint=file_identity(path);
    for (auto pair:std::vector<std::pair<hid_t,std::string>>{{density,"Density"},{ionized,"HII_Fraction"},{temperature,"Temperature"},{velocity,"Velocities"}}) {
        validate_numeric(pair.first,pair.second);
        validate_virtual(pair.first,path.parent_path(),fingerprint,pair.second=="Density"?&storage_chunks:nullptr);
    }
    auto zds=h5::dataset(file,"Redshifts");
    if (h5::shape(zds)!=std::vector<hsize_t>{header.depth+1}) throw std::runtime_error("Redshifts must contain depth+1 cell edges");
    redshifts=h5::read(zds);
    for (std::size_t j=0;j<redshifts.size();++j)
        if (!std::isfinite(redshifts[j]) || redshifts[j]<=-1 || (j && redshifts[j]>=redshifts[j-1]))
            throw std::runtime_error("Redshifts must be finite, greater than -1, and strictly decreasing");
    if (!cfg.legacy) {
        auto ds=h5::dataset(file,"Distances");
        if (h5::shape(ds)!=std::vector<hsize_t>{header.depth+1}) throw std::runtime_error("Distances must contain depth+1 cell edges");
        distances=h5::read(ds);
        double factor=h5::attr_or(ds,"to_cgs",header.unit_length/header.hubble_param);
        if (!std::isfinite(factor) || factor<=0) throw std::runtime_error("Invalid distance conversion");
        for (std::size_t j=0;j<distances.size();++j) {
            distances[j]*=factor;
            if (!std::isfinite(distances[j]) || (j && distances[j]>=distances[j-1]))
                throw std::runtime_error("Distances must be finite and strictly decreasing");
        }
    }
    identity=hash_string(fingerprint);
}

RawSlab Input::read_slab(const Region& r, std::uint64_t z0, std::uint64_t depth) {
    RawSlab out; out.region=r; out.z0=z0; out.depth=depth;
    std::vector<hsize_t> start{r.x0,r.y0,z0}, shape{r.x1-r.x0,r.y1-r.y0,depth};
    out.density=h5::slice(density,start,shape); out.ionized=h5::slice(ionized,start,shape);
    out.temperature=h5::slice(temperature,start,shape);
    start.push_back(0); shape.push_back(3); out.velocity=h5::slice(velocity,start,shape);
    return out;
}
void Input::describe() const {
    std::cout<<"Input: "<<path<<"\nGrid: "<<header.pixels<<" x "<<header.pixels<<" x "<<header.depth
        <<"\nRedshift coverage: "<<std::setprecision(12)<<redshifts.front()<<" to "<<redshifts.back()<<"\nStorage chunks:";
    for (auto n:storage_chunks) std::cout<<' '<<n;
    std::cout<<"\nInput identity: "<<identity<<'\n';
}

std::string Product::signature(const Input& in, const Config& c, const Region& r, const Source& s, bool region) {
    std::string value=configuration(c,r,region)+";input="+in.identity+";source="+number(s.requested);
    for (double z:s.spread) value+=";spread="+number(z);
    return value;
}
bool Product::matches(const fs::path& path,const std::string& signature) {
    auto f=h5::open(path);
    return H5Aexists(f,"Complete")>0 && h5::attr(f,"Complete")==1 &&
        H5Aexists(f,"RunSignature")>0 && h5::string_attr(f,"RunSignature")==signature;
}

Product::Product(const fs::path& target,const Input& in,const Config& c,const Region& r,const Source& s,const Spectrum& grid)
    : target_(target),config_(c),region_(r),source_(s),depth_(in.header.depth),total_transmission_(5,0) {
    fs::create_directories(target.parent_path());
    lock_=target; lock_+=".lock";
    if (!fs::create_directory(lock_)) throw std::runtime_error("Output is locked: "+target.string());
    try {
        if (fs::exists(target_) && !c.overwrite) throw std::runtime_error("Output exists: "+target_.string());
        temporary_=target_; temporary_+=".partial."+std::to_string(getpid());
        file_=h5::Handle(H5Fcreate(temporary_.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT),H5Fclose);
        common_attributes(file_,in.header,r,s,c);
        if (!c.legacy) {
            double distance=in.distances.back();
            if (s.start_cell<in.header.depth) {
                auto j=s.start_cell;
                double fraction=(s.actual-in.redshifts[j+1])/(in.redshifts[j]-in.redshifts[j+1]);
                distance=in.distances[j+1]+fraction*(in.distances[j]-in.distances[j+1]);
            }
            h5::put_attr(file_,"SourceDistanceComovingCm",distance);
        }
        h5::put_string(file_,"RunSignature",signature(in,c,r,s,true));
        h5::put_string(file_,"MergeSignature",signature(in,c,r,s,false));
        h5::put_string(file_,"Configuration",configuration(c,r,true));
        h5::put_string(file_,"InputPath",in.path.string()); h5::put_string(file_,"InputIdentity",in.identity);
        h5::put_string(file_,"Sampling",c.sampling==Sampling::Band?"band":"comb");
        h5::put_string(file_,"Profile",c.profile==Profile::Voigt?"voigt":"delta");
        h5::put_string(file_,"SpectralRepresentation",c.sampling==Sampling::Band?"bin-effective optical depth":"sample optical depth");
        h5::put_string(file_,"FrequencyConvention",c.legacy?"legacy wavelength-offset approximation":"nu_source=nu0/(1+Dv/c); frequency-width weights");
        h5::put_string(file_,"Geometry",c.legacy?"matter-dominated, upstream cell values":"tabulated chi(z); midpoint-anchored linear cell drift");
        h5::put_string(file_,"CumulativeCoordinate","downstream input-cell redshift edge");
        h5::put_string(file_,"Precision","float64; no fast math");
        h5::put_index(file_,"Legacy",c.legacy); h5::put_index(file_,"VoigtExpansionOrder",c.legacy?1:2);
        h5::put_attr(file_,"MaximumSupportedVoigtDamping",0.1);
        h5::put_attr(file_,"MaxDln",c.max_dln); h5::put_attr(file_,"DvStep",c.dv_step);
        h5::put_index(file_,"NumFreq",grid.frequencies.size());
        std::vector<double> dvs,qlo,qhi,weights;
        for (const auto& f:grid.frequencies) { dvs.push_back(f.velocity); qlo.push_back(f.q_lo); qhi.push_back(f.q_hi); }
        for (int b=0;b<5;++b) for (const auto& f:grid.frequencies) weights.push_back(f.weights[b]);
        h5::vector(file_,"Dvs",dvs); h5::vector(file_,"Dv_edges",grid.velocity_edges);
        h5::vector(file_,"FrequencyLowerOverNu0",qlo); h5::vector(file_,"FrequencyUpperOverNu0",qhi);
        h5::vector(file_,"freq_band_edges",grid.band_edges); h5::vector(file_,"freq_bands",grid.band_edges);
        auto wd=h5::create(file_,"frequency_weights",{5,grid.frequencies.size()});
        h5::write(wd,{0,0},{5,grid.frequencies.size()},weights.data());
        if (s.spread.size()>1) h5::vector(file_,"z_spread",s.spread);
        auto nx=r.x1-r.x0,ny=r.y1-r.y0;
        maps_=h5::create(file_,"tau_band_avgs",{5,nx,ny},{1,std::min<std::uint64_t>(nx,128),std::min<std::uint64_t>(ny,128)});
        if (c.spectra) spectra_=h5::create(file_,"taus",{nx,ny,grid.frequencies.size()},
            {std::min<std::uint64_t>(nx,16),std::min<std::uint64_t>(ny,16),std::min<std::size_t>(grid.frequencies.size(),128)});
        if (c.cumulative) {
            auto first=std::min<std::uint64_t>(s.start_cell+1,in.redshifts.size());
            std::vector<double> downstream(in.redshifts.begin()+first,in.redshifts.end());
            h5::vector(file_,"T_redshifts",downstream);
        }
    } catch (...) {
        file_=h5::Handle();
        if (!temporary_.empty()) { std::error_code ec; fs::remove(temporary_,ec); }
        std::error_code ec; fs::remove(lock_,ec);
        throw;
    }
}
Product::~Product() {
    maps_=h5::Handle(); spectra_=h5::Handle(); file_=h5::Handle();
    std::error_code ec;
    if (!committed_ && !temporary_.empty()) fs::remove(temporary_,ec);
    if (!lock_.empty()) fs::remove(lock_,ec);
}
void Product::write_tile(const Region& tile,const Spectrum& grid,const double* tau) {
    auto rays=tile.rays(), nf=grid.frequencies.size();
    std::vector<double> bands(5*rays);
    for (std::uint64_t ray=0;ray<rays;++ray) for (int b=0;b<5;++b) {
        double minimum=std::numeric_limits<double>::infinity();
        for (std::size_t f=0;f<nf;++f) if (grid.frequencies[f].weights[b]>0) minimum=std::min(minimum,tau[ray*nf+f]);
        double value=minimum;
        if (std::isfinite(minimum)) {
            double sum=0;
            for (std::size_t f=0;f<nf;++f) if (grid.frequencies[f].weights[b]>0)
                sum+=grid.frequencies[f].weights[b]*std::exp(minimum-tau[ray*nf+f]);
            value=std::max(0.,minimum-std::log(sum));
        }
        bands[b*rays+ray]=value; total_transmission_[b]+=std::exp(-value);
    }
    auto nx=tile.x1-tile.x0,ny=tile.y1-tile.y0;
    h5::write(maps_,{0,tile.x0-region_.x0,tile.y0-region_.y0},{5,nx,ny},bands.data());
    if (config_.spectra) h5::write(spectra_,{tile.x0-region_.x0,tile.y0-region_.y0,0},{nx,ny,nf},tau);
}
void Product::complete(const std::vector<double>& cumulative,double reading,double transfer,double kernel,double writing) {
    std::vector<double> means=total_transmission_;
    for (auto& value:means) value/=region_.rays();
    if (config_.cumulative) {
        auto length=depth_-source_.start_cell;
        std::vector<double> data(5*length);
        for (int b=0;b<5;++b) for (std::uint64_t j=0;j<length;++j)
            data[b*length+j]=cumulative[b*depth_+source_.start_cell+j]/region_.rays();
        for (int b=0;b<5;++b) {
            double last=length?data[b*length+length-1]:1.;
            if (std::abs(last-means[b])>1e-8*(1.+std::abs(last)))
                throw std::runtime_error("Cumulative final transmission disagrees with map average");
            means[b]=last;
        }
        auto ds=h5::create(file_,"T_cum_bands",{5,length});
        h5::write(ds,{0,0},{5,length},data.data());
    }
    h5::vector(file_,"T_bands",means);
    h5::put_attr(file_,"ReadSeconds",reading); h5::put_attr(file_,"TransferSeconds",transfer);
    h5::put_attr(file_,"KernelSeconds",kernel); h5::put_attr(file_,"WriteSeconds",writing);
    h5::put_attr(file_,"Complete",1);
    h5::check(H5Fflush(file_,H5F_SCOPE_GLOBAL),"flushing completed output");
    maps_=h5::Handle(); spectra_=h5::Handle(); file_=h5::Handle();
    if (!config_.overwrite && fs::exists(target_)) throw std::runtime_error("Output appeared while processing: "+target_.string());
    fs::rename(temporary_,target_); committed_=true;
    std::cout<<"Wrote "<<target_<<'\n';
}
} // namespace lya
