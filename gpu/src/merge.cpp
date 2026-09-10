#include "lya/io.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unistd.h>

namespace lya {
namespace fs=std::filesystem;
namespace {
constexpr const char* timing_names[]={"ReadSeconds","TransferSeconds","KernelSeconds","WriteSeconds"};
struct Tile {
    fs::path path;
    Region region;
    std::string identity;
    std::array<double,5> means{};
};
struct Metadata {
    std::vector<hsize_t> shape;
    std::vector<double> values;
};
std::uint64_t index_attr(hid_t f,const char* name) {
    double n=h5::attr(f,name);
    if (!std::isfinite(n) || n<0 || std::floor(n)!=n || n>9007199254740991.)
        throw std::runtime_error(std::string("Invalid merge index: ")+name);
    return static_cast<std::uint64_t>(n);
}
std::uint64_t area(const Region& region) {
    auto x=region.x1-region.x0,y=region.y1-region.y0;
    if (y && x>UINT64_MAX/y) throw std::runtime_error("Merge area overflow");
    return x*y;
}
std::string file_identity(const fs::path& path) {
    return path.string()+":"+std::to_string(fs::file_size(path))+":"+
        std::to_string(fs::last_write_time(path).time_since_epoch().count());
}
std::string hash_identities(const std::vector<Tile>& tiles) {
    std::vector<std::string> values;
    for (const auto& tile:tiles) values.push_back(tile.identity);
    std::sort(values.begin(),values.end());
    std::uint64_t hash=14695981039346656037ULL;
    for (const auto& value:values) {
        auto framed=std::to_string(value.size())+":"+value;
        for (unsigned char ch:framed) { hash^=ch; hash*=1099511628211ULL; }
    }
    std::ostringstream out; out<<std::hex<<hash; return out.str();
}
void numeric_dataset(hid_t ds,const std::string& name) {
    h5::Handle type(H5Dget_type(ds),H5Tclose);
    if (H5Tget_class(type)!=H5T_FLOAT || (H5Tget_size(type)!=4 && H5Tget_size(type)!=8))
        throw std::runtime_error("Merge dataset must be float32/float64: "+name);
}
h5::Handle shaped_dataset(hid_t f,const std::string& name,const std::vector<hsize_t>& dims) {
    auto ds=h5::dataset(f,name);
    if (h5::shape(ds)!=dims) throw std::runtime_error("Merge dataset shape mismatch: "+name);
    numeric_dataset(ds,name);
    return ds;
}
void transmission(double value,const std::string& name) {
    if (!std::isfinite(value) || value < -1e-12 || value > 1.+1e-12)
        throw std::runtime_error("Merge transmission must be finite and in [0,1]: "+name);
}
void memory_budget(const Config& c,std::uint64_t nf,std::uint64_t depth,std::uint64_t spread=0) {
    // Both the reference coordinates and a comparison/read buffer can coexist.
    // The fixed allowance covers bounded map/spectrum slabs and bookkeeping.
    long double bytes=8.L*(20.L*nf+16.L*depth+2.L*spread)+4.L*1024*1024;
    long double budget=static_cast<long double>(c.host_mib)*1024*1024;
    if (!std::isfinite(c.host_mib) || c.host_mib<=0 || bytes>budget ||
        bytes>static_cast<long double>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("Merge metadata and cumulative buffers exceed --host-memory-mib");
}
void copy_attributes(hid_t source,hid_t target) {
    const std::set<std::string> excluded={"X0","X1","Y0","Y1","PixelCount","Chunk","ChunkSize",
        "RunSignature","Configuration","MergedTileCount","Complete",
        "ReadSeconds","TransferSeconds","KernelSeconds","WriteSeconds"};
    int count=H5Aget_num_attrs(source);
    if (count<0) throw std::runtime_error("Cannot read merge attributes");
    for (int i=0;i<count;++i) {
        h5::Handle a(H5Aopen_by_idx(source,".",H5_INDEX_NAME,H5_ITER_INC,i,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
        auto len=H5Aget_name(a,0,nullptr);
        if (len<0) throw std::runtime_error("Cannot read attribute name");
        std::vector<char> buf(len+1);
        H5Aget_name(a,buf.size(),buf.data());
        std::string name(buf.data());
        if (excluded.count(name)) continue;
        h5::Handle type(H5Aget_type(a),H5Tclose);
        if (H5Tget_class(type)==H5T_STRING) h5::put_string(target,name,h5::string_attr(source,name));
        else if (H5Tget_class(type)==H5T_INTEGER) {
            h5::Handle space(H5Aget_space(a),H5Sclose);
            if (H5Sget_simple_extent_npoints(space)!=1 || H5Tget_size(type)>8)
                throw std::runtime_error("Expected one integer value in attribute "+name);
            if (H5Tget_sign(type)==H5T_SGN_NONE) {
                std::uint64_t value;
                h5::check(H5Aread(a,H5T_NATIVE_UINT64,&value),"reading integer "+name);
                h5::put_index(target,name,value);
            } else {
                std::int64_t value;
                h5::check(H5Aread(a,H5T_NATIVE_INT64,&value),"reading integer "+name);
                if (value>=0) h5::put_index(target,name,static_cast<std::uint64_t>(value));
                else {
                    h5::Handle copied(H5Acreate2(target,name.c_str(),H5T_STD_I64LE,space,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
                    h5::check(H5Awrite(copied,H5T_NATIVE_INT64,&value),"copying integer "+name);
                }
            }
        }
        else h5::put_attr(target,name,h5::attr(source,name));
    }
}
} // namespace

void merge_products(const Config& c) {
    if (c.merge_inputs.empty()) throw std::runtime_error("Merge requires at least one input");
    std::vector<Tile> tiles;
    std::string signature,physics_configuration;
    Region bounds{UINT64_MAX,0,UINT64_MAX,0};
    std::uint64_t pixels=0,nf=0,depth=0,total_area=0;
    bool has_spectra=false,has_cumulative=false;
    std::set<fs::path> distinct;
    std::map<std::string,Metadata> reference_metadata;
    std::map<std::string,double> reference_numbers;
    std::map<std::string,std::string> reference_strings;
    std::array<double,4> timing{};
    std::array<bool,4> timing_complete{true,true,true,true};
    const auto canonical_target=fs::weakly_canonical(fs::absolute(c.output));
    for (const auto& name:c.merge_inputs) {
        auto path=fs::canonical(name);
        if (!distinct.insert(path).second) throw std::runtime_error("Duplicate merge input: "+name);
        if (canonical_target==path) throw std::runtime_error("Merge output cannot replace an input");
        auto f=h5::open(path);
        if (H5Aexists(f,"Complete")<=0 || h5::attr(f,"Complete")!=1) throw std::runtime_error("Incomplete merge input: "+name);
        auto current=h5::string_attr(f,"MergeSignature");
        auto n=index_attr(f,"NumPixels"),channels=index_attr(f,"NumFreq");
        if (!n || !channels) throw std::runtime_error("Merge dimensions must be nonzero");
        bool spectral=H5Lexists(f,"taus",H5P_DEFAULT)>0;
        bool cumulative=H5Lexists(f,"T_cum_bands",H5P_DEFAULT)>0;
        if (tiles.empty()) { signature=current; pixels=n; nf=channels; has_spectra=spectral; has_cumulative=cumulative; }
        if (current!=signature || n!=pixels || channels!=nf || spectral!=has_spectra || cumulative!=has_cumulative)
            throw std::runtime_error("Merge inputs differ in source, input data, physics, spectra, or products");
        Region r{index_attr(f,"X0"),index_attr(f,"X1"),index_attr(f,"Y0"),index_attr(f,"Y1")};
        if (r.x0>=r.x1 || r.y0>=r.y1 || r.x1>pixels || r.y1>pixels) throw std::runtime_error("Invalid tile bounds");
        auto tile_area=area(r);
        if (index_attr(f,"PixelCount")!=tile_area) throw std::runtime_error("PixelCount disagrees with tile bounds");
        shaped_dataset(f,"tau_band_avgs",{5,r.x1-r.x0,r.y1-r.y0});
        if (spectral) shaped_dataset(f,"taus",{r.x1-r.x0,r.y1-r.y0,nf});
        if (cumulative) {
            auto ds=h5::dataset(f,"T_cum_bands");
            auto shape=h5::shape(ds);
            if (shape.size()!=2 || shape[0]!=5 || (!tiles.empty() && shape[1]!=depth)) throw std::runtime_error("Cumulative shape mismatch");
            numeric_dataset(ds,"T_cum_bands");
            depth=shape[1];
        }
        memory_budget(c,nf,depth);
        Tile tile{path,r,file_identity(path),{}};
        auto bd=shaped_dataset(f,"T_bands",{5});
        auto band=h5::read(bd);
        for (int b=0;b<5;++b) { transmission(band[b],"T_bands"); tile.means[b]=band[b]; }
        if (cumulative) {
            auto cd=shaped_dataset(f,"T_cum_bands",{5,depth});
            auto values=h5::read(cd);
            for (int b=0;b<5;++b) {
                double previous=1.;
                for (std::uint64_t j=0;j<depth;++j) {
                    double value=values[b*depth+j]; transmission(value,"T_cum_bands");
                    if (value>previous+1e-12) throw std::runtime_error("Cumulative transmission increases downstream");
                    previous=value;
                }
                if (std::abs(previous-band[b])>1e-8*(1.+std::abs(previous)))
                    throw std::runtime_error("Final cumulative transmission disagrees with T_bands");
            }
        }
        bool legacy=index_attr(f,"Legacy")!=0;
        auto sampling=h5::string_attr(f,"Sampling");
        if (sampling!="band" && sampling!="comb") throw std::runtime_error("Invalid merge sampling");
        std::vector<std::pair<std::string,std::vector<hsize_t>>> required={
            {"Dvs",{nf}},{"Dv_edges",{legacy?0:(sampling=="band"?nf+1:nf)}},
            {"FrequencyLowerOverNu0",{nf}},{"FrequencyUpperOverNu0",{nf}},
            {"freq_band_edges",{6}},{"freq_bands",{6}},{"frequency_weights",{5,nf}}};
        if (cumulative) required.push_back({"T_redshifts",{depth}});
        else if (H5Lexists(f,"T_redshifts",H5P_DEFAULT)>0) throw std::runtime_error("T_redshifts exists without cumulative transmission");
        bool spread=H5Lexists(f,"z_spread",H5P_DEFAULT)>0;
        if (spread) {
            auto ds=h5::dataset(f,"z_spread"); auto shape=h5::shape(ds);
            if (shape.size()!=1 || !shape[0]) throw std::runtime_error("Invalid z_spread shape");
            memory_budget(c,nf,depth,shape[0]);
            required.push_back({"z_spread",shape});
        }
        if (!tiles.empty() && spread!=(reference_metadata.count("z_spread")!=0)) throw std::runtime_error("Source-spread metadata differs");
        for (const auto& item:required) {
            auto ds=shaped_dataset(f,item.first,item.second);
            auto values=h5::read(ds);
            for (double value:values) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite merge coordinate: "+item.first);
            if (item.first=="T_redshifts") for (std::size_t j=0;j<values.size();++j)
                if (values[j]<=-1 || (j && values[j]>=values[j-1])) throw std::runtime_error("T_redshifts must decrease downstream");
            if (item.first=="frequency_weights") for (int b=0;b<5;++b) {
                double sum=0;
                for (std::uint64_t j=0;j<nf;++j) {
                    double weight=values[b*nf+j];
                    if (weight<0) throw std::runtime_error("Negative frequency weight");
                    sum+=weight;
                }
                if (std::abs(sum-1.)>1e-10) throw std::runtime_error("Frequency weights are not normalized");
            }
            if (tiles.empty()) reference_metadata.emplace(item.first,Metadata{item.second,std::move(values)});
            else {
                const auto& expected=reference_metadata.at(item.first);
                if (item.second!=expected.shape || values!=expected.values)
                    throw std::runtime_error("Merge coordinate metadata differs: "+item.first);
            }
        }
        // Signatures are necessary, but coordinate and source metadata are also
        // checked directly so a damaged file cannot silently adopt tile one's axes.
        for (const char* key:{"HubbleParam","Omega0","OmegaBaryon","Redshift","SourceRedshiftRequested",
                "SourceRedshift","SourceCell","Legacy","VoigtExpansionOrder","MaxDln","DvStep","SourceDistanceComovingCm"}) {
            bool present=H5Aexists(f,key)>0;
            if (!tiles.empty() && present!=(reference_numbers.count(key)!=0)) throw std::runtime_error(std::string("Merge metadata differs: ")+key);
            if (present) {
                double value=h5::attr(f,key);
                if (!std::isfinite(value)) throw std::runtime_error(std::string("Nonfinite merge metadata: ")+key);
                if (tiles.empty()) reference_numbers[key]=value;
                else if (reference_numbers.at(key)!=value) throw std::runtime_error(std::string("Merge metadata differs: ")+key);
            }
        }
        for (const char* key:{"Sampling","Profile","FrequencyConvention","Geometry","SpectralRepresentation","InputIdentity"}) {
            auto value=h5::string_attr(f,key);
            if (tiles.empty()) reference_strings[key]=value;
            else if (reference_strings.at(key)!=value) throw std::runtime_error(std::string("Merge metadata differs: ")+key);
        }
        auto config=h5::string_attr(f,"Configuration");
        auto region_start=config.find(";region=");
        if (region_start!=std::string::npos) config.resize(region_start);
        if (tiles.empty()) physics_configuration=config;
        else if (config!=physics_configuration) throw std::runtime_error("Merge configuration metadata differs");
        for (int i=0;i<4;++i) {
            if (H5Aexists(f,timing_names[i])<=0) timing_complete[i]=false;
            else {
                double value=h5::attr(f,timing_names[i]);
                if (!std::isfinite(value) || value<0 || !std::isfinite(timing[i]+value)) throw std::runtime_error("Invalid input timing metadata");
                timing[i]+=value;
            }
        }
        for (const auto& t:tiles)
            if (std::max(t.region.x0,r.x0)<std::min(t.region.x1,r.x1) &&
                std::max(t.region.y0,r.y0)<std::min(t.region.y1,r.y1)) throw std::runtime_error("Merge tiles overlap");
        bounds.x0=std::min(bounds.x0,r.x0); bounds.x1=std::max(bounds.x1,r.x1);
        bounds.y0=std::min(bounds.y0,r.y0); bounds.y1=std::max(bounds.y1,r.y1);
        if (total_area>UINT64_MAX-tile_area) throw std::runtime_error("Merge total area overflow");
        total_area+=tile_area; tiles.push_back(std::move(tile));
    }
    if (total_area!=area(bounds)) throw std::runtime_error("Merge tiles leave gaps in their bounding rectangle");
    auto region_text=";region="+std::to_string(bounds.x0)+","+std::to_string(bounds.x1)+","+
        std::to_string(bounds.y0)+","+std::to_string(bounds.y1);
    auto merged_signature="merge:"+signature+region_text+";tile_files="+hash_identities(tiles);
    if (c.dry_run) {
        std::cout<<"Merge validated "<<tiles.size()<<" tiles covering "<<total_area<<" pixels; no output written.\n";
        return;
    }
    fs::path target=fs::absolute(c.output),lock=target.string()+".lock",temp=target.string()+".partial."+std::to_string(getpid());
    if (fs::exists(target) && c.resume) {
        if (!Product::matches(target,merged_signature)) throw std::runtime_error("Existing merged product has a different configuration or input tile files");
        std::cout<<"Resume: already complete "<<target<<'\n'; return;
    }
    fs::create_directories(target.parent_path());
    if (!fs::create_directory(lock)) throw std::runtime_error("Merge output is locked");
    try {
        if (fs::exists(target) && !c.overwrite) throw std::runtime_error("Merge output exists; select --overwrite or --resume");
        auto first=h5::open(tiles.front().path);
        h5::Handle out(H5Fcreate(temp.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT),H5Fclose);
        copy_attributes(first,out);
        h5::put_index(out,"X0",bounds.x0); h5::put_index(out,"X1",bounds.x1);
        h5::put_index(out,"Y0",bounds.y0); h5::put_index(out,"Y1",bounds.y1);
        h5::put_index(out,"PixelCount",total_area); h5::put_index(out,"Chunk",0); h5::put_index(out,"ChunkSize",bounds.x1-bounds.x0);
        h5::put_string(out,"RunSignature",merged_signature);
        h5::put_string(out,"Configuration",physics_configuration+region_text);
        h5::put_index(out,"MergedTileCount",tiles.size());
        for (int i=0;i<4;++i) if (timing_complete[i]) h5::put_attr(out,timing_names[i],timing[i]);
        for (const auto& item:reference_metadata)
            h5::check(H5Ocopy(first,item.first.c_str(),out,item.first.c_str(),H5P_DEFAULT,H5P_DEFAULT),"copying metadata "+item.first);
        auto nx=bounds.x1-bounds.x0,ny=bounds.y1-bounds.y0;
        auto maps=h5::create(out,"tau_band_avgs",{5,nx,ny},{1,std::min<std::uint64_t>(nx,128),std::min<std::uint64_t>(ny,128)});
        h5::Handle spectra;
        if (has_spectra) spectra=h5::create(out,"taus",{nx,ny,nf},{std::min<std::uint64_t>(nx,16),std::min<std::uint64_t>(ny,16),std::min<std::uint64_t>(nf,128)});
        std::vector<double> means(5,0),cum(5*depth,0);
        for (const auto& tile:tiles) {
            if (file_identity(tile.path)!=tile.identity) throw std::runtime_error("Merge input changed after validation");
            auto in=h5::open(tile.path);
            auto ds=h5::dataset(in,"tau_band_avgs");
            auto tx=tile.region.x1-tile.region.x0,ty=tile.region.y1-tile.region.y0;
            std::array<double,5> map_sums{};
            for (std::uint64_t x=0;x<tx;x+=32) for (std::uint64_t y=0;y<ty;y+=32) {
                auto dx=std::min<std::uint64_t>(32,tx-x),dy=std::min<std::uint64_t>(32,ty-y);
                auto values=h5::slice(ds,{0,x,y},{5,dx,dy});
                for (int b=0;b<5;++b) for (std::uint64_t j=0;j<dx*dy;++j) {
                    double tau=values[b*dx*dy+j];
                    if (std::isnan(tau) || tau<0) throw std::runtime_error("Invalid optical depth in merge map");
                    map_sums[b]+=std::exp(-tau);
                }
                h5::write(maps,{0,tile.region.x0-bounds.x0+x,tile.region.y0-bounds.y0+y},{5,dx,dy},values.data());
            }
            for (int b=0;b<5;++b)
                if (std::abs(map_sums[b]/area(tile.region)-tile.means[b])>1e-8*(1.+std::abs(tile.means[b])))
                    throw std::runtime_error("Merge map average disagrees with T_bands");
            if (has_spectra) {
                auto sd=h5::dataset(in,"taus");
                for (std::uint64_t x=0;x<tx;x+=16) for (std::uint64_t y=0;y<ty;y+=16) for (std::uint64_t f=0;f<nf;f+=128) {
                    auto dx=std::min<std::uint64_t>(16,tx-x),dy=std::min<std::uint64_t>(16,ty-y),df=std::min<std::uint64_t>(128,nf-f);
                    auto values=h5::slice(sd,{x,y,f},{dx,dy,df});
                    for (double tau:values) if (std::isnan(tau) || tau<0) throw std::runtime_error("Invalid optical depth in merge spectrum");
                    h5::write(spectra,{tile.region.x0-bounds.x0+x,tile.region.y0-bounds.y0+y,f},{dx,dy,df},values.data());
                }
            }
            double weight=static_cast<double>(area(tile.region))/total_area;
            for (int b=0;b<5;++b) means[b]+=weight*tile.means[b];
            if (has_cumulative) {
                auto cd=h5::dataset(in,"T_cum_bands"); auto values=h5::read(cd);
                for (std::size_t i=0;i<cum.size();++i) cum[i]+=weight*values[i];
            }
        }
        if (has_cumulative) {
            for (int b=0;b<5;++b) means[b]=depth?cum[b*depth+depth-1]:1.;
            auto ds=h5::create(out,"T_cum_bands",{5,depth}); h5::write(ds,{0,0},{5,depth},cum.data());
        }
        h5::vector(out,"T_bands",means); h5::put_attr(out,"Complete",1);
        h5::check(H5Fflush(out,H5F_SCOPE_GLOBAL),"flushing merged output");
        maps=h5::Handle(); spectra=h5::Handle(); out=h5::Handle();
        if (!c.overwrite && fs::exists(target)) throw std::runtime_error("Merge output appeared while writing");
        fs::rename(temp,target);
        fs::remove(lock);
        std::cout<<"Merged "<<tiles.size()<<" tiles into "<<target<<'\n';
    } catch (...) {
        std::error_code ec; fs::remove(temp,ec); fs::remove(lock,ec); throw;
    }
}
} // namespace lya
