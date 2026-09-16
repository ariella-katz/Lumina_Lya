#pragma once

#include <hdf5.h>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lya::h5 {
inline void check(herr_t value, const std::string& what) {
    if (value < 0) throw std::runtime_error("HDF5: " + what);
}
class Handle {
    hid_t id_ = -1;
    herr_t (*close_)(hid_t) = nullptr;
public:
    Handle() = default;
    Handle(hid_t id, herr_t (*closer)(hid_t)) : id_(id), close_(closer) {
        if (id < 0) throw std::runtime_error("Cannot open or create HDF5 object");
    }
    ~Handle() { if (id_ >= 0) close_(id_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept : id_(other.id_), close_(other.close_) { other.id_ = -1; }
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            if (id_ >= 0) close_(id_);
            id_ = other.id_; close_ = other.close_; other.id_ = -1;
        }
        return *this;
    }
    operator hid_t() const { return id_; }
};
inline Handle open(const std::filesystem::path& path) {
    auto id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (id < 0) throw std::runtime_error("Cannot open HDF5 file: " + path.string());
    return Handle(id, H5Fclose);
}
inline Handle dataset(hid_t file, const std::string& name) {
    auto id = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
    if (id < 0) throw std::runtime_error("Missing or unreadable dataset: " + name);
    return Handle(id, H5Dclose);
}
inline std::vector<hsize_t> shape(hid_t ds) {
    Handle space(H5Dget_space(ds), H5Sclose);
    int rank = H5Sget_simple_extent_ndims(space);
    if (rank < 0) throw std::runtime_error("Cannot read dataset rank");
    std::vector<hsize_t> dims(rank);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    return dims;
}
inline double attr(hid_t object, const std::string& name) {
    Handle a(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose);
    Handle space(H5Aget_space(a), H5Sclose);
    Handle type(H5Aget_type(a), H5Tclose);
    auto kind = H5Tget_class(type);
    if (H5Sget_simple_extent_npoints(space) != 1 ||
        (kind != H5T_FLOAT && kind != H5T_INTEGER))
        throw std::runtime_error("Expected one numeric value in attribute " + name);
    double value;
    check(H5Aread(a, H5T_NATIVE_DOUBLE, &value), "reading attribute " + name);
    return value;
}
inline double attr_or(hid_t object, const std::string& name, double fallback) {
    return H5Aexists(object, name.c_str()) > 0 ? attr(object, name) : fallback;
}
inline void put_attr(hid_t object, const std::string& name, double value) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle a(H5Acreate2(object, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check(H5Awrite(a, H5T_NATIVE_DOUBLE, &value), "writing attribute " + name);
}
inline void put_index(hid_t object, const std::string& name, std::uint64_t value) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle a(H5Acreate2(object, name.c_str(), H5T_STD_U64LE, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check(H5Awrite(a, H5T_NATIVE_UINT64, &value), "writing integer attribute " + name);
}
inline void put_string(hid_t object, const std::string& name, const std::string& value) {
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check(H5Tset_size(type, value.size() + 1), "string size");
    check(H5Tset_strpad(type, H5T_STR_NULLTERM), "string padding");
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle a(H5Acreate2(object, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check(H5Awrite(a, type, value.c_str()), "writing string " + name);
}
inline std::string string_attr(hid_t object, const std::string& name) {
    Handle a(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose);
    Handle type(H5Aget_type(a), H5Tclose);
    Handle space(H5Aget_space(a), H5Sclose);
    if (H5Tget_class(type) != H5T_STRING || H5Sget_simple_extent_npoints(space) != 1)
        throw std::runtime_error("Expected one string value in attribute " + name);
    if (H5Tis_variable_str(type)) {
        char* value = nullptr;
        check(H5Aread(a, type, &value), "reading string " + name);
        std::string result = value ? value : "";
        H5free_memory(value);
        return result;
    }
    std::vector<char> data(H5Tget_size(type) + 1, '\0');
    check(H5Aread(a, type, data.data()), "reading string " + name);
    return data.data();
}
inline std::uint64_t count(const std::vector<hsize_t>& dims) {
    std::uint64_t n = 1;
    for (auto d : dims) {
        if (d && n > UINT64_MAX / d) throw std::runtime_error("Dataset size overflow");
        n *= d;
    }
    return n;
}
inline std::vector<double> read(hid_t ds) {
    std::vector<double> result(count(shape(ds)));
    if (!result.empty()) check(H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()), "reading dataset");
    return result;
}
inline std::vector<double> slice(hid_t ds, const std::vector<hsize_t>& start,
                                 const std::vector<hsize_t>& dims) {
    std::vector<double> result(count(dims));
    if (result.empty()) return result;
    Handle file_space(H5Dget_space(ds), H5Sclose);
    check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr, dims.data(), nullptr), "selecting read slab");
    Handle mem_space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
    check(H5Dread(ds, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, result.data()), "reading slab");
    return result;
}
inline Handle create(hid_t file, const std::string& name, const std::vector<hsize_t>& dims,
                     const std::vector<hsize_t>& chunks = {}) {
    Handle space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
    Handle plist(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    if (!chunks.empty() && count(dims))
        check(H5Pset_chunk(plist, static_cast<int>(chunks.size()), chunks.data()), "setting chunks");
    return Handle(H5Dcreate2(file, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, plist, H5P_DEFAULT), H5Dclose);
}
inline void write(hid_t ds, const std::vector<hsize_t>& start, const std::vector<hsize_t>& dims,
                  const double* data) {
    if (!count(dims)) return;
    Handle fs(H5Dget_space(ds), H5Sclose);
    check(H5Sselect_hyperslab(fs, H5S_SELECT_SET, start.data(), nullptr, dims.data(), nullptr), "selecting write slab");
    Handle ms(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
    check(H5Dwrite(ds, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT, data), "writing slab");
}
inline void vector(hid_t file, const std::string& name, const std::vector<double>& values) {
    auto ds = create(file, name, {values.size()});
    if (!values.empty()) check(H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()), "writing " + name);
}
} // namespace lya::h5
