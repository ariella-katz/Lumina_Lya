#include "lya/backend.hpp"
#include "lya/transport.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace lya {
namespace {
constexpr unsigned threads = 128;

void check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

std::size_t multiply(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("CUDA allocation size overflow");
    return static_cast<std::size_t>(a * b);
}

template<class T> void device_allocate(T*& pointer, std::size_t count) {
    if (!count) return;
    const auto bytes = multiply(count, sizeof(T));
    const auto code = cudaMalloc(reinterpret_cast<void**>(&pointer), bytes);
    if (code != cudaSuccess)
        throw std::runtime_error("CUDA allocation of " + std::to_string(bytes) + " bytes failed: "
            + cudaGetErrorString(code) + "; reduce --tile-size or --depth-slab");
}
template<class T> void pinned_allocate(T*& pointer, std::size_t count) {
    if (!count) return;
    const auto bytes = multiply(count, sizeof(T));
    const auto code = cudaMallocHost(reinterpret_cast<void**>(&pointer), bytes);
    if (code != cudaSuccess)
        throw std::runtime_error("Pinned host allocation of " + std::to_string(bytes) + " bytes failed: "
            + cudaGetErrorString(code) + "; reduce --tile-size or --depth-slab");
}

unsigned blocks_for(std::uint64_t size) {
    return static_cast<unsigned>(std::min<std::uint64_t>((size + threads - 1) / threads, 65535));
}

// Raw slots pack density, ionization, temperature and the three velocity
// components. The layout is unchanged from each HDF5 hyperslab read.
__global__ void preprocess(const double* raw, GasCell* gas, std::uint64_t depth,
    std::uint64_t rays, Region region, Header header, BackendSettings settings, int* invalid) {
    const std::uint64_t cells = rays * depth;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        i < cells; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const double rho = raw[i], ionized = raw[cells + i], temperature = raw[2 * cells + i];
        const double vx = raw[3 * cells + 3 * i];
        const double vy = raw[3 * cells + 3 * i + 1];
        const double vz = raw[3 * cells + 3 * i + 2];
        if (!(rho >= 0.) || !isfinite(rho) || !(ionized >= 0. && ionized <= 1.) ||
            !(temperature > 0.) || !isfinite(temperature) ||
            !isfinite(vx) || !isfinite(vy) || !isfinite(vz)) {
            atomicOr(invalid, 1);
            gas[i] = {0., 1., 0.};
        } else {
            gas[i] = prepare_gas(rho, ionized, temperature, vx, vy, vz,
                i / depth, region, header);
            if (!isfinite(gas[i].neutral_a1) || !isfinite(gas[i].inverse_thermal_speed) ||
                !isfinite(gas[i].outward_velocity_a1)) atomicOr(invalid, 1);
            if (!voigt_expansion_valid(gas[i], settings)) {
                atomicOr(invalid, 4);
                gas[i].neutral_a1 = 0.;
            }
        }
    }
}

__global__ void accumulate(double* __restrict__ tau, const GasCell* __restrict__ gas,
    const Segment* __restrict__ segments, const std::uint64_t* __restrict__ offsets,
    const Frequency* __restrict__ frequencies, const double* __restrict__ source_redshifts,
    std::uint64_t source_count, std::uint64_t rays, std::uint64_t frequency_count,
    std::uint64_t depth, BackendSettings settings, int* invalid) {
    constexpr unsigned stage_size = 32;
    __shared__ Segment staged_segments[stage_size];
    __shared__ GasCell staged_gas[stage_size];
    const std::uint64_t frequency_tiles = (frequency_count + threads - 1) / threads;
    const std::uint64_t count = source_count * rays * frequency_tiles;
    // Each block owns one ray/source/frequency tile. Cooperative staging loads
    // each depth segment and gas cell once for all 128 frequency lanes.
    for (std::uint64_t work = blockIdx.x; work < count; work += gridDim.x) {
        const std::uint64_t frequency = (work % frequency_tiles) * threads + threadIdx.x;
        const std::uint64_t ray_source = work / frequency_tiles;
        const std::uint64_t ray = ray_source % rays, source = ray_source / rays;
        const auto index = ray_source * frequency_count + frequency;
        const bool active = frequency < frequency_count;
        double value = active ? tau[index] : 0.;
        const Frequency local_frequency = active ? frequencies[frequency] : Frequency{};
        const double source_redshift = source_redshifts[source];
        for (auto part = offsets[source]; part < offsets[source + 1]; part += stage_size) {
            const auto remaining = offsets[source + 1] - part;
            const unsigned staged = remaining < stage_size ? static_cast<unsigned>(remaining) : stage_size;
            if (threadIdx.x < staged) {
                const auto segment = segments[part + threadIdx.x];
                staged_segments[threadIdx.x] = segment;
                staged_gas[threadIdx.x] = gas[ray * depth + segment.cell];
            }
            __syncthreads();
            if (active)
                for (unsigned i = 0; i < staged; ++i) {
                    const double increment = segment_tau(staged_gas[i], staged_segments[i],
                        source_redshift, local_frequency, settings);
                    if (!(increment >= 0.) || !isfinite(increment)) atomicOr(invalid, 2);
                    value += increment;
                }
            __syncthreads();
        }
        if (active) {
            if (!isfinite(value)) atomicOr(invalid, 2);
            tau[index] = value;
        }
    }
}

// Cumulative products use a separate, slower path: a block owns one ray/source
// and reduces every frequency before a single spatial atomic add per band.
// The spatial summation order can vary between launches; all sums use FP64.
__global__ void accumulate_cumulative(double* tau, double* cumulative, const GasCell* gas,
    const Segment* segments, const std::uint64_t* offsets,
    const Frequency* frequencies, const double* source_redshifts,
    std::uint64_t source_count, std::uint64_t rays, std::uint64_t frequency_count,
    std::uint64_t depth, std::uint64_t z0, std::uint64_t total_depth,
    BackendSettings settings, int* invalid) {
    __shared__ double reduction[5 * threads];
    __shared__ Segment staged_segment;
    __shared__ GasCell staged_gas;
    for (std::uint64_t ray_source = blockIdx.x; ray_source < rays * source_count;
        ray_source += gridDim.x) {
        const auto ray = ray_source % rays, source = ray_source / rays;
        const auto base = ray_source * frequency_count;
        for (auto part = offsets[source]; part < offsets[source + 1]; ++part) {
            if (threadIdx.x == 0) {
                staged_segment = segments[part];
                staged_gas = gas[ray * depth + staged_segment.cell];
            }
            __syncthreads();
            const Segment segment = staged_segment;
            for (std::uint64_t f = threadIdx.x; f < frequency_count; f += blockDim.x) {
                const double increment = segment_tau(staged_gas, segment,
                    source_redshifts[source], frequencies[f], settings);
                if (!(increment >= 0.) || !isfinite(increment)) atomicOr(invalid, 2);
                tau[base + f] += increment;
                if (!isfinite(tau[base + f])) atomicOr(invalid, 2);
            }
            if (segment.report) {
                double sums[5] = {};
                for (std::uint64_t f = threadIdx.x; f < frequency_count; f += blockDim.x) {
                    const double transmission = exp(-tau[base + f]);
                    for (unsigned band = 0; band < 5; ++band)
                        sums[band] += transmission * frequencies[f].weights[band];
                }
                for (unsigned band = 0; band < 5; ++band)
                    reduction[band * threads + threadIdx.x] = sums[band];
                __syncthreads();
                for (unsigned stride = threads / 2; stride; stride /= 2) {
                    if (threadIdx.x < stride)
                        for (unsigned band = 0; band < 5; ++band)
                            reduction[band * threads + threadIdx.x]
                                += reduction[band * threads + threadIdx.x + stride];
                    __syncthreads();
                }
                if (threadIdx.x < 5)
                    atomicAdd(cumulative + (source * 5 + threadIdx.x) * total_depth
                        + z0 + segment.cell, reduction[threadIdx.x * threads]);
            }
            __syncthreads();
        }
    }
}

struct Slot {
    double* host_raw = nullptr;
    double* device_raw = nullptr;
    GasCell* device_gas = nullptr;
    Segment* host_segments = nullptr;
    Segment* device_segments = nullptr;
    std::uint64_t* host_offsets = nullptr;
    std::uint64_t* device_offsets = nullptr;
    std::size_t segment_capacity = 0;
    cudaStream_t transfers = nullptr;
    cudaEvent_t transfer_start = nullptr, transfer_end = nullptr;
    cudaEvent_t kernel_start = nullptr, kernel_end = nullptr;
    bool pending = false;

    ~Slot() {
        if (transfers) cudaStreamSynchronize(transfers);
        if (kernel_end && pending) cudaEventSynchronize(kernel_end);
        cudaFree(device_raw); cudaFree(device_gas);
        cudaFree(device_segments); cudaFree(device_offsets);
        cudaFreeHost(host_raw); cudaFreeHost(host_segments); cudaFreeHost(host_offsets);
        if (transfer_start) cudaEventDestroy(transfer_start);
        if (transfer_end) cudaEventDestroy(transfer_end);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (kernel_end) cudaEventDestroy(kernel_end);
        if (transfers) cudaStreamDestroy(transfers);
    }
};

class CudaBackend final : public Backend {
    Header header_;
    Spectrum spectrum_;
    std::vector<Source> sources_;
    BackendSettings settings_;
    Region region_;
    std::uint64_t max_depth_ = 0;
    std::size_t tau_count_ = 0, cumulative_count_ = 0, next_slot_ = 0;
    bool initialized_ = false;
    std::array<std::unique_ptr<Slot>, 2> slots_;
    cudaStream_t compute_ = nullptr;
    double* tau_ = nullptr;
    double* cumulative_ = nullptr;
    Frequency* frequencies_ = nullptr;
    double* source_redshifts_ = nullptr;
    int* invalid_ = nullptr;
    BackendTiming timing_;

    void collect(Slot& slot) {
        if (!slot.pending) return;
        check(cudaEventSynchronize(slot.kernel_end), "wait for completed slab");
        float milliseconds = 0.f;
        check(cudaEventElapsedTime(&milliseconds, slot.transfer_start, slot.transfer_end), "time slab transfer");
        timing_.transfer_seconds += milliseconds * .001;
        check(cudaEventElapsedTime(&milliseconds, slot.kernel_start, slot.kernel_end), "time slab kernels");
        timing_.kernel_seconds += milliseconds * .001;
        slot.pending = false;
    }

    void release_tile() {
        // The stream fence also protects cleanup after a failed submit that did
        // not reach its completion-event record.
        if (compute_) cudaStreamSynchronize(compute_);
        for (auto& slot : slots_) slot.reset();
        cudaFree(tau_); tau_ = nullptr;
        cudaFree(cumulative_); cumulative_ = nullptr;
        initialized_ = false;
    }

    std::vector<double> download(const double* data, std::size_t count) {
        finish();
        std::vector<double> result(count);
        if (!count) return result;
        cudaEvent_t start = nullptr, end = nullptr;
        check(cudaEventCreate(&start), "create output timing event");
        try {
            check(cudaEventCreate(&end), "create output timing event");
            check(cudaEventRecord(start, compute_), "record output transfer start");
            check(cudaMemcpyAsync(result.data(), data, multiply(count, sizeof(double)),
                cudaMemcpyDeviceToHost, compute_), "copy output to host");
            check(cudaEventRecord(end, compute_), "record output transfer end");
            check(cudaEventSynchronize(end), "wait for output transfer");
            float milliseconds = 0.f;
            check(cudaEventElapsedTime(&milliseconds, start, end), "time output transfer");
            timing_.transfer_seconds += milliseconds * .001;
        } catch (...) {
            cudaEventDestroy(start);
            if (end) cudaEventDestroy(end);
            throw;
        }
        cudaEventDestroy(start); cudaEventDestroy(end);
        return result;
    }

public:
    CudaBackend(const Header& h, const Spectrum& f, const std::vector<Source>& s,
        const BackendSettings& o) : header_(h), spectrum_(f), sources_(s), settings_(o) {
        if (f.frequencies.empty() || s.empty())
            throw std::invalid_argument("Backend requires at least one source and frequency");
        check(cudaSetDevice(settings_.device), "select CUDA device (run on an allocated GPU node)");
        try {
            check(cudaStreamCreateWithFlags(&compute_, cudaStreamNonBlocking), "create compute stream");
            device_allocate(frequencies_, spectrum_.frequencies.size());
            device_allocate(source_redshifts_, sources_.size());
            device_allocate(invalid_, 1);
            check(cudaMemcpy(frequencies_, spectrum_.frequencies.data(),
                multiply(spectrum_.frequencies.size(), sizeof(Frequency)), cudaMemcpyHostToDevice), "copy frequency grid");
            std::vector<double> redshifts;
            for (const auto& source : sources_) redshifts.push_back(source.actual);
            check(cudaMemcpy(source_redshifts_, redshifts.data(),
                multiply(redshifts.size(), sizeof(double)), cudaMemcpyHostToDevice), "copy source redshifts");
        } catch (...) {
            cudaFree(frequencies_); cudaFree(source_redshifts_); cudaFree(invalid_);
            if (compute_) cudaStreamDestroy(compute_);
            throw;
        }
    }

    ~CudaBackend() override {
        release_tile();
        cudaFree(frequencies_); cudaFree(source_redshifts_); cudaFree(invalid_);
        if (compute_) cudaStreamDestroy(compute_);
    }

    void begin(const Region& region, std::uint64_t max_depth) override {
        if (initialized_) finish();
        release_tile();
        if (region.x1 <= region.x0 || region.y1 <= region.y0 ||
            region.x1 > header_.pixels || region.y1 > header_.pixels || !max_depth)
            throw std::invalid_argument("Invalid backend tile or slab depth");
        region_ = region; max_depth_ = max_depth; next_slot_ = 0;
        tau_count_ = multiply(multiply(region.rays(), sources_.size()), spectrum_.frequencies.size());
        cumulative_count_ = settings_.cumulative ? multiply(multiply(sources_.size(), 5), header_.depth) : 0;
        device_allocate(tau_, tau_count_);
        device_allocate(cumulative_, cumulative_count_);
        check(cudaMemsetAsync(tau_, 0, multiply(tau_count_, sizeof(double)), compute_), "clear optical depths");
        if (cumulative_count_)
            check(cudaMemsetAsync(cumulative_, 0, multiply(cumulative_count_, sizeof(double)), compute_), "clear cumulative sums");
        check(cudaMemsetAsync(invalid_, 0, sizeof(int), compute_), "clear validation status");
        const auto cells = multiply(region.rays(), max_depth);
        for (auto& pointer : slots_) {
            pointer = std::make_unique<Slot>();
            auto& slot = *pointer;
            check(cudaStreamCreateWithFlags(&slot.transfers, cudaStreamNonBlocking), "create transfer stream");
            check(cudaEventCreate(&slot.transfer_start), "create transfer event");
            check(cudaEventCreate(&slot.transfer_end), "create transfer event");
            check(cudaEventCreate(&slot.kernel_start), "create kernel event");
            check(cudaEventCreate(&slot.kernel_end), "create kernel event");
            pinned_allocate(slot.host_raw, multiply(cells, 6));
            device_allocate(slot.device_raw, multiply(cells, 6));
            device_allocate(slot.device_gas, cells);
            pinned_allocate(slot.host_offsets, sources_.size() + 1);
            device_allocate(slot.device_offsets, sources_.size() + 1);
        }
        initialized_ = true;
    }

    void submit(const RawSlab& slab, const Batch& batch) override {
        if (!initialized_ || slab.region.x0 != region_.x0 || slab.region.x1 != region_.x1 ||
            slab.region.y0 != region_.y0 || slab.region.y1 != region_.y1 ||
            !slab.depth || slab.depth > max_depth_ || slab.z0 > header_.depth ||
            slab.depth > header_.depth - slab.z0)
            throw std::invalid_argument("Raw slab does not match active CUDA tile");
        const auto cells = multiply(region_.rays(), slab.depth);
        if (slab.density.size() != cells || slab.ionized.size() != cells ||
            slab.temperature.size() != cells || slab.velocity.size() != multiply(cells, 3))
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
        auto& slot = *slots_[next_slot_];
        next_slot_ = (next_slot_ + 1) % slots_.size();
        collect(slot);
        // The caller may release RawSlab and Batch as soon as submit returns.
        const auto bytes = multiply(cells, sizeof(double));
        std::memcpy(slot.host_raw, slab.density.data(), bytes);
        std::memcpy(slot.host_raw + cells, slab.ionized.data(), bytes);
        std::memcpy(slot.host_raw + 2 * cells, slab.temperature.data(), bytes);
        std::memcpy(slot.host_raw + 3 * cells, slab.velocity.data(), multiply(bytes, 3));
        if (batch.segments.size() > slot.segment_capacity) {
            cudaFree(slot.device_segments); slot.device_segments = nullptr;
            cudaFreeHost(slot.host_segments); slot.host_segments = nullptr;
            slot.segment_capacity = batch.segments.size();
            pinned_allocate(slot.host_segments, slot.segment_capacity);
            device_allocate(slot.device_segments, slot.segment_capacity);
        }
        if (!batch.segments.empty())
            std::memcpy(slot.host_segments, batch.segments.data(), multiply(batch.segments.size(), sizeof(Segment)));
        std::memcpy(slot.host_offsets, batch.offsets.data(), multiply(batch.offsets.size(), sizeof(std::uint64_t)));
        check(cudaEventRecord(slot.transfer_start, slot.transfers), "record transfer start");
        check(cudaMemcpyAsync(slot.device_raw, slot.host_raw, multiply(bytes, 6),
            cudaMemcpyHostToDevice, slot.transfers), "transfer gas slab");
        if (!batch.segments.empty())
            check(cudaMemcpyAsync(slot.device_segments, slot.host_segments,
                multiply(batch.segments.size(), sizeof(Segment)), cudaMemcpyHostToDevice, slot.transfers), "transfer segments");
        check(cudaMemcpyAsync(slot.device_offsets, slot.host_offsets,
            multiply(batch.offsets.size(), sizeof(std::uint64_t)), cudaMemcpyHostToDevice, slot.transfers), "transfer segment offsets");
        check(cudaEventRecord(slot.transfer_end, slot.transfers), "record transfer end");
        check(cudaStreamWaitEvent(compute_, slot.transfer_end, 0), "wait for gas transfer");
        check(cudaEventRecord(slot.kernel_start, compute_), "record kernel start");
        preprocess<<<blocks_for(cells), threads, 0, compute_>>>(slot.device_raw, slot.device_gas,
            slab.depth, region_.rays(), region_, header_, settings_, invalid_);
        check(cudaGetLastError(), "launch gas preprocessing");
        if (!batch.segments.empty()) {
            if (settings_.cumulative) {
                const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(region_.rays() * sources_.size(), 65535));
                accumulate_cumulative<<<blocks, threads, 0, compute_>>>(tau_, cumulative_, slot.device_gas,
                    slot.device_segments, slot.device_offsets, frequencies_, source_redshifts_,
                    sources_.size(), region_.rays(), spectrum_.frequencies.size(), slab.depth,
                    slab.z0, header_.depth, settings_, invalid_);
            } else {
                const auto frequency_tiles = (spectrum_.frequencies.size() + threads - 1) / threads;
                const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(
                    multiply(multiply(region_.rays(), sources_.size()), frequency_tiles), 65535));
                accumulate<<<blocks, threads, 0, compute_>>>(tau_, slot.device_gas,
                    slot.device_segments, slot.device_offsets, frequencies_, source_redshifts_,
                    sources_.size(), region_.rays(), spectrum_.frequencies.size(), slab.depth, settings_, invalid_);
            }
            check(cudaGetLastError(), "launch transmission kernel");
        }
        check(cudaEventRecord(slot.kernel_end, compute_), "record kernel end");
        slot.pending = true;
    }

    void finish() override {
        if (!initialized_) return;
        for (auto& slot : slots_) collect(*slot);
        check(cudaStreamSynchronize(compute_), "finish transmission kernels");
        int invalid = 0;
        check(cudaMemcpy(&invalid, invalid_, sizeof(int), cudaMemcpyDeviceToHost), "read validation status");
        if (invalid & 4)
            throw std::runtime_error("Voigt expansion requires damping a <= 0.1; lower-temperature input needs a full-profile solver");
        if (invalid)
            throw std::runtime_error((invalid & 1) ? "Nonphysical or nonfinite input gas in CUDA slab"
                : "Invalid optical depth from CUDA absorption operator");
    }

    std::vector<double> optical_depths() override { return download(tau_, tau_count_); }
    std::vector<double> cumulative_sums() override { return download(cumulative_, cumulative_count_); }
    BackendTiming timing() const override { return timing_; }
};
} // namespace

std::unique_ptr<Backend> make_cuda_backend(const Header& h, const Spectrum& f,
    const std::vector<Source>& s, const BackendSettings& o) {
    return std::make_unique<CudaBackend>(h, f, s, o);
}

std::uint64_t cuda_free_memory(int device) {
    check(cudaSetDevice(device), "select CUDA device (run on an allocated GPU node)");
    std::size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    return free_bytes;
}
} // namespace lya
