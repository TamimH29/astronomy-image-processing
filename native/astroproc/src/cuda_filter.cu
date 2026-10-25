#include "astroproc/astroproc.hpp"
#include "internal.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace astroproc {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    std::size_t capacity = 0;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void reserve(std::size_t count) {
        if (count <= capacity) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::overflow_error("CUDA allocation overflow");
        T* replacement = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(T)), "cudaMalloc");
        if (data) cudaFree(data);
        data = replacement; capacity = count;
    }
};
template<class T> struct PinnedBuffer {
    T* data = nullptr;
    std::size_t capacity = 0;
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    void reserve(std::size_t count) {
        if (count <= capacity) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::overflow_error("pinned allocation overflow");
        T* replacement = nullptr;
        check(cudaMallocHost(reinterpret_cast<void**>(&replacement), count * sizeof(T)), "cudaMallocHost");
        if (data) cudaFreeHost(data);
        data = replacement; capacity = count;
    }
};
struct Slot {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, uploaded = nullptr, computed = nullptr, completed = nullptr;
    DeviceBuffer<float> input, horizontal, horizontal_weight, output, kernel;
    DeviceBuffer<std::uint8_t> mask;
    PinnedBuffer<float> host_input, host_output, host_kernel;
    PinnedBuffer<std::uint8_t> host_mask;
    bool busy = false;
    std::uint64_t job = 0;
    std::size_t width = 0, height = 0;
    ~Slot() {
        if (stream) cudaStreamSynchronize(stream);
        if (start) cudaEventDestroy(start);
        if (uploaded) cudaEventDestroy(uploaded);
        if (computed) cudaEventDestroy(computed);
        if (completed) cudaEventDestroy(completed);
        if (stream) cudaStreamDestroy(stream);
    }
    void initialize() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check(cudaEventCreate(&start), "cudaEventCreate");
        check(cudaEventCreate(&uploaded), "cudaEventCreate");
        check(cudaEventCreate(&computed), "cudaEventCreate");
        check(cudaEventCreate(&completed), "cudaEventCreate");
    }
    void reserve(std::size_t n, std::size_t kernel_size) {
        input.reserve(n); horizontal.reserve(n); horizontal_weight.reserve(n); output.reserve(n); mask.reserve(n);
        host_input.reserve(n); host_output.reserve(n); host_mask.reserve(n);
        kernel.reserve(kernel_size); host_kernel.reserve(kernel_size);
    }
};

__device__ int reflected(int i, int n) {
    if (n == 1) return 0;
    const long long period = 2LL * n;
    long long j = i % period;
    if (j < 0) j += period;
    return static_cast<int>(j < n ? j : period - 1 - j);
}
// Grid-stride loops avoid assumptions about maximum grid dimensions.
__global__ void horizontal_filter(const float* input, const std::uint8_t* mask,
                                  const float* kernel, float* values, float* weights,
                                  int width, int height, int radius) {
    const std::size_t count = static_cast<std::size_t>(width) * height;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const int x = static_cast<int>(i % width), y = static_cast<int>(i / width);
        float value = 0, weight = 0;
        for (int dx = -radius; dx <= radius; ++dx) {
            const auto j = static_cast<std::size_t>(y) * width + reflected(x + dx, width);
            if (!mask[j]) { value += kernel[dx + radius] * input[j]; weight += kernel[dx + radius]; }
        }
        values[i] = value; weights[i] = weight;
    }
}
__global__ void vertical_filter(const float* horizontal, const float* weights,
                                const std::uint8_t* mask, const float* kernel,
                                float* output, int width, int height, int radius) {
    const std::size_t count = static_cast<std::size_t>(width) * height;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const int x = static_cast<int>(i % width), y = static_cast<int>(i / width);
        float value = 0, weight = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            const auto j = static_cast<std::size_t>(reflected(y + dy, height)) * width + x;
            value += kernel[dy + radius] * horizontal[j];
            weight += kernel[dy + radius] * weights[j];
        }
        output[i] = mask[i] || weight <= 0 ? 0 : value / weight;
    }
}
std::atomic<std::uint64_t> next_job{1};
} // namespace

struct CudaFilterContext::Impl {
    int device = 0;
    std::vector<std::unique_ptr<Slot>> slots;
    ~Impl() {
        // Buffer destructors must execute while their allocating device is current.
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        slots.clear();
        cudaSetDevice(previous);
    }
    Slot& find(FilterJob job) const {
        for (const auto& slot : slots) if (slot->busy && slot->job == job.id) return *slot;
        throw std::invalid_argument("unknown, consumed, or foreign CUDA job handle");
    }
    void select_device() const { check(cudaSetDevice(device), "cudaSetDevice"); }
};

CudaAvailability cuda_availability() {
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) { cudaGetLastError(); return {false, cudaGetErrorString(status)}; }
    if (!count) return {false, "no CUDA devices available"};
    return {true, {}};
}
CudaFilterContext::CudaFilterContext(std::size_t count) {
    if (!count || count > 64) throw std::invalid_argument("CUDA slot count must be 1..64");
    const auto available = cuda_availability();
    if (!available.available) throw std::runtime_error("CUDA unavailable: " + available.reason);
    impl_ = std::make_unique<Impl>();
    check(cudaGetDevice(&impl_->device), "cudaGetDevice");
    for (std::size_t i = 0; i < count; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->initialize(); impl_->slots.push_back(std::move(slot));
    }
}
CudaFilterContext::~CudaFilterContext() = default;
std::size_t CudaFilterContext::capacity() const { return impl_->slots.size(); }

std::optional<FilterJob> CudaFilterContext::submit(ImageView input, const ProcessingConfig& config) {
    validate_image(input); validate_config(config);
    Slot* slot = nullptr;
    for (const auto& candidate : impl_->slots) if (!candidate->busy) { slot = candidate.get(); break; }
    if (!slot) return std::nullopt;
    impl_->select_device();
    const auto kernel = detail::gaussian_kernel(config);
    const auto n = input.width * input.height;
    slot->reserve(n, kernel.size());
    slot->width = input.width; slot->height = input.height;
    for (std::size_t y = 0; y < input.height; ++y) for (std::size_t x = 0; x < input.width; ++x) {
        const auto i = y * input.width + x;
        slot->host_mask.data[i] = !detail::valid(input, x, y);
        slot->host_input.data[i] = slot->host_mask.data[i] ? 0 : input.pixels[y * detail::stride(input) + x];
    }
    std::copy(kernel.begin(), kernel.end(), slot->host_kernel.data);
    try {
        check(cudaEventRecord(slot->start, slot->stream), "record transfer start");
        check(cudaMemcpyAsync(slot->input.data, slot->host_input.data, n * sizeof(float), cudaMemcpyHostToDevice, slot->stream), "upload pixels");
        check(cudaMemcpyAsync(slot->mask.data, slot->host_mask.data, n, cudaMemcpyHostToDevice, slot->stream), "upload mask");
        check(cudaMemcpyAsync(slot->kernel.data, slot->host_kernel.data, kernel.size() * sizeof(float), cudaMemcpyHostToDevice, slot->stream), "upload coefficients");
        check(cudaEventRecord(slot->uploaded, slot->stream), "record upload completion");
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((n + threads - 1) / threads, 65535));
        const int radius = filter_radius(config), w = static_cast<int>(input.width), h = static_cast<int>(input.height);
        horizontal_filter<<<blocks, threads, 0, slot->stream>>>(slot->input.data, slot->mask.data, slot->kernel.data,
                                                             slot->horizontal.data, slot->horizontal_weight.data, w, h, radius);
        check(cudaGetLastError(), "launch horizontal filter");
        vertical_filter<<<blocks, threads, 0, slot->stream>>>(slot->horizontal.data, slot->horizontal_weight.data, slot->mask.data,
                                                           slot->kernel.data, slot->output.data, w, h, radius);
        check(cudaGetLastError(), "launch vertical filter");
        check(cudaEventRecord(slot->computed, slot->stream), "record kernel completion");
        check(cudaMemcpyAsync(slot->host_output.data, slot->output.data, n * sizeof(float), cudaMemcpyDeviceToHost, slot->stream), "download filtered pixels");
        check(cudaEventRecord(slot->completed, slot->stream), "record download completion");
        slot->job = next_job.fetch_add(1);
        slot->busy = true;
        return FilterJob{slot->job};
    } catch (...) {
        cudaStreamSynchronize(slot->stream);
        throw;
    }
}
bool CudaFilterContext::ready(FilterJob job) const {
    impl_->select_device();
    auto status = cudaEventQuery(impl_->find(job).completed);
    if (status == cudaErrorNotReady) return false;
    check(status, "query filter completion");
    return true;
}
CudaFilterResult CudaFilterContext::collect(FilterJob job) {
    impl_->select_device();
    auto& slot = impl_->find(job);
    try {
        check(cudaEventSynchronize(slot.completed), "wait for filter completion");
        CudaFilterResult result;
        result.image = Image(slot.width, slot.height);
        const auto n = slot.width * slot.height;
        std::copy_n(slot.host_output.data, n, result.image.pixels.data());
        std::copy_n(slot.host_mask.data, n, result.image.invalid.data());
        for (float value : result.image.pixels)
            if (!std::isfinite(value)) throw std::overflow_error("CUDA Gaussian filtering overflowed float32");
        float upload = 0, kernels = 0, download = 0;
        check(cudaEventElapsedTime(&upload, slot.start, slot.uploaded), "time upload");
        check(cudaEventElapsedTime(&kernels, slot.uploaded, slot.computed), "time kernels");
        check(cudaEventElapsedTime(&download, slot.computed, slot.completed), "time download");
        result.h2d_ms = upload; result.kernel_ms = kernels; result.d2h_ms = download;
        slot.busy = false;
        return result;
    } catch (...) {
        cudaStreamSynchronize(slot.stream);
        slot.busy = false;
        throw;
    }
}
Image filter_cuda(ImageView image, const ProcessingConfig& config) {
    CudaFilterContext context(1);
    return context.collect(*context.submit(image, config)).image;
}
} // namespace astroproc
