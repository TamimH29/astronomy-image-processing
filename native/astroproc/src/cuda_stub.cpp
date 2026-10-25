#include "astroproc/astroproc.hpp"
#include <stdexcept>

namespace astroproc {
struct CudaFilterContext::Impl {};
CudaAvailability cuda_availability() { return {false, "CUDA backend was not compiled; enable ASTROPROC_ENABLE_CUDA"}; }
CudaFilterContext::CudaFilterContext(std::size_t) { throw std::runtime_error(cuda_availability().reason); }
CudaFilterContext::~CudaFilterContext() = default;
std::optional<FilterJob> CudaFilterContext::submit(ImageView, const ProcessingConfig&) {
    throw std::runtime_error(cuda_availability().reason);
}
bool CudaFilterContext::ready(FilterJob) const { throw std::runtime_error(cuda_availability().reason); }
CudaFilterResult CudaFilterContext::collect(FilterJob) { throw std::runtime_error(cuda_availability().reason); }
std::size_t CudaFilterContext::capacity() const { return 0; }
Image filter_cuda(ImageView, const ProcessingConfig&) { throw std::runtime_error(cuda_availability().reason); }
} // namespace astroproc
