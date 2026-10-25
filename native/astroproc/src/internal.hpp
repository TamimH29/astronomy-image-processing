#pragma once
#include "astroproc/astroproc.hpp"
#include <cmath>
#include <vector>

namespace astroproc::detail {
inline std::size_t stride(ImageView v) { return v.stride ? v.stride : v.width; }
inline std::size_t mask_stride(ImageView v) { return v.mask_stride ? v.mask_stride : v.width; }
inline bool valid(ImageView v, std::size_t x, std::size_t y) {
    return (!v.invalid || !v.invalid[y * mask_stride(v) + x]) &&
           std::isfinite(v.pixels[y * stride(v) + x]);
}
inline int reflect(int i, int n) {
    if (n == 1) return 0;
    const long long period = 2LL * n;
    long long j = i % period;
    if (j < 0) j += period;
    return static_cast<int>(j < n ? j : period - 1 - j);
}
std::vector<float> gaussian_kernel(const ProcessingConfig& config);
} // namespace astroproc::detail
