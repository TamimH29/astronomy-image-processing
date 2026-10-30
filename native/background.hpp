#pragma once

#include <cstddef>
#include <stdexcept>

namespace astro_background {

struct FloatingPointFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Timings {
    double mesh_ms = 0;
    double interpolation_ms = 0;
    double noise_ms = 0;
    // Subset of noise_ms: only exact global median selection, not vector work.
    double global_median_ms = 0;
    double total_ms = 0;
    std::size_t effective_threads = 1;
};

double estimate(const float* pixels, std::size_t height, std::size_t width,
                std::size_t row_stride, std::size_t box, std::size_t threads,
                float* background, Timings* timings = nullptr);

} // namespace astro_background
