#pragma once

#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <vector>

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
    double source_mask_ms = 0;
    std::size_t refinement_passes = 0;
    double total_ms = 0;
    std::size_t effective_threads = 1;
};

struct Diagnostics {
    std::vector<std::uint8_t> source_mask, measured_mesh;
    std::vector<double> mesh;
    std::size_t ny = 0, nx = 0;
};

double estimate(const float* pixels, std::size_t height, std::size_t width,
                std::size_t row_stride, std::size_t box, std::size_t threads,
                float* background, Timings* timings = nullptr,
                const bool* valid_mask = nullptr, bool refine = true,
                Diagnostics* diagnostics = nullptr);

} // namespace astro_background
