#pragma once
#include "workers.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace astro_cpu {
using Mask = std::vector<std::uint8_t>;
struct Component {
    std::vector<std::size_t> pixels;
    std::size_t x0, y0, x1, y1;
};
std::vector<Component> components(const Mask& selected, std::size_t height, std::size_t width);
Mask source_seeds(const std::vector<float>& values, const Mask& valid, double cutoff,
                  std::size_t height, std::size_t width);
Mask dilate_sources(const Mask& seeds, const Mask& valid, std::size_t height,
                    std::size_t width, Workers& workers);
struct Measurement {
    double x, y;
    std::size_t x0, y0, x1, y1, area;
    double flux;
};
struct DetectionTimings {
    double label_ms = 0, deblend_ms = 0, measurement_ms = 0, total_ms = 0;
    std::size_t effective_threads = 1;
};
struct DeblendProfile {
    std::size_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, pixels = 0;
    std::size_t markers = 0, children = 0, passes = 0;
    std::size_t queue_pushes = 0, queue_pops = 0, queue_peak = 0;
    double prepare_ms = 0, marker_ms = 0, watershed_ms = 0, pruning_ms = 0, total_ms = 0;
};
struct DetectionResult {
    std::vector<Measurement> rows;
    std::vector<std::int32_t> labels;
    DetectionTimings timings;
    std::vector<DeblendProfile> deblend_components;
};
DetectionResult detect(const float* residual, const float* filtered, const bool* valid,
                       std::size_t height, std::size_t width, double threshold,
                       std::size_t min_area, bool deblend, std::size_t separation,
                       double prominence, std::size_t threads,
                       const float* peak_image = nullptr, std::size_t centroid_radius = 2,
                       bool profile = false);
} // namespace astro_cpu
