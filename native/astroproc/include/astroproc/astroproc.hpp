#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace astroproc {

// Strides are in elements, not bytes. A zero stride means tightly packed.
// Nonzero mask bytes and nonfinite input values are invalid.
struct ImageView {
    const float* pixels = nullptr;
    std::size_t width = 0, height = 0, stride = 0;
    const std::uint8_t* invalid = nullptr;
    std::size_t mask_stride = 0;
};

struct Image {
    std::size_t width = 0, height = 0;
    std::vector<float> pixels;
    std::vector<std::uint8_t> invalid;
    Image() = default;
    Image(std::size_t width, std::size_t height);
    ImageView view() const;
};

struct ImageMetadata {
    std::string image_id;
    std::string intensity_unit = "unknown";
    std::int64_t offset_x = 0, offset_y = 0;
};

struct ProcessingConfig {
    float gaussian_sigma = 1.0f;
    float detection_sigma = 5.0f;
    std::size_t min_source_area = 5;
    std::size_t background_mesh = 64;
    std::size_t min_mesh_samples = 32;
    int clip_iterations = 3;
    float clip_sigma = 3.0f;
};

struct BackgroundModel {
    Image level;
    Image noise;
    std::vector<std::string> warnings;
};
struct PreparedImage {
    Image background;
    Image residual;
    std::vector<std::string> warnings;
};

enum SourceFlag : std::uint32_t {
    TouchesRegionEdge = 1u,
    AdjacentInvalidPixel = 2u
};
struct Source {
    std::uint64_t id = 0;
    double centroid_x = 0, centroid_y = 0;
    std::int64_t x_min = 0, y_min = 0, x_max = 0, y_max = 0;
    std::size_t area = 0;
    double integrated_brightness = 0;
    std::uint32_t flags = 0;
};
struct DetectionResult {
    std::vector<Source> sources;
    // Labels correspond to retained sources; 0 means background/rejected.
    std::vector<std::uint64_t> labels;
    std::vector<std::string> warnings;
};

enum class Backend { Cpu, Cuda };
struct StageTimings {
    double prepare_ms = 0, filter_ms = 0, detect_ms = 0;
};
struct ProcessingResult {
    Image filtered;
    DetectionResult detection;
    ImageMetadata metadata;
    ProcessingConfig config;
    StageTimings timings;
};

void validate_image(ImageView image);
void validate_config(const ProcessingConfig& config);
int filter_radius(const ProcessingConfig& config);
BackgroundModel estimate_background(ImageView image, const ProcessingConfig& config = {});
PreparedImage prepare_image(ImageView image, const ProcessingConfig& config = {});
Image filter_cpu(ImageView residual, const ProcessingConfig& config = {});
DetectionResult detect_sources(ImageView original, ImageView background,
                               ImageView filtered_residual, const ImageMetadata& metadata = {},
                               const ProcessingConfig& config = {});
ProcessingResult process_image(ImageView image, const ImageMetadata& metadata = {},
                               const ProcessingConfig& config = {}, Backend backend = Backend::Cpu);

// Public API contains no CUDA headers. One dispatcher thread owns a context.
// All memory is bounded by slot_count; submit returns nullopt when full.
struct CudaAvailability { bool available = false; std::string reason; };
CudaAvailability cuda_availability();
struct FilterJob { std::uint64_t id = 0; };
struct CudaFilterResult {
    Image image;
    double h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
};
class CudaFilterContext {
public:
    explicit CudaFilterContext(std::size_t slot_count = 2);
    ~CudaFilterContext();
    CudaFilterContext(const CudaFilterContext&) = delete;
    CudaFilterContext& operator=(const CudaFilterContext&) = delete;
    CudaFilterContext(CudaFilterContext&&) = delete;
    CudaFilterContext& operator=(CudaFilterContext&&) = delete;
    // Input is copied before return and can immediately be released/reused.
    std::optional<FilterJob> submit(ImageView residual, const ProcessingConfig& config = {});
    bool ready(FilterJob job) const;
    // Waits only for this job; handles are consumed and context-specific.
    CudaFilterResult collect(FilterJob job);
    std::size_t capacity() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
Image filter_cuda(ImageView residual, const ProcessingConfig& config = {});

} // namespace astroproc
