#include "astroproc/astroproc.hpp"
#include "internal.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace astroproc {
namespace {
std::size_t checked_size(std::size_t w, std::size_t h) {
    if (!w || !h || w > static_cast<std::size_t>(INT_MAX / 2) ||
        h > static_cast<std::size_t>(INT_MAX / 2) || w > std::numeric_limits<std::size_t>::max() / h)
        throw std::invalid_argument("image dimensions must be positive, representable, and non-overflowing");
    return w * h;
}
float median(std::vector<float> values) {
    const auto n = values.size();
    if (!n) throw std::invalid_argument("no valid samples for background estimation");
    auto mid = values.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(values.begin(), mid, values.end());
    const float high = *mid;
    return n % 2 ? high : static_cast<float>((static_cast<double>(high) +
                  *std::max_element(values.begin(), mid)) * 0.5);
}
struct Stats { float level, noise; };
Stats summarize(const std::vector<float>& samples) {
    float center = median(samples);
    std::vector<float> deviation;
    deviation.reserve(samples.size());
    for (float v : samples) deviation.push_back(std::abs(v - center));
    const float noise = 1.4826022185f * median(std::move(deviation));
    if (!std::isfinite(noise)) throw std::overflow_error("background noise estimate overflowed float32");
    return {center, noise};
}
Stats clipped_stats(std::vector<float> samples, const ProcessingConfig& config) {
    Stats s = summarize(samples);
    for (int i = 0; i < config.clip_iterations; ++i) {
        std::vector<float> kept;
        kept.reserve(samples.size());
        const double limit = static_cast<double>(config.clip_sigma) * s.noise;
        for (float v : samples) if (std::abs(static_cast<double>(v) - s.level) <= limit) kept.push_back(v);
        if (kept.empty() || kept.size() == samples.size()) break;
        samples = std::move(kept);
        s = summarize(samples);
    }
    return s;
}
struct AxisPosition { std::size_t a, b; float t; };
AxisPosition interpolation(const std::vector<double>& centers, double coordinate) {
    auto it = std::upper_bound(centers.begin(), centers.end(), coordinate);
    if (it == centers.begin()) return {0, 0, 0};
    if (it == centers.end()) return {centers.size() - 1, centers.size() - 1, 0};
    std::size_t b = static_cast<std::size_t>(it - centers.begin()), a = b - 1;
    return {a, b, static_cast<float>((coordinate - centers[a]) / (centers[b] - centers[a]))};
}
void same_shape(ImageView a, ImageView b) {
    validate_image(b);
    if (a.width != b.width || a.height != b.height) throw std::invalid_argument("image shapes differ");
}
std::int64_t coordinate(std::int64_t offset, std::size_t local) {
    if (offset > std::numeric_limits<std::int64_t>::max() - static_cast<std::int64_t>(local))
        throw std::invalid_argument("coordinate offset overflows int64");
    return offset + static_cast<std::int64_t>(local);
}
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace

Image::Image(std::size_t w, std::size_t h)
    : width(w), height(h), pixels(checked_size(w, h)), invalid(pixels.size(), 0) {}
ImageView Image::view() const {
    if (pixels.size() != checked_size(width, height) || (!invalid.empty() && invalid.size() != pixels.size()))
        throw std::invalid_argument("owned image buffers do not match dimensions");
    return {pixels.data(), width, height, width, invalid.empty() ? nullptr : invalid.data(), width};
}
void validate_image(ImageView image) {
    checked_size(image.width, image.height);
    if (!image.pixels) throw std::invalid_argument("null pixel pointer");
    auto s = detail::stride(image), ms = detail::mask_stride(image);
    if (s < image.width || (image.invalid && ms < image.width))
        throw std::invalid_argument("row stride is smaller than image width");
    const auto max = std::numeric_limits<std::size_t>::max();
    if (image.height > 1 && (s > (max - image.width) / (image.height - 1) ||
        (image.invalid && ms > (max - image.width) / (image.height - 1))))
        throw std::invalid_argument("row stride address overflows");
}
void validate_config(const ProcessingConfig& c) {
    if (!std::isfinite(c.gaussian_sigma) || c.gaussian_sigma <= 0 || c.gaussian_sigma > 1024 ||
        !std::isfinite(c.detection_sigma) || c.detection_sigma <= 0 ||
        !c.min_source_area || !c.background_mesh || c.background_mesh > static_cast<std::size_t>(INT_MAX / 2) ||
        !c.min_mesh_samples || c.clip_iterations < 0 || c.clip_iterations > 100 ||
        !std::isfinite(c.clip_sigma) || c.clip_sigma <= 0)
        throw std::invalid_argument("invalid processing configuration (sigma must be finite and positive; Gaussian sigma <= 1024)");
}
int filter_radius(const ProcessingConfig& c) {
    validate_config(c);
    return static_cast<int>(std::ceil(3.0 * c.gaussian_sigma));
}
namespace detail {
std::vector<float> gaussian_kernel(const ProcessingConfig& c) {
    const int r = filter_radius(c);
    std::vector<float> k(static_cast<std::size_t>(2 * r + 1));
    double sum = 0;
    for (int i = -r; i <= r; ++i) {
        const double v = std::exp(-0.5 * (static_cast<double>(i) / c.gaussian_sigma) *
                                        (static_cast<double>(i) / c.gaussian_sigma));
        k[static_cast<std::size_t>(i + r)] = static_cast<float>(v);
        sum += v;
    }
    for (auto& v : k) v = static_cast<float>(v / sum);
    return k;
}
} // namespace detail

BackgroundModel estimate_background(ImageView image, const ProcessingConfig& c) {
    validate_image(image);
    validate_config(c);
    std::vector<float> samples;
    samples.reserve(image.width * image.height);
    for (std::size_t y = 0; y < image.height; ++y)
        for (std::size_t x = 0; x < image.width; ++x)
            if (detail::valid(image, x, y)) samples.push_back(image.pixels[y * detail::stride(image) + x]);
    if (samples.empty()) throw std::invalid_argument("image has no usable pixels");
    const Stats global = clipped_stats(std::move(samples), c);
    const std::size_t nx = (image.width + c.background_mesh - 1) / c.background_mesh;
    const std::size_t ny = (image.height + c.background_mesh - 1) / c.background_mesh;
    std::vector<Stats> mesh(nx * ny);
    std::vector<double> cx(nx), cy(ny);
    BackgroundModel result{Image(image.width, image.height), Image(image.width, image.height), {}};
    bool fallback = false;
    for (std::size_t iy = 0; iy < ny; ++iy) {
        const auto y0 = iy * c.background_mesh, y1 = std::min(y0 + c.background_mesh, image.height);
        cy[iy] = (static_cast<double>(y0) + y1 - 1) * 0.5;
        for (std::size_t ix = 0; ix < nx; ++ix) {
            const auto x0 = ix * c.background_mesh, x1 = std::min(x0 + c.background_mesh, image.width);
            cx[ix] = (static_cast<double>(x0) + x1 - 1) * 0.5;
            std::vector<float> local;
            local.reserve((x1 - x0) * (y1 - y0));
            for (auto y = y0; y < y1; ++y) for (auto x = x0; x < x1; ++x)
                if (detail::valid(image, x, y)) local.push_back(image.pixels[y * detail::stride(image) + x]);
            if (local.size() < c.min_mesh_samples) { mesh[iy * nx + ix] = global; fallback = true; }
            else mesh[iy * nx + ix] = clipped_stats(std::move(local), c);
        }
    }
    if (fallback) result.warnings.push_back("insufficient mesh samples: used image-wide background/noise");
    bool any_noise = false;
    for (std::size_t y = 0; y < image.height; ++y) {
        const auto py = interpolation(cy, static_cast<double>(y));
        for (std::size_t x = 0; x < image.width; ++x) {
            const auto px = interpolation(cx, static_cast<double>(x));
            const auto a = mesh[py.a * nx + px.a], b = mesh[py.a * nx + px.b];
            const auto d = mesh[py.b * nx + px.a], e = mesh[py.b * nx + px.b];
            auto lerp = [&](float va, float vb, float vd, float ve) {
                return (va * (1 - px.t) + vb * px.t) * (1 - py.t) +
                       (vd * (1 - px.t) + ve * px.t) * py.t;
            };
            const auto i = y * image.width + x;
            result.level.pixels[i] = lerp(a.level, b.level, d.level, e.level);
            result.noise.pixels[i] = lerp(a.noise, b.noise, d.noise, e.noise);
            any_noise |= result.noise.pixels[i] > 0;
        }
    }
    if (!any_noise) result.warnings.push_back("zero estimated noise: constant/noiseless data; use a roundoff floor for detection");
    return result;
}

PreparedImage prepare_image(ImageView image, const ProcessingConfig& c) {
    auto model = estimate_background(image, c);
    PreparedImage result{std::move(model.level), Image(image.width, image.height), std::move(model.warnings)};
    for (std::size_t y = 0; y < image.height; ++y) for (std::size_t x = 0; x < image.width; ++x) {
        const auto i = y * image.width + x;
        result.residual.invalid[i] = !detail::valid(image, x, y);
        if (!result.residual.invalid[i]) {
            result.residual.pixels[i] = image.pixels[y * detail::stride(image) + x] - result.background.pixels[i];
            if (!std::isfinite(result.residual.pixels[i])) throw std::overflow_error("background subtraction overflowed float32");
        }
    }
    return result;
}

Image filter_cpu(ImageView input, const ProcessingConfig& c) {
    validate_image(input);
    const auto kernel = detail::gaussian_kernel(c);
    const int r = filter_radius(c), w = static_cast<int>(input.width), h = static_cast<int>(input.height);
    Image output(input.width, input.height);
    std::vector<float> numerator(output.pixels.size()), weights(output.pixels.size());
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        float value = 0, weight = 0;
        for (int dx = -r; dx <= r; ++dx) {
            const auto sx = static_cast<std::size_t>(detail::reflect(x + dx, w));
            if (detail::valid(input, sx, y)) {
                const auto k = kernel[static_cast<std::size_t>(dx + r)];
                value += k * input.pixels[y * detail::stride(input) + sx];
                weight += k;
            }
        }
        const auto i = static_cast<std::size_t>(y) * input.width + x;
        numerator[i] = value; weights[i] = weight;
    }
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        float value = 0, weight = 0;
        for (int dy = -r; dy <= r; ++dy) {
            const auto sy = static_cast<std::size_t>(detail::reflect(y + dy, h));
            const auto i = sy * input.width + x;
            const auto k = kernel[static_cast<std::size_t>(dy + r)];
            value += k * numerator[i]; weight += k * weights[i];
        }
        const auto i = static_cast<std::size_t>(y) * input.width + x;
        output.invalid[i] = !detail::valid(input, x, y) || weight <= 0;
        output.pixels[i] = output.invalid[i] ? 0 : value / weight;
        if (!std::isfinite(output.pixels[i])) throw std::overflow_error("Gaussian filtering overflowed float32");
    }
    return output;
}

DetectionResult detect_sources(ImageView original, ImageView background, ImageView filtered,
                               const ImageMetadata& meta, const ProcessingConfig& c) {
    validate_image(original);
    same_shape(original, background); same_shape(original, filtered); validate_config(c);
    coordinate(meta.offset_x, original.width); coordinate(meta.offset_y, original.height);
    Image signal(original.width, original.height);
    for (std::size_t y = 0; y < original.height; ++y) for (std::size_t x = 0; x < original.width; ++x) {
        const auto i = y * original.width + x;
        signal.invalid[i] = !detail::valid(original, x, y) || !detail::valid(background, x, y) || !detail::valid(filtered, x, y);
        signal.pixels[i] = signal.invalid[i] ? 0 : filtered.pixels[y * detail::stride(filtered) + x];
    }
    auto noise_model = estimate_background(signal.view(), c);
    DetectionResult result;
    result.warnings = std::move(noise_model.warnings);
    const auto n = signal.pixels.size();
    result.labels.assign(n, 0);
    std::vector<std::uint8_t> candidate(n), visited(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Scale the floor in the input's own units; a fixed counts floor would
        // suppress otherwise identical sources expressed in very small units.
        const double scale = std::max({std::abs(static_cast<double>(noise_model.level.pixels[i])),
                                       std::abs(static_cast<double>(signal.pixels[i])),
                                       static_cast<double>(std::numeric_limits<float>::min())});
        const double floor = std::numeric_limits<float>::epsilon() * scale;
        candidate[i] = !signal.invalid[i] && static_cast<double>(signal.pixels[i]) >
            noise_model.level.pixels[i] + c.detection_sigma * std::max(floor, static_cast<double>(noise_model.noise.pixels[i]));
    }
    std::vector<std::size_t> component;
    for (std::size_t seed = 0; seed < n; ++seed) {
        if (!candidate[seed] || visited[seed]) continue;
        component.clear(); component.push_back(seed); visited[seed] = 1;
        for (std::size_t q = 0; q < component.size(); ++q) {
            const auto i = component[q];
            const int x = static_cast<int>(i % original.width), y = static_cast<int>(i / original.width);
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= static_cast<int>(original.width) || yy >= static_cast<int>(original.height)) continue;
                const auto neighbor = static_cast<std::size_t>(yy) * original.width + xx;
                if (candidate[neighbor] && !visited[neighbor]) { visited[neighbor] = 1; component.push_back(neighbor); }
            }
        }
        if (component.size() < c.min_source_area) continue;
        Source s;
        s.id = result.sources.size() + 1; s.area = component.size();
        std::size_t xmin = original.width, ymin = original.height, xmax = 0, ymax = 0;
        double sx = 0, sy = 0, weight = 0, ux = 0, uy = 0;
        for (auto i : component) {
            const auto x = i % original.width, y = i / original.width;
            xmin = std::min(xmin, x); ymin = std::min(ymin, y);
            xmax = std::max(xmax, x + 1); ymax = std::max(ymax, y + 1);
            const double v = static_cast<double>(original.pixels[y * detail::stride(original) + x]) -
                             background.pixels[y * detail::stride(background) + x];
            s.integrated_brightness += v;
            const double positive = std::max(v, 0.0);
            sx += x * positive; sy += y * positive; weight += positive; ux += x; uy += y;
            result.labels[i] = s.id;
            if (x == 0 || y == 0 || x + 1 == original.width || y + 1 == original.height) s.flags |= TouchesRegionEdge;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const auto xx = static_cast<long long>(x) + dx, yy = static_cast<long long>(y) + dy;
                if (xx >= 0 && yy >= 0 && xx < static_cast<long long>(original.width) && yy < static_cast<long long>(original.height) &&
                    signal.invalid[static_cast<std::size_t>(yy) * original.width + xx]) s.flags |= AdjacentInvalidPixel;
            }
        }
        s.centroid_x = (weight > 0 ? sx / weight : ux / s.area) + meta.offset_x;
        s.centroid_y = (weight > 0 ? sy / weight : uy / s.area) + meta.offset_y;
        s.x_min = coordinate(meta.offset_x, xmin); s.x_max = coordinate(meta.offset_x, xmax);
        s.y_min = coordinate(meta.offset_y, ymin); s.y_max = coordinate(meta.offset_y, ymax);
        result.sources.push_back(s);
    }
    return result;
}

ProcessingResult process_image(ImageView image, const ImageMetadata& meta,
                               const ProcessingConfig& config, Backend backend) {
    auto start = Clock::now();
    auto prepared = prepare_image(image, config);
    ProcessingResult result;
    result.metadata = meta; result.config = config;
    result.timings.prepare_ms = elapsed(start);
    start = Clock::now();
    result.filtered = backend == Backend::Cpu ? filter_cpu(prepared.residual.view(), config) :
                                               filter_cuda(prepared.residual.view(), config);
    result.timings.filter_ms = elapsed(start);
    start = Clock::now();
    result.detection = detect_sources(image, prepared.background.view(), result.filtered.view(), meta, config);
    result.timings.detect_ms = elapsed(start);
    result.detection.warnings.insert(result.detection.warnings.begin(), prepared.warnings.begin(), prepared.warnings.end());
    return result;
}
} // namespace astroproc
