#include "background.hpp"
#include "workers.hpp"
#include "image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <chrono>
#include <numeric>

namespace astro_background {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

using astro_cpu::Serial;
using astro_cpu::Workers;

struct Workspace {
    std::vector<double> samples, scratch, compacted;
    std::vector<std::size_t> counts, offsets;
};
struct Stats { double center, noise; };

double median(std::vector<double>& scratch, double* selection_ms) {
    const auto started = selection_ms ? Clock::now() : Clock::time_point{};
    const auto middle = scratch.begin() + static_cast<std::ptrdiff_t>(scratch.size() / 2);
    std::nth_element(scratch.begin(), middle, scratch.end());
    double result = *middle;
    if (scratch.size() % 2 == 0)
        result = (*std::max_element(scratch.begin(), middle) + result) / 2;
    if (selection_ms) *selection_ms += milliseconds(started);
    return result;
}

template<class Executor>
Stats clipped_stats(Workspace& data, Executor& executor, double* selection_ms = nullptr) {
    // Match background.py::_clipped_stats for both mesh levels and global noise:
    // center = median(samples), noise = 1.4826 * median(abs(samples - center));
    // retain center +/- 3 * noise for at most five passes, then summarize again.
    // These fixed constants are independent of DetectionParams.threshold_sigma.
    data.counts.resize(executor.size());
    data.offsets.resize(executor.size());
    auto summarize = [&] {
        // Samples are private to this call and only their multiset matters.
        // Select the exact center in place instead of copying the whole image
        // before every clipping pass. Caller-owned image pixels are untouched.
        const auto center = median(data.samples, selection_ms);
        data.scratch.resize(data.samples.size());
        executor.run(data.samples.size(), [&](auto, auto begin, auto end) {
            for (auto i = begin; i < end; ++i)
                data.scratch[i] = std::abs(data.samples[i] - center);
        });
        return Stats{center, 1.4826 * median(data.scratch, selection_ms)};
    };
    for (int iteration = 0; iteration < 5; ++iteration) {
        const auto stats = summarize();
        // A zero MAD also occurs in quantized or constant samples. Coverage and
        // source exclusion are applied before this statistical definition.
        if (stats.noise == 0) return stats;
        const double limit = 3 * stats.noise; // Background clipping, not source detection.
        executor.run(data.samples.size(), [&](auto worker, auto begin, auto end) {
            // Compact each disjoint worker range while counting it. Evaluate
            // the clipping predicate once, then join the retained ranges with
            // contiguous copies rather than testing every sample a second time.
            const auto first = data.samples.begin() + begin;
            const auto retained = std::remove_if(first, data.samples.begin() + end,
                [&](double value) { return std::abs(value - stats.center) > limit; });
            data.counts[worker] = static_cast<std::size_t>(retained - first);
        });
        std::size_t kept = 0;
        for (std::size_t i = 0; i < executor.size(); ++i) {
            data.offsets[i] = kept;
            kept += data.counts[i];
        }
        if (kept == data.samples.size()) return stats;
        if (executor.size() == 1) {
            // Mesh statistics and one-thread global statistics are already
            // compact: no second buffer or merge pass is needed.
            data.samples.resize(kept);
            continue;
        }
        data.compacted.resize(kept);
        executor.run(data.samples.size(), [&](auto worker, auto begin, auto) {
            std::copy_n(data.samples.begin() + begin, data.counts[worker],
                        data.compacted.begin() + data.offsets[worker]);
        });
        data.samples.swap(data.compacted);
    }
    return summarize();
}

struct Axis {
    std::vector<std::size_t> lower, upper;
    std::vector<double> weight;
};
Axis axis_weights(const std::vector<double>& centers, std::size_t length) {
    Axis axis{std::vector<std::size_t>(length), std::vector<std::size_t>(length),
              std::vector<double>(length)};
    if (centers.size() == 1) return axis;
    for (std::size_t i = 0; i < length; ++i) {
        const auto found = std::lower_bound(centers.begin(), centers.end(), static_cast<double>(i));
        const auto position = static_cast<std::size_t>(found - centers.begin());
        const auto lower = std::min(position ? position - 1 : 0, centers.size() - 2);
        axis.lower[i] = lower;
        axis.upper[i] = lower + 1;
        axis.weight[i] = (static_cast<double>(i) - centers[lower]) /
                         (centers[lower + 1] - centers[lower]);
    }
    return axis;
}

bool solve(double a[3][3], double b[3], std::size_t size) {
    double scale = 0;
    for (std::size_t i = 0; i < size; ++i) for (std::size_t j = 0; j < size; ++j)
        scale = std::max(scale, std::abs(a[i][j]));
    for (std::size_t col = 0; col < size; ++col) {
        auto pivot = col;
        for (auto row = col + 1; row < size; ++row)
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
        if (std::abs(a[pivot][col]) <= 1e-12 * scale) return false;
        for (std::size_t j = 0; j < size; ++j) std::swap(a[col][j], a[pivot][j]);
        std::swap(b[col], b[pivot]);
        const auto divisor = a[col][col];
        for (std::size_t j = 0; j < size; ++j) a[col][j] /= divisor;
        b[col] /= divisor;
        for (std::size_t row = 0; row < size; ++row) {
            if (row == col) continue;
            const auto factor = a[row][col];
            for (std::size_t j = 0; j < size; ++j) a[row][j] -= factor * a[col][j];
            b[row] -= factor * b[col];
        }
    }
    return true;
}

astro_cpu::Mask fill_mesh(std::vector<double>& mesh, const std::vector<double>& cy,
                          const std::vector<double>& cx, std::size_t box, Workers& workers) {
    astro_cpu::Mask measured(mesh.size());
    std::vector<std::size_t> good;
    for (std::size_t i = 0; i < mesh.size(); ++i) {
        measured[i] = std::isfinite(mesh[i]);
        if (measured[i]) good.push_back(i);
    }
    if (good.empty()) throw std::invalid_argument("insufficient background coverage: no usable mesh cells");
    const auto original = mesh;
    workers.run(mesh.size(), [&](auto, auto begin, auto end) {
        for (auto cell = begin; cell < end; ++cell) {
            if (measured[cell]) continue;
            std::vector<std::size_t> order(good.size()); std::iota(order.begin(), order.end(), 0);
            std::vector<double> dx(good.size()), dy(good.size()), distance(good.size());
            for (std::size_t i = 0; i < good.size(); ++i) {
                dx[i] = (cx[good[i] % cx.size()] - cx[cell % cx.size()]) / box;
                dy[i] = (cy[good[i] / cx.size()] - cy[cell / cx.size()]) / box;
                distance[i] = dx[i] * dx[i] + dy[i] * dy[i];
            }
            std::sort(order.begin(), order.end(), [&](auto a, auto b) { return distance[a] < distance[b] || (distance[a] == distance[b] && a < b); });
            auto neighbors = std::min(std::size_t{8}, good.size());
            for (;;) {
                const auto size = std::size_t{1} + (cx.size() > 1) + (cy.size() > 1);
                double matrix[3][3]{}, rhs[3]{};
                for (std::size_t j = 0; j < neighbors; ++j) {
                    const auto i = order[j]; double v[3]{1, 0, 0}; std::size_t n = 1;
                    if (cx.size() > 1) v[n++] = dx[i];
                    if (cy.size() > 1) v[n++] = dy[i];
                    const auto weight = 1.0 / std::max(distance[i], 1e-12);
                    for (std::size_t row = 0; row < size; ++row) {
                        rhs[row] += weight * v[row] * original[good[i]];
                        for (std::size_t col = 0; col < size; ++col) matrix[row][col] += weight * v[row] * v[col];
                    }
                }
                if (solve(matrix, rhs, size)) { mesh[cell] = rhs[0]; break; }
                if (neighbors == good.size()) throw std::invalid_argument("insufficient background coverage: independent spatial support required");
                neighbors = std::min(2 * neighbors, good.size());
            }
        }
    });
    return measured;
}
} // namespace

double estimate(const float* pixels, std::size_t height, std::size_t width,
                std::size_t row_stride, std::size_t box, std::size_t threads,
                float* background, Timings* timings, const bool* valid_mask,
                bool refine, Diagnostics* diagnostics) {
    const auto started = Clock::now();
    if (!height || !width || !box || !threads || !pixels || !background || row_stride < width)
        throw std::invalid_argument("nonempty image, positive box/threads, and valid strides required");
    if (height > std::numeric_limits<std::size_t>::max() / width)
        throw std::invalid_argument("image dimensions overflow");
    const auto count = height * width;
    Workers workers(std::min({threads, static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency())), count}));
    Timings measured; measured.effective_threads = workers.size();
    std::vector<float> data(count), residual(count);
    astro_cpu::Mask valid(count), seed(count, 0), source(count, 0);
    workers.run(height, [&](auto, auto begin, auto end) {
        for (auto y = begin; y < end; ++y) for (std::size_t x = 0; x < width; ++x) {
            const auto p = y * width + x;
            data[p] = pixels[y * row_stride + x];
            if (!std::isfinite(data[p])) throw std::invalid_argument("image pixels must be finite");
            valid[p] = !valid_mask || valid_mask[p];
        }
    });
    if (std::none_of(valid.begin(), valid.end(), [](auto x) { return x != 0; }))
        throw std::invalid_argument("image has no usable observations");
    Workspace global;
    auto gather = [&](bool subtract, bool exclude_sources) {
        global.counts.assign(workers.size(), 0); global.offsets.resize(workers.size());
        workers.run(count, [&](auto worker, auto begin, auto end) {
            std::size_t kept = 0;
            for (auto p = begin; p < end; ++p)
                kept += valid[p] && (!exclude_sources || !source[p]);
            // Publish once instead of repeatedly writing neighboring workers'
            // counters on the same cache line during the pixel scan.
            global.counts[worker] = kept;
        });
        std::size_t kept = 0;
        for (std::size_t i = 0; i < workers.size(); ++i) {
            global.offsets[i] = kept; kept += global.counts[i];
        }
        if (!kept) throw std::invalid_argument("insufficient background coverage: no source-free observations");
        global.samples.resize(kept);
        workers.run(count, [&](auto worker, auto begin, auto end) {
            auto output = global.offsets[worker];
            for (auto p = begin; p < end; ++p) if (valid[p] && (!exclude_sources || !source[p]))
                global.samples[output++] = subtract ? static_cast<double>(data[p]) - static_cast<double>(background[p]) : static_cast<double>(data[p]);
        });
    };
    auto phase = Clock::now();
    if (refine) {
        gather(false, false);
        const auto stats = clipped_stats(global, workers);
        double range = 0;
        for (std::size_t p = 0; p < count; ++p) if (valid[p]) range = std::max(range, std::abs(static_cast<double>(data[p]) - stats.center));
        const auto floor = 8 * std::numeric_limits<float>::epsilon() * range;
        seed = astro_cpu::source_seeds(data, valid, stats.center + std::max(3 * stats.noise, floor), height, width);
    }
    measured.source_mask_ms += milliseconds(phase);
    const auto nx = (width - 1) / box + 1, ny = (height - 1) / box + 1;
    std::vector<double> cx(nx), cy(ny), mesh(nx * ny);
    for (std::size_t x = 0; x < nx; ++x) {
        const auto start = x * box; cx[x] = (static_cast<double>(start) + start + std::min(box, width - start) - 1) / 2;
    }
    for (std::size_t y = 0; y < ny; ++y) {
        const auto start = y * box; cy[y] = (static_cast<double>(start) + start + std::min(box, height - start) - 1) / 2;
    }
    const auto ax = axis_weights(cx, width), ay = axis_weights(cy, height);
    std::vector<double> rows(ny * width);
    std::vector<Workspace> local(workers.size());
    astro_cpu::Mask measured_mesh;
    double noise = 0;
    for (int iteration = 0; iteration < (refine ? 3 : 1); ++iteration) {
        measured.refinement_passes = static_cast<std::size_t>(iteration + 1);
        phase = Clock::now();
        source = astro_cpu::dilate_sources(seed, valid, height, width, workers);
        measured.source_mask_ms += milliseconds(phase);
        phase = Clock::now();
        std::fill(mesh.begin(), mesh.end(), std::numeric_limits<double>::quiet_NaN());
        workers.run(nx * ny, [&](auto worker, auto begin, auto end) {
            Serial serial; auto& workspace = local[worker];
            for (auto cell = begin; cell < end; ++cell) {
                const auto y0 = (cell / nx) * box, x0 = (cell % nx) * box;
                const auto y1 = y0 + std::min(box, height - y0), x1 = x0 + std::min(box, width - x0);
                workspace.samples.clear(); std::size_t covered = 0;
                for (auto y = y0; y < y1; ++y) for (auto x = x0; x < x1; ++x) {
                    const auto p = y * width + x;
                    if (valid[p]) { ++covered; if (!source[p]) workspace.samples.push_back(data[p]); }
                }
                if (10 * covered >= 9 * (y1 - y0) * (x1 - x0) && !workspace.samples.empty() && 2 * workspace.samples.size() >= covered)
                    mesh[cell] = clipped_stats(workspace, serial).center;
            }
        });
        measured.mesh_ms += milliseconds(phase);
        phase = Clock::now();
        measured_mesh = fill_mesh(mesh, cy, cx, box, workers);
        workers.run(ny, [&](auto, auto begin, auto end) {
            for (auto y = begin; y < end; ++y) for (std::size_t x = 0; x < width; ++x)
                rows[y * width + x] = mesh[y * nx + ax.lower[x]] * (1 - ax.weight[x]) + mesh[y * nx + ax.upper[x]] * ax.weight[x];
        });
        workers.run(height, [&](auto, auto begin, auto end) {
            for (auto y = begin; y < end; ++y) for (std::size_t x = 0; x < width; ++x) {
                const auto p = y * width + x;
                const double value = rows[ay.lower[y] * width + x] * (1 - ay.weight[y]) + rows[ay.upper[y] * width + x] * ay.weight[y];
                const auto rounded = static_cast<float>(value);
                if (!std::isfinite(rounded)) throw FloatingPointFailure("background interpolation overflowed float32");
                background[p] = valid[p] ? rounded : 0;
            }
        });
        measured.interpolation_ms += milliseconds(phase);
        phase = Clock::now();
        gather(true, true);
        noise = clipped_stats(global, workers, timings ? &measured.global_median_ms : nullptr).noise;
        measured.noise_ms += milliseconds(phase);
        if (!refine || iteration == 2) break;
        phase = Clock::now();
        std::vector<double> ranges(workers.size(), 0);
        workers.run(count, [&](auto worker, auto begin, auto end) {
            for (auto p = begin; p < end; ++p) {
                residual[p] = valid[p] ? data[p] - background[p] : 0;
                if (!std::isfinite(residual[p])) throw FloatingPointFailure("residual overflowed float32");
                ranges[worker] = std::max(ranges[worker], std::abs(static_cast<double>(residual[p])));
            }
        });
        const auto floor = 8 * std::numeric_limits<float>::epsilon() * *std::max_element(ranges.begin(), ranges.end());
        const auto additions = astro_cpu::source_seeds(residual, valid, std::max(3 * noise, floor), height, width);
        bool changed = false;
        for (std::size_t p = 0; p < count; ++p) if (additions[p] && !seed[p]) { changed = true; seed[p] = 1; }
        measured.source_mask_ms += milliseconds(phase);
        if (!changed) break;
    }
    measured.total_ms = milliseconds(started);
    if (timings) *timings = measured;
    if (diagnostics) {
        diagnostics->mesh = std::move(mesh); diagnostics->measured_mesh = std::move(measured_mesh);
        diagnostics->source_mask = std::move(source); diagnostics->ny = ny; diagnostics->nx = nx;
    }
    return noise;
}
} // namespace astro_background
