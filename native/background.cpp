#include "background.hpp"

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

namespace astro_background {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class Serial {
public:
    std::size_t size() const { return 1; }
    template<class F> void run(std::size_t n, F&& task) {
        task(std::size_t{0}, std::size_t{0}, n);
    }
};

// The calling thread is worker 0. Workers and their scratch storage are reused
// through every stage of this image; no process-wide mutable pool is shared.
class Workers {
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_, done_;
    std::function<void(std::size_t, std::size_t, std::size_t)> task_;
    std::size_t size_, count_ = 0, generation_ = 0, remaining_ = 0;
    bool stopping_ = false;
    std::exception_ptr failure_;

    void invoke(std::size_t worker) {
        try {
            const auto base = count_ / size_, extra = count_ % size_;
            const auto begin = worker * base + std::min(worker, extra);
            task_(worker, begin, begin + base + (worker < extra));
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
        }
    }
    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        work_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
public:
    explicit Workers(std::size_t size) : size_(size) {
        try {
            for (std::size_t i = 1; i < size_; ++i) {
                workers_.emplace_back([this, i] {
                    std::size_t seen = 0;
                    std::unique_lock lock(mutex_);
                    for (;;) {
                        work_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                        if (stopping_) return;
                        seen = generation_;
                        lock.unlock();
                        invoke(i);
                        lock.lock();
                        if (--remaining_ == 0) done_.notify_one();
                    }
                });
            }
        } catch (...) {
            stop();
            throw;
        }
    }
    ~Workers() { stop(); }
    std::size_t size() const { return size_; }
    template<class F> void run(std::size_t count, F&& task) {
        if (size_ == 1) {
            task(std::size_t{0}, std::size_t{0}, count);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            task_ = std::forward<F>(task);
            count_ = count;
            failure_ = nullptr;
            remaining_ = size_ - 1;
            ++generation_;
        }
        work_.notify_all();
        invoke(0);
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
        task_ = {};
        if (failure_) std::rethrow_exception(failure_);
    }
};

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
    data.counts.resize(executor.size());
    data.offsets.resize(executor.size());
    auto summarize = [&] {
        data.scratch.resize(data.samples.size());
        executor.run(data.samples.size(), [&](auto, auto begin, auto end) {
            std::copy(data.samples.begin() + begin, data.samples.begin() + end,
                      data.scratch.begin() + begin);
        });
        const auto center = median(data.scratch, selection_ms);
        executor.run(data.samples.size(), [&](auto, auto begin, auto end) {
            for (auto i = begin; i < end; ++i)
                data.scratch[i] = std::abs(data.samples[i] - center);
        });
        return Stats{center, 1.4826 * median(data.scratch, selection_ms)};
    };
    for (int iteration = 0; iteration < 5; ++iteration) {
        const auto stats = summarize();
        if (stats.noise == 0) return stats;
        const double limit = 3 * stats.noise;
        executor.run(data.samples.size(), [&](auto worker, auto begin, auto end) {
            std::size_t kept = 0;
            for (auto i = begin; i < end; ++i)
                kept += std::abs(data.samples[i] - stats.center) <= limit;
            data.counts[worker] = kept;
        });
        std::size_t kept = 0;
        for (std::size_t i = 0; i < executor.size(); ++i) {
            data.offsets[i] = kept;
            kept += data.counts[i];
        }
        if (kept == data.samples.size()) return stats;
        data.compacted.resize(kept);
        executor.run(data.samples.size(), [&](auto worker, auto begin, auto end) {
            auto output = data.offsets[worker];
            for (auto i = begin; i < end; ++i)
                if (std::abs(data.samples[i] - stats.center) <= limit)
                    data.compacted[output++] = data.samples[i];
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
} // namespace

double estimate(const float* pixels, std::size_t height, std::size_t width,
                std::size_t row_stride, std::size_t box, std::size_t threads,
                float* background, Timings* timings) {
    const auto started = Clock::now();
    if (!height || !width || !box || !threads || !pixels || !background || row_stride < width)
        throw std::invalid_argument("nonempty image, positive box/threads, and valid strides required");
    if (height > std::numeric_limits<std::size_t>::max() / width)
        throw std::invalid_argument("image dimensions overflow");
    const auto count = height * width;
    const auto hardware = std::max(1u, std::thread::hardware_concurrency());
    Workers workers(std::min({threads, static_cast<std::size_t>(hardware), count}));
    Timings measured;
    measured.effective_threads = workers.size();
    workers.run(height, [&](auto, auto begin, auto end) {
        for (auto y = begin; y < end; ++y)
            for (std::size_t x = 0; x < width; ++x)
                if (!std::isfinite(pixels[y * row_stride + x]))
                    throw std::invalid_argument("image pixels must be finite");
    });

    const auto mesh_started = Clock::now();
    const auto nx = (width - 1) / box + 1, ny = (height - 1) / box + 1;
    std::vector<double> centers_x(nx), centers_y(ny), mesh(nx * ny);
    for (std::size_t x = 0; x < nx; ++x) {
        const auto start = x * box;
        centers_x[x] = (static_cast<double>(start) + start + std::min(box, width - start) - 1) / 2;
    }
    for (std::size_t y = 0; y < ny; ++y) {
        const auto start = y * box;
        centers_y[y] = (static_cast<double>(start) + start + std::min(box, height - start) - 1) / 2;
    }
    std::vector<Workspace> local(workers.size());
    workers.run(nx * ny, [&](auto worker, auto begin, auto end) {
        Serial serial;
        auto& workspace = local[worker];
        for (auto cell = begin; cell < end; ++cell) {
            const auto y0 = (cell / nx) * box, x0 = (cell % nx) * box;
            const auto y1 = y0 + std::min(box, height - y0);
            const auto x1 = x0 + std::min(box, width - x0);
            workspace.samples.clear();
            for (auto y = y0; y < y1; ++y)
                for (auto x = x0; x < x1; ++x)
                    workspace.samples.push_back(pixels[y * row_stride + x]);
            mesh[cell] = clipped_stats(workspace, serial).center;
        }
    });
    measured.mesh_ms = milliseconds(mesh_started);

    const auto interpolation_started = Clock::now();
    const auto ax = axis_weights(centers_x, width), ay = axis_weights(centers_y, height);
    std::vector<double> rows(ny * width);
    workers.run(ny, [&](auto, auto begin, auto end) {
        for (auto y = begin; y < end; ++y)
            for (std::size_t x = 0; x < width; ++x)
                rows[y * width + x] = mesh[y * nx + ax.lower[x]] * (1 - ax.weight[x]) +
                                      mesh[y * nx + ax.upper[x]] * ax.weight[x];
    });
    workers.run(height, [&](auto, auto begin, auto end) {
        for (auto y = begin; y < end; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const double value = rows[ay.lower[y] * width + x] * (1 - ay.weight[y]) +
                                     rows[ay.upper[y] * width + x] * ay.weight[y];
                const auto rounded = static_cast<float>(value);
                if (!std::isfinite(rounded))
                    throw FloatingPointFailure("background interpolation overflowed float32");
                background[y * width + x] = rounded;
            }
        }
    });
    measured.interpolation_ms = milliseconds(interpolation_started);

    const auto noise_started = Clock::now();
    Workspace global;
    global.samples.resize(count);
    workers.run(height, [&](auto, auto begin, auto end) {
        for (auto y = begin; y < end; ++y)
            for (std::size_t x = 0; x < width; ++x)
                global.samples[y * width + x] = static_cast<double>(pixels[y * row_stride + x]) -
                                               static_cast<double>(background[y * width + x]);
    });
    const auto noise = clipped_stats(global, workers, timings ? &measured.global_median_ms : nullptr).noise;
    measured.noise_ms = milliseconds(noise_started);
    measured.total_ms = milliseconds(started);
    if (timings) *timings = measured;
    return noise;
}
} // namespace astro_background
