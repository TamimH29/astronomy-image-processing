#include "image_ops.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace astro_cpu {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
template<class F> void neighbors(std::size_t p, std::size_t h, std::size_t w, F&& visit) {
    const auto y = static_cast<std::ptrdiff_t>(p / w), x = static_cast<std::ptrdiff_t>(p % w);
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const auto yy = y + dy, xx = x + dx;
        if ((dy || dx) && yy >= 0 && xx >= 0 && yy < static_cast<std::ptrdiff_t>(h) &&
                xx < static_cast<std::ptrdiff_t>(w))
            visit(static_cast<std::size_t>(yy) * w + static_cast<std::size_t>(xx));
    }
}

std::vector<std::size_t> markers(const std::vector<float>& values, const Mask& region,
                                 std::size_t h, std::size_t w, double prominence,
                                 std::size_t separation) {
    std::vector<std::size_t> order, candidates;
    for (std::size_t p = 0; p < region.size(); ++p) if (region[p]) order.push_back(p);
    auto brighter = [&](auto a, auto b) { return values[a] > values[b] || (values[a] == values[b] && a < b); };
    std::sort(order.begin(), order.end(), brighter);
    std::vector<std::ptrdiff_t> parent(region.size(), -1);
    auto root = [&](std::size_t p) {
        while (parent[p] != static_cast<std::ptrdiff_t>(p)) {
            parent[p] = parent[static_cast<std::size_t>(parent[p])];
            p = static_cast<std::size_t>(parent[p]);
        }
        return p;
    };
    // Superlevel merge tree, identical to _segmentation._markers. A losing
    // root's birth intensity minus this union level is peak-to-saddle height.
    for (auto p : order) {
        parent[p] = static_cast<std::ptrdiff_t>(p);
        neighbors(p, h, w, [&](auto q) {
            if (parent[q] < 0) return;
            auto a = root(p), b = root(q);
            if (a == b) return;
            if (brighter(b, a)) std::swap(a, b);
            if (static_cast<double>(values[b]) - values[p] >= prominence) candidates.push_back(b);
            parent[b] = static_cast<std::ptrdiff_t>(a);
        });
    }
    candidates.push_back(root(order.front()));
    std::sort(candidates.begin(), candidates.end(), brighter);
    std::vector<std::size_t> kept;
    for (auto p : candidates) {
        bool separate = true;
        for (auto q : kept) {
            const double dy = static_cast<double>(p / w) - static_cast<double>(q / w);
            const double dx = static_cast<double>(p % w) - static_cast<double>(q % w);
            if (dy * dy + dx * dx < static_cast<double>(separation) * separation) { separate = false; break; }
        }
        if (separate) kept.push_back(p);
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}
struct Event { double level; std::size_t sequence, pixel; std::int32_t owner; bool seed; };
struct Later {
    bool operator()(const Event& a, const Event& b) const {
        return std::tie(a.level, a.sequence) > std::tie(b.level, b.sequence);
    }
};
struct Workspace {
    Mask pending;
    std::vector<Event> heap;
    std::vector<std::size_t> areas, retained;
};
void watershed(const std::vector<float>& values, const Mask& region,
               std::size_t h, std::size_t w, const std::vector<std::size_t>& seeds,
               std::vector<std::int32_t>& owners, Workspace& work, DeblendProfile* profile) {
    owners.assign(region.size(), 0);
    work.pending.assign(region.size(), 0);
    work.heap.clear();
    std::size_t sequence = 0;
    auto push = [&](Event event) {
        work.pending[event.pixel] = 1;
        work.heap.push_back(event);
        std::push_heap(work.heap.begin(), work.heap.end(), Later{});
        if (profile) {
            ++profile->queue_pushes;
            profile->queue_peak = std::max(profile->queue_peak, work.heap.size());
        }
    };
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        owners[seeds[i]] = static_cast<std::int32_t>(i + 1);
        push(Event{-static_cast<double>(values[seeds[i]]), sequence++, seeds[i], owners[seeds[i]], true});
    }
    while (!work.heap.empty()) {
        const auto event = work.heap.front();
        std::pop_heap(work.heap.begin(), work.heap.end(), Later{});
        work.heap.pop_back();
        if (profile) ++profile->queue_pops;
        if (!event.seed) {
            owners[event.pixel] = event.owner;
        }
        neighbors(event.pixel, h, w, [&](auto q) {
            // Path levels never decrease. The first queued event wins both
            // priority and FIFO ties; later events cannot improve it. Ownership
            // still changes only on pop, and marker pixels remain protected.
            if (region[q] && !work.pending[q])
                push(Event{std::max(event.level, -static_cast<double>(values[q])),
                           sequence++, q, event.owner, false});
        });
    }
}
struct Group {
    std::size_t h = 0, w = 0, children = 0;
    std::vector<std::int32_t> labels;
    std::vector<Measurement> rows;
    std::vector<std::size_t> peaks;
};
} // namespace

std::vector<Component> components(const Mask& selected, std::size_t h, std::size_t w) {
    Mask visited(selected.size(), 0);
    std::vector<Component> result;
    for (std::size_t p = 0; p < selected.size(); ++p) {
        if (!selected[p] || visited[p]) continue;
        Component component{{p}, p % w, p / w, p % w, p / w};
        visited[p] = 1;
        for (std::size_t i = 0; i < component.pixels.size(); ++i) {
            const auto pixel = component.pixels[i];
            component.x0 = std::min(component.x0, pixel % w); component.x1 = std::max(component.x1, pixel % w);
            component.y0 = std::min(component.y0, pixel / w); component.y1 = std::max(component.y1, pixel / w);
            neighbors(pixel, h, w, [&](auto q) {
                if (selected[q] && !visited[q]) { visited[q] = 1; component.pixels.push_back(q); }
            });
        }
        result.push_back(std::move(component));
    }
    return result;
}

Mask source_seeds(const std::vector<float>& values, const Mask& valid, double cutoff,
                  std::size_t h, std::size_t w) {
    Mask selected(values.size());
    for (std::size_t p = 0; p < values.size(); ++p) selected[p] = valid[p] && static_cast<double>(values[p]) > cutoff;
    Mask result(values.size(), 0);
    for (const auto& component : components(selected, h, w))
        if (component.pixels.size() >= 5) for (auto p : component.pixels) result[p] = 1;
    return result;
}

Mask dilate_sources(const Mask& seeds, const Mask& valid, std::size_t h,
                    std::size_t w, Workers& workers) {
    if (std::none_of(seeds.begin(), seeds.end(), [](auto value) { return value != 0; })) return seeds;
    Mask result(seeds.size(), 0);
    workers.run(h, [&](auto, auto begin, auto end) {
        for (auto y = begin; y < end; ++y) {
            // A disk is the union of 17 horizontal spans. Sliding counts give
            // exact binary dilation in O(17 * pixels), without per-pixel disks.
            for (int dy = -8; dy <= 8; ++dy) {
                const auto yy = static_cast<std::ptrdiff_t>(y) + dy;
                if (yy < 0 || yy >= static_cast<std::ptrdiff_t>(h)) continue;
                const auto radius = static_cast<std::size_t>(std::sqrt(64 - dy * dy));
                const auto row = static_cast<std::size_t>(yy) * w;
                std::size_t found = 0;
                for (std::size_t x = 0; x < std::min(w, radius + 1); ++x) found += seeds[row + x] != 0;
                for (std::size_t x = 0; x < w; ++x) {
                    if (valid[y * w + x] && found) result[y * w + x] = 1;
                    if (x >= radius) found -= seeds[row + x - radius] != 0;
                    if (x + radius + 1 < w) found += seeds[row + x + radius + 1] != 0;
                }
            }
        }
    });
    return result;
}

DetectionResult detect(const float* residual, const float* filtered, const bool* valid,
                       std::size_t h, std::size_t w, double threshold,
                       std::size_t min_area, bool deblend, std::size_t separation,
                       double prominence, std::size_t threads,
                       const float* peak_image, std::size_t centroid_radius, bool profile) {
    const auto started = Clock::now();
    if (!h || !w || !threads || !min_area || !separation || !centroid_radius || !std::isfinite(threshold) || threshold < 0 ||
            !std::isfinite(prominence) || prominence <= 0 || h > std::numeric_limits<std::size_t>::max() / w)
        throw std::invalid_argument("valid dimensions, thresholds, and positive detection settings required");
    const auto count = h * w;
    Workers workers(std::min({threads, static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency())), count}));
    DetectionResult result;
    result.timings.effective_threads = workers.size();
    Mask selected(count);
    for (std::size_t p = 0; p < count; ++p) {
        if (!std::isfinite(residual[p]) || !std::isfinite(filtered[p]) || (peak_image && !std::isfinite(peak_image[p]))) throw std::invalid_argument("image pixels must be finite");
        selected[p] = (!valid || valid[p]) && static_cast<double>(filtered[p]) > threshold;
    }
    auto parents = components(selected, h, w);
    parents.erase(std::remove_if(parents.begin(), parents.end(), [&](const auto& c) { return c.pixels.size() < min_area; }), parents.end());
    result.timings.label_ms = ms(started);
    auto phase = Clock::now();
    std::vector<Group> groups(parents.size());
    std::vector<Workspace> workspace(workers.size());
    if (profile) result.deblend_components.resize(parents.size());
    workers.run(parents.size(), [&](auto worker, auto begin, auto end) {
        auto& work = workspace[worker];
        for (auto i = begin; i < end; ++i) {
            auto& parent = parents[i]; auto& group = groups[i];
            auto* detail = profile ? &result.deblend_components[i] : nullptr;
            const auto started_group = detail ? Clock::now() : Clock::time_point{};
            if (detail) {
                detail->x0 = parent.x0; detail->y0 = parent.y0;
                detail->x1 = parent.x1; detail->y1 = parent.y1;
                detail->pixels = parent.pixels.size();
            }
            std::sort(parent.pixels.begin(), parent.pixels.end());
            group.h = parent.y1 - parent.y0 + 1; group.w = parent.x1 - parent.x0 + 1;
            Mask region(group.h * group.w, 0);
            std::vector<float> values(region.size(), 0);
            for (auto p : parent.pixels) {
                const auto q = (p / w - parent.y0) * group.w + p % w - parent.x0;
                region[q] = 1; values[q] = peak_image ? peak_image[p] : filtered[p];
            }
            if (detail) detail->prepare_ms = ms(started_group);
            const auto marker_start = detail ? Clock::now() : Clock::time_point{};
            auto seeds = deblend ? markers(values, region, group.h, group.w, prominence, separation) : std::vector<std::size_t>{};
            if (detail) {
                detail->marker_ms = ms(marker_start);
                detail->markers = seeds.size();
            }
            if (seeds.size() <= 1) {
                group.labels.assign(region.begin(), region.end()); group.children = 1;
            } else for (;;) {
                const auto flood_start = detail ? Clock::now() : Clock::time_point{};
                watershed(values, region, group.h, group.w, seeds, group.labels, work, detail);
                if (detail) {
                    ++detail->passes;
                    detail->watershed_ms += ms(flood_start);
                }
                const auto prune_start = detail ? Clock::now() : Clock::time_point{};
                work.areas.assign(seeds.size() + 1, 0);
                work.retained.clear();
                for (auto owner : group.labels) ++work.areas[static_cast<std::size_t>(owner)];
                for (std::size_t j = 0; j < seeds.size(); ++j)
                    if (work.areas[j + 1] >= min_area) work.retained.push_back(seeds[j]);
                const bool complete = work.retained.size() == seeds.size();
                if (!complete && work.retained.empty())
                    work.retained.push_back(*std::min_element(seeds.begin(), seeds.end(), [&](auto a, auto b) { return values[a] > values[b] || (values[a] == values[b] && a < b); }));
                if (detail) detail->pruning_ms += ms(prune_start);
                if (complete) { group.children = seeds.size(); group.peaks = seeds; break; }
                seeds.swap(work.retained);
                if (seeds.size() == 1) { group.labels.assign(region.begin(), region.end()); group.children = 1; break; }
            }
            if (detail) {
                detail->children = group.children;
                detail->total_ms = ms(started_group);
            }
        }
    });
    result.timings.deblend_ms = ms(phase);
    phase = Clock::now();
    workers.run(groups.size(), [&](auto, auto begin, auto end) {
        for (auto i = begin; i < end; ++i) {
            const auto& parent = parents[i]; auto& group = groups[i];
            struct Sums { double weight = 0, x = 0, y = 0, gx = 0, gy = 0; std::size_t count = 0; };
            std::vector<Sums> sums(group.children);
            group.rows.assign(group.children, Measurement{0, 0, w, h, 0, 0, 0, 0});
            for (auto p : parent.pixels) {
                const auto q = (p / w - parent.y0) * group.w + p % w - parent.x0;
                const auto child = static_cast<std::size_t>(group.labels[q] - 1);
                auto& row = group.rows[child]; auto& sum = sums[child];
                const auto y = p / w, x = p % w; const double value = residual[p], weight = std::max(value, 0.0);
                row.flux += value; ++row.area;
                row.x0 = std::min(row.x0, x); row.x1 = std::max(row.x1, x);
                row.y0 = std::min(row.y0, y); row.y1 = std::max(row.y1, y);
                bool central = true;
                if (group.children > 1) {
                    const auto peak = group.peaks[child];
                    const auto py = parent.y0 + peak / group.w, px = parent.x0 + peak % group.w;
                    central = std::abs(static_cast<double>(x) - px) <= centroid_radius &&
                              std::abs(static_cast<double>(y) - py) <= centroid_radius;
                }
                if (central) {
                    sum.weight += weight; sum.x += x * weight; sum.y += y * weight;
                    sum.gx += x; sum.gy += y; ++sum.count;
                }
            }
            for (std::size_t j = 0; j < group.children; ++j) {
                auto& row = group.rows[j]; const auto& sum = sums[j];
                if (!sum.count) throw std::logic_error("source marker must own its centroid window");
                row.x = sum.weight > 0 ? sum.x / sum.weight : sum.gx / sum.count;
                row.y = sum.weight > 0 ? sum.y / sum.weight : sum.gy / sum.count;
            }
        }
    });
    result.timings.measurement_ms = ms(phase);
    result.labels.assign(count, 0);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const auto& group = groups[i]; const auto& parent = parents[i];
        if (offset + group.children > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) throw std::overflow_error("too many source labels");
        for (auto p : parent.pixels) {
            const auto q = (p / w - parent.y0) * group.w + p % w - parent.x0;
            result.labels[p] = group.labels[q] + static_cast<std::int32_t>(offset);
        }
        result.rows.insert(result.rows.end(), group.rows.begin(), group.rows.end()); offset += group.children;
    }
    result.timings.total_ms = ms(started);
    return result;
}
} // namespace astro_cpu
