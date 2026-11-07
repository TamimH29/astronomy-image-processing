#include "background.hpp"
#include "image_ops.hpp"
#include <algorithm>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
using Pixels = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Mask = py::array_t<bool, py::array::c_style | py::array::forcecast>;
void image_shape(const Pixels& pixels) {
    if (pixels.ndim() != 2 || pixels.shape(0) <= 0 || pixels.shape(1) <= 0)
        throw py::value_error("image must be a nonempty 2D array");
}
Mask mask_input(const py::object& value, const Pixels& pixels) {
    if (value.is_none()) return Mask();
    auto raw = py::array::ensure(value);
    if (!raw || !raw.dtype().is(py::dtype::of<bool>()) || raw.ndim() != 2 ||
            raw.shape(0) != pixels.shape(0) || raw.shape(1) != pixels.shape(1))
        throw py::value_error("valid_mask must be a boolean array with the image shape");
    return Mask(raw);
}
py::object estimate(Pixels pixels, py::ssize_t box, py::ssize_t threads,
                    py::object valid_mask, bool refine, bool profile,
                    py::object statistics) {
    image_shape(pixels);
    if (box <= 0 || threads <= 0) throw py::value_error("box and threads must be positive integers");
    auto mask = mask_input(valid_mask, pixels);
    const auto height = static_cast<std::size_t>(pixels.shape(0)), width = static_cast<std::size_t>(pixels.shape(1));
    py::array_t<float> output({pixels.shape(0), pixels.shape(1)});
    auto* destination = output.mutable_data(); const auto* source = pixels.data();
    const bool* valid = valid_mask.is_none() ? nullptr : mask.data();
    astro_background::GlobalStatistics provider;
    if (!statistics.is_none()) {
        if (!PyCallable_Check(statistics.ptr()))
            throw py::type_error("global_statistics must be callable");
        // Reuse an owning mask snapshot so a callback cannot retain a dangling
        // view into the native refinement loop's temporary source mask.
        py::array_t<std::uint8_t> snapshot({pixels.shape(0), pixels.shape(1)});
        auto* snapshot_data = snapshot.mutable_data();
        provider = [statistics, output, snapshot, snapshot_data, height, width]
                   (const float* level, const std::uint8_t* source_mask) {
            py::gil_scoped_acquire acquire;
            py::object level_arg = py::none(), source_arg = py::none();
            if (level) {
                py::array_t<float> view(
                    {static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width)},
                    {static_cast<py::ssize_t>(width * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
                    level, output);
                view.attr("setflags")(false);
                level_arg = std::move(view);
                std::copy_n(source_mask, height * width, snapshot_data);
                py::array_t<std::uint8_t> source_view(
                    {static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width)},
                    {static_cast<py::ssize_t>(width), py::ssize_t{1}},
                    snapshot_data, snapshot);
                source_view.attr("setflags")(false);
                source_arg = std::move(source_view);
            }
            const auto result = statistics(level_arg, source_arg)
                                    .cast<std::pair<double, double>>();
            if (!std::isfinite(result.first) || !std::isfinite(result.second) || result.second < 0)
                throw py::value_error("global_statistics must return a finite center and nonnegative finite noise");
            return result;
        };
    }
    astro_background::Timings timings; astro_background::Diagnostics diagnostics;
    double noise;
    {
        py::gil_scoped_release release;
        noise = astro_background::estimate(source, height, width, width,
                    static_cast<std::size_t>(box), static_cast<std::size_t>(threads),
                    destination, profile ? &timings : nullptr, valid, refine,
                    profile ? &diagnostics : nullptr, provider);
    }
    if (!profile) return py::make_tuple(output, noise);
    py::dict measured;
    measured["mesh_ms"] = timings.mesh_ms;
    measured["interpolation_ms"] = timings.interpolation_ms;
    measured["noise_ms"] = timings.noise_ms;
    measured["global_median_ms"] = timings.global_median_ms;
    measured["source_mask_ms"] = timings.source_mask_ms;
    measured["refinement_passes"] = timings.refinement_passes;
    measured["total_ms"] = timings.total_ms;
    measured["effective_threads"] = timings.effective_threads;
    py::array_t<bool> sources({pixels.shape(0), pixels.shape(1)});
    std::copy(diagnostics.source_mask.begin(), diagnostics.source_mask.end(), sources.mutable_data());
    py::array_t<double> mesh({static_cast<py::ssize_t>(diagnostics.ny), static_cast<py::ssize_t>(diagnostics.nx)});
    std::copy(diagnostics.mesh.begin(), diagnostics.mesh.end(), mesh.mutable_data());
    py::array_t<bool> observed({static_cast<py::ssize_t>(diagnostics.ny), static_cast<py::ssize_t>(diagnostics.nx)});
    std::copy(diagnostics.measured_mesh.begin(), diagnostics.measured_mesh.end(), observed.mutable_data());
    py::dict result;
    result["background"] = output; result["noise"] = noise; result["timings"] = measured;
    result["source_mask"] = sources; result["mesh"] = mesh; result["measured_mesh"] = observed;
    return result;
}

py::object detect(Pixels residual, Pixels filtered, double threshold, py::ssize_t min_area,
                  bool deblend, py::ssize_t separation, double prominence, py::ssize_t threads,
                  py::object valid_mask, bool profile, py::object peak_image, py::ssize_t centroid_radius) {
    image_shape(residual); image_shape(filtered);
    if (filtered.shape(0) != residual.shape(0) || filtered.shape(1) != residual.shape(1))
        throw py::value_error("filtered image must have the residual shape");
    if (min_area <= 0 || separation <= 0 || threads <= 0 || centroid_radius <= 0)
        throw py::value_error("detection sizes and threads must be positive");
    auto mask = mask_input(valid_mask, residual);
    Pixels peaks;
    if (!peak_image.is_none()) {
        peaks = Pixels::ensure(peak_image);
        if (!peaks || peaks.ndim() != 2 || peaks.shape(0) != residual.shape(0) || peaks.shape(1) != residual.shape(1))
            throw py::value_error("peak_image must have the residual shape");
    }
    const auto* raw = residual.data(); const auto* image = filtered.data();
    const auto* valid = valid_mask.is_none() ? nullptr : mask.data();
    const auto* sharp = peak_image.is_none() ? nullptr : peaks.data();
    const auto h = static_cast<std::size_t>(residual.shape(0)), w = static_cast<std::size_t>(residual.shape(1));
    astro_cpu::DetectionResult computed;
    {
        py::gil_scoped_release release;
        computed = astro_cpu::detect(raw, image, valid, h, w, threshold,
                    static_cast<std::size_t>(min_area), deblend, static_cast<std::size_t>(separation),
                    prominence, static_cast<std::size_t>(threads), sharp,
                    static_cast<std::size_t>(centroid_radius));
    }
    py::list rows;
    for (const auto& row : computed.rows)
        rows.append(py::make_tuple(row.x, row.y, row.x0, row.y0, row.x1, row.y1, row.area, row.flux));
    if (!profile) return rows;
    py::array_t<std::int32_t> labels({residual.shape(0), residual.shape(1)});
    std::copy(computed.labels.begin(), computed.labels.end(), labels.mutable_data());
    py::dict timings;
    timings["label_ms"] = computed.timings.label_ms;
    timings["deblend_ms"] = computed.timings.deblend_ms;
    timings["measurement_ms"] = computed.timings.measurement_ms;
    timings["effective_threads"] = computed.timings.effective_threads;
    timings["total_ms"] = computed.timings.total_ms;
    py::dict result; result["rows"] = rows; result["labels"] = labels; result["timings"] = timings;
    return result;
}
} // namespace

PYBIND11_MODULE(_background_native, module) {
    module.attr("API_VERSION") = 2;
    module.attr("GLOBAL_STATISTICS_API_VERSION") = 1;
    py::register_exception<astro_background::FloatingPointFailure>(module, "FloatingPointFailure", PyExc_FloatingPointError);
    module.def("estimate_background", [](Pixels pixels, py::ssize_t box, py::ssize_t threads, py::object valid, bool refine, py::object statistics) {
        return estimate(std::move(pixels), box, threads, std::move(valid), refine, false, std::move(statistics));
    }, py::arg("pixels"), py::arg("box") = 64, py::arg("threads") = 1,
       py::kw_only(), py::arg("valid_mask") = py::none(), py::arg("refine") = true,
       py::arg("global_statistics") = py::none());
    module.def("profile_background", [](Pixels pixels, py::ssize_t box, py::ssize_t threads, py::object valid, bool refine, py::object statistics) {
        return estimate(std::move(pixels), box, threads, std::move(valid), refine, true, std::move(statistics));
    }, py::arg("pixels"), py::arg("box") = 64, py::arg("threads") = 1,
       py::kw_only(), py::arg("valid_mask") = py::none(), py::arg("refine") = true,
       py::arg("global_statistics") = py::none());
    module.def("detect_sources", &detect, py::arg("residual"), py::arg("filtered"),
       py::arg("threshold"), py::arg("min_area"), py::arg("deblend"), py::arg("separation"),
       py::arg("prominence"), py::arg("threads") = 1, py::kw_only(),
       py::arg("valid_mask") = py::none(), py::arg("profile") = false,
       py::arg("peak_image") = py::none(), py::arg("centroid_radius") = 2);
}
