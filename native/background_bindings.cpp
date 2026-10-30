#include "background.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace {
using Pixels = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::object estimate(Pixels pixels, py::ssize_t box, py::ssize_t threads, bool profile) {
    if (pixels.ndim() != 2 || pixels.shape(0) <= 0 || pixels.shape(1) <= 0)
        throw py::value_error("image must be a nonempty 2D array");
    if (box <= 0 || threads <= 0)
        throw py::value_error("box and threads must be positive integers");
    py::array_t<float> output({pixels.shape(0), pixels.shape(1)});
    auto* destination = output.mutable_data();
    const auto* source = pixels.data();
    const auto height = static_cast<std::size_t>(pixels.shape(0));
    const auto width = static_cast<std::size_t>(pixels.shape(1));
    astro_background::Timings timings;
    double noise;
    {
        py::gil_scoped_release release;
        noise = astro_background::estimate(source, height, width, width,
                                           static_cast<std::size_t>(box),
                                           static_cast<std::size_t>(threads), destination,
                                           profile ? &timings : nullptr);
    }
    if (!profile) return py::make_tuple(output, noise);
    py::dict measured;
    measured["mesh_ms"] = timings.mesh_ms;
    measured["interpolation_ms"] = timings.interpolation_ms;
    measured["noise_ms"] = timings.noise_ms;
    measured["global_median_ms"] = timings.global_median_ms;
    measured["total_ms"] = timings.total_ms;
    measured["effective_threads"] = timings.effective_threads;
    py::dict result;
    result["background"] = output;
    result["noise"] = noise;
    result["timings"] = measured;
    return result;
}
} // namespace

PYBIND11_MODULE(_background_native, module) {
    py::register_exception<astro_background::FloatingPointFailure>(
        module, "FloatingPointFailure", PyExc_FloatingPointError);
    module.def("estimate_background", [](Pixels data, py::ssize_t box, py::ssize_t threads) {
        return estimate(std::move(data), box, threads, false);
    }, py::arg("pixels"), py::arg("box") = 64, py::arg("threads") = 1);
    module.def("profile_background", [](Pixels data, py::ssize_t box, py::ssize_t threads) {
        return estimate(std::move(data), box, threads, true);
    }, py::arg("pixels"), py::arg("box") = 64, py::arg("threads") = 1);
}
