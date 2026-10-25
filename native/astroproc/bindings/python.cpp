#include "astroproc/astroproc.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;
using namespace astroproc;
namespace {
using Pixels = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Mask = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
// Holds converted arrays for the complete duration of each synchronous call.
struct Input {
    Pixels pixels;
    Mask mask;
    ImageView image;
    Input(Pixels data, py::object invalid) : pixels(std::move(data)) {
        if (pixels.ndim() != 2 || pixels.shape(0) <= 0 || pixels.shape(1) <= 0)
            throw std::invalid_argument("pixels must be a nonempty 2D array");
        image = {pixels.data(), static_cast<std::size_t>(pixels.shape(1)), static_cast<std::size_t>(pixels.shape(0)), 0, nullptr, 0};
        if (!invalid.is_none()) {
            // Compare before conversion so nonzero values such as 256 remain invalid.
            auto raw = py::array::ensure(invalid);
            if (!raw || raw.ndim() != 2 || raw.shape(0) != pixels.shape(0) || raw.shape(1) != pixels.shape(1))
                throw std::invalid_argument("invalid mask shape must match pixels");
            py::object bool_mask = py::module_::import("numpy").attr("not_equal")(raw, 0);
            mask = Mask::ensure(bool_mask);
            if (!mask) throw std::invalid_argument("invalid mask cannot be converted");
            image.invalid = mask.data();
        }
        validate_image(image);
    }
};
py::array_t<float> pixels_array(const Image& image) {
    py::array_t<float> result({static_cast<py::ssize_t>(image.height), static_cast<py::ssize_t>(image.width)});
    std::memcpy(result.mutable_data(), image.pixels.data(), image.pixels.size() * sizeof(float));
    return result;
}
py::array_t<std::uint8_t> mask_array(const Image& image) {
    py::array_t<std::uint8_t> result({static_cast<py::ssize_t>(image.height), static_cast<py::ssize_t>(image.width)});
    if (image.invalid.empty()) std::memset(result.mutable_data(), 0, image.pixels.size());
    else std::memcpy(result.mutable_data(), image.invalid.data(), image.invalid.size());
    return result;
}
py::list sources_list(const DetectionResult& result) {
    py::list sources;
    for (const auto& s : result.sources) {
        py::dict item;
        item["id"] = s.id; item["centroid_x"] = s.centroid_x; item["centroid_y"] = s.centroid_y;
        item["x_min"] = s.x_min; item["x_max"] = s.x_max; item["y_min"] = s.y_min; item["y_max"] = s.y_max;
        item["area"] = s.area; item["integrated_brightness"] = s.integrated_brightness; item["flags"] = s.flags;
        sources.append(std::move(item));
    }
    return sources;
}
Backend backend_value(const std::string& name) {
    if (name == "cpu") return Backend::Cpu;
    if (name == "cuda") return Backend::Cuda;
    throw std::invalid_argument("backend must be cpu or cuda");
}
} // namespace

PYBIND11_MODULE(astroproc, m) {
    m.doc() = "Native C++/CUDA astronomy processing; Python supplies arrays only.";
    py::class_<ProcessingConfig>(m, "ProcessingConfig")
        .def(py::init<>())
        .def_readwrite("gaussian_sigma", &ProcessingConfig::gaussian_sigma)
        .def_readwrite("detection_sigma", &ProcessingConfig::detection_sigma)
        .def_readwrite("min_source_area", &ProcessingConfig::min_source_area)
        .def_readwrite("background_mesh", &ProcessingConfig::background_mesh)
        .def_readwrite("min_mesh_samples", &ProcessingConfig::min_mesh_samples)
        .def_readwrite("clip_iterations", &ProcessingConfig::clip_iterations)
        .def_readwrite("clip_sigma", &ProcessingConfig::clip_sigma);
    py::class_<ImageMetadata>(m, "ImageMetadata")
        .def(py::init<>())
        .def_readwrite("image_id", &ImageMetadata::image_id)
        .def_readwrite("intensity_unit", &ImageMetadata::intensity_unit)
        .def_readwrite("offset_x", &ImageMetadata::offset_x)
        .def_readwrite("offset_y", &ImageMetadata::offset_y);
    m.def("cuda_availability", [] {
        auto a = cuda_availability(); py::dict result;
        result["available"] = a.available; result["reason"] = a.reason; return result;
    });
    m.def("filter_radius", &filter_radius, py::arg("config") = ProcessingConfig{});
    m.def("prepare_image", [](Pixels pixels, py::object invalid, ProcessingConfig c) {
        Input input(std::move(pixels), invalid); PreparedImage prepared;
        { py::gil_scoped_release release; prepared = prepare_image(input.image, c); }
        py::dict result;
        result["background"] = pixels_array(prepared.background); result["residual"] = pixels_array(prepared.residual);
        result["invalid"] = mask_array(prepared.residual); result["warnings"] = prepared.warnings; return result;
    }, py::arg("pixels"), py::arg("invalid") = py::none(), py::arg("config") = ProcessingConfig{});
    m.def("filter_image", [](Pixels pixels, py::object invalid, ProcessingConfig c, const std::string& backend) {
        Input input(std::move(pixels), invalid); const auto b = backend_value(backend); Image filtered;
        { py::gil_scoped_release release; filtered = b == Backend::Cpu ? filter_cpu(input.image, c) : filter_cuda(input.image, c); }
        py::dict result; result["pixels"] = pixels_array(filtered); result["invalid"] = mask_array(filtered); return result;
    }, py::arg("pixels"), py::arg("invalid") = py::none(), py::arg("config") = ProcessingConfig{}, py::arg("backend") = "cpu");
    m.def("detect_sources", [](Pixels original, Pixels background, Pixels filtered, py::object invalid,
                                py::object filtered_invalid, ImageMetadata metadata, ProcessingConfig c) {
        Input raw(std::move(original), invalid), bg(std::move(background), py::none()), smooth(std::move(filtered), filtered_invalid);
        DetectionResult detection;
        { py::gil_scoped_release release; detection = detect_sources(raw.image, bg.image, smooth.image, metadata, c); }
        py::dict result; result["sources"] = sources_list(detection); result["warnings"] = detection.warnings;
        py::array_t<std::uint64_t> labels({static_cast<py::ssize_t>(raw.image.height), static_cast<py::ssize_t>(raw.image.width)});
        std::memcpy(labels.mutable_data(), detection.labels.data(), detection.labels.size() * sizeof(std::uint64_t));
        result["labels"] = std::move(labels); return result;
    }, py::arg("original"), py::arg("background"), py::arg("filtered_residual"), py::arg("invalid") = py::none(),
       py::arg("filtered_invalid") = py::none(), py::arg("metadata") = ImageMetadata{}, py::arg("config") = ProcessingConfig{});
    m.def("process_image", [](Pixels pixels, py::object invalid, ImageMetadata metadata,
                               ProcessingConfig c, const std::string& backend) {
        Input input(std::move(pixels), invalid); const auto b = backend_value(backend); ProcessingResult processed;
        { py::gil_scoped_release release; processed = process_image(input.image, metadata, c, b); }
        py::dict result;
        result["filtered"] = pixels_array(processed.filtered); result["invalid"] = mask_array(processed.filtered);
        result["sources"] = sources_list(processed.detection); result["warnings"] = processed.detection.warnings;
        result["image_id"] = processed.metadata.image_id; result["intensity_unit"] = processed.metadata.intensity_unit;
        py::dict timings; timings["prepare_ms"] = processed.timings.prepare_ms;
        timings["filter_ms"] = processed.timings.filter_ms; timings["detect_ms"] = processed.timings.detect_ms;
        result["timings"] = timings; return result;
    }, py::arg("pixels"), py::arg("invalid") = py::none(), py::arg("metadata") = ImageMetadata{},
       py::arg("config") = ProcessingConfig{}, py::arg("backend") = "cpu");
}
