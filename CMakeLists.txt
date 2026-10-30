cmake_minimum_required(VERSION 3.24)
project(astronomy_background LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 2.13 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_status)
  if(NOT pybind11_status EQUAL 0)
    message(FATAL_ERROR "Install pybind11>=2.13 in the selected Python environment or set pybind11_DIR")
  endif()
  find_package(pybind11 2.13 CONFIG REQUIRED HINTS "${pybind11_cmake_dir}")
endif()

pybind11_add_module(_background_native native/background.cpp native/background_bindings.cpp)
target_link_libraries(_background_native PRIVATE Threads::Threads)
target_compile_features(_background_native PRIVATE cxx_std_20)
target_compile_options(_background_native PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math;-ffp-contract=off>")
set_target_properties(_background_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/astro_pipeline")
