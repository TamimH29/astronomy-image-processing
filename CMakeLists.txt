cmake_minimum_required(VERSION 3.24)
project(astronomy_image_processing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ASTROPROC_ENABLE_CUDA "Build Samuel's custom CUDA Gaussian filter" OFF)
option(ASTROPROC_BUILD_PYTHON "Build the NumPy adapter" ON)

include(CTest)
add_subdirectory(native/astroproc)
