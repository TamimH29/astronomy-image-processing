add_library(astroproc src/processing.cpp)
add_library(astroproc::astroproc ALIAS astroproc)
target_include_directories(astroproc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(astroproc PUBLIC cxx_std_20)
set_target_properties(astroproc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(astroproc PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")

if(ASTROPROC_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "nvcc host compiler")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(astroproc PRIVATE src/cuda_filter.cu)
  target_link_libraries(astroproc PRIVATE CUDA::cudart)
  target_compile_options(astroproc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  set_target_properties(astroproc PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
else()
  target_sources(astroproc PRIVATE src/cuda_stub.cpp)
endif()

if(ASTROPROC_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 2.13 CONFIG REQUIRED)
  pybind11_add_module(astroproc_python bindings/python.cpp)
  set_target_properties(astroproc_python PROPERTIES
    OUTPUT_NAME astroproc
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_link_libraries(astroproc_python PRIVATE astroproc)
  if(BUILD_TESTING)
    add_test(NAME filtering_integration
      COMMAND "${Python_EXECUTABLE}" -m pytest -q "${PROJECT_SOURCE_DIR}/tests")
    set_tests_properties(filtering_integration PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:astroproc_python>:${PROJECT_SOURCE_DIR};PYTHONDONTWRITEBYTECODE=1")
  endif()
endif()
