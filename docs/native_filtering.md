# Native filtering integration

Samuel's existing C++20 library and custom CUDA kernels are in
`native/astroproc/`. The numerical code and pybind11 binding are copied from
the standalone implementation without algorithm changes. Python passes arrays;
the CPU filter executes C++, and the GPU filter executes our CUDA kernels.
No CuPy dependency is needed.

## Build and test

Target GCC **13.2.0 or 14.2.0**, CMake 3.24+, Python development headers,
NumPy, SciPy, pytest, and pybind11 2.13+. Use a CUDA toolkit compatible with
the selected GCC when enabling CUDA. For example, CUDA 12.8 supports both
target compiler versions. Select the compiler actually installed on your node.

```sh
python -m pip install pybind11 numpy scipy pytest
cmake -S . -B build-cpu -DCMAKE_CXX_COMPILER=g++-13 \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

Use a separate build directory for CUDA:

```sh
cmake -S . -B build-cuda -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DASTROPROC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build-cuda -j
PYTHONPATH=build-cuda:. python -m pytest -q tests
```

Architecture `90` targets GH200; change it for other GPUs. GPU execution tests
skip when CUDA is unavailable and include its reported reason. A requested
`backend="gpu"` raises an error if unavailable; it never falls back to CPU.
To build only the C++ library, set `ASTROPROC_BUILD_PYTHON=OFF`.

## Humza's interface

The branch inherits Humza's original commits through an explicit merge of
`main` (`d5ee8f5`) and `humza_dev` (`e48f843`). The tiler and its tests belong
to that merged history; this native change does not introduce copies of them.
Shared `interfaces.py` stays at version 2.
The public filtering entry point and backend names are unchanged:

```python
from astro_pipeline.filtering import smooth
from astro_pipeline.tiling import make_tiles
from astro_pipeline.interfaces import ImageData
import math

# residual is an already background-subtracted 2D float32 array.
sigma = 2.0
tiles = make_tiles(ImageData("exposure", residual), 512, math.ceil(4 * sigma))
for tile in tiles:
    filtered = smooth(tile.data, sigma, backend="gpu")
    # Humza's caller consumes the filtered core using tile.core / x0 / y0.
```

`smooth` returns an independent float32 array with the input shape, preserves
the input, accepts noncontiguous arrays, and releases the GIL during native
computation. Nonfinite pixels are excluded from weighted sums; invalid target
pixels remain NaN because this shared API cannot return a separate mask.

The existing operator uses normalized, separable spatial Gaussian convolution,
reflection at boundaries, and **radius ceil(3*sigma)** on both backends.
Humza's four-sigma halo is sufficient for it. His existing SciPy tests use a
different, four-sigma operator; new integration tests compare actual native
tile cores against actual native whole-image filtering and use an independent
SciPy oracle with the native radius. This preserves the current kernels; it
does not establish four-sigma scientific reference equivalence.

## Scope and validation limits

Only the shared filtering functions are connected in this change. Existing
native background/detection APIs remain available in `astroproc`, but their
shared Python pipeline stubs are unchanged. FITS I/O, source catalogs,
`parallel.run_batch`, and benchmarking remain their owners' work.

Each GPU call currently uploads a tile and downloads its result using the
existing synchronous convenience function. No persistent-device interface,
Python async job bindings, new worker pool, or scheduling redesign is added.
Filtering core equivalence does not prove source catalog equivalence across
tile boundaries. Performance and CPU/GPU overlap require a later integration
and measurements on the target GPU.

The initial local check compiled both extension variants directly with GCC
11.4 and CUDA 12.3 (CUDA target `sm_90`), since CMake and the target GCC versions
were unavailable. Each variant passed 127 pytest cases; 33 GPU cases skipped
because no usable driver was available, alongside four pre-existing pipeline
placeholders. The original three native Python adapter tests also passed.
These results do not verify the CMake build, GCC 13.2/14.2, GPU numerical
parity, or GPU performance.
