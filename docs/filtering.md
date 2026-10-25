# FFT filtering baseline

The team baseline uses `scipy.signal.fftconvolve` on CPU and
`cupyx.scipy.signal.fftconvolve` on GPU. CuPy uses cuFFT for NVIDIA FFTs;
there is no separate custom cuFFT wrapper or native build for this branch.
Custom C++/CUDA experiments remain on `samuel-cuda-testing`.

## Shared interface

```python
from astro_pipeline.filtering import smooth

# residual or tile.data: already background-subtracted, finite 2D pixels
filtered = smooth(residual, kernel_sigma=2.0, backend="cpu")
# Same API on a GPU node with CuPy installed:
filtered_gpu = smooth(residual, kernel_sigma=2.0, backend="gpu")
```

The function preserves the input and returns an independent, contiguous
float32 NumPy array with the same shape. It accepts strided arrays and converts
real numeric inputs to float32. Ingestion must resolve NaN/inf as specified in
`io_fits.read_fits`; the filter rejects them to prevent FFT contamination.

Both backends use the same normalized, sampled Gaussian kernel with radius
**ceil(4 * kernel_sigma)**. The image is padded with edge-repeating reflection
(`pad(mode="symmetric")`, equivalent to ndimage's `mode="reflect"`). Linear FFT
convolution in `valid` mode then removes padding. This avoids circular wraparound
and artificial dark borders from zero-padding the science image.

This finite support matches the shared tile halo requirement. Tile cores should
agree with whole-image filtering within float32 FFT roundoff. For fractional
sigma, the explicit ceiling radius can differ from SciPy's default rounded
radius; comparisons must supply the same radius.

## Setup and validation

```sh
python -m pip install -r docs/requirements.txt
python -m pytest -q tests
```

On the GPU node, load the appropriate CUDA environment and install its matching
CuPy package, for example `python -m pip install cupy-cuda12x` for a compatible
CUDA 12 environment. CPU-only imports and filtering do not require CuPy.

GPU tests skip only if CuPy or a usable CUDA device is unavailable. Once a GPU
is available, convolution failures and numerical mismatches fail the tests.
Tests compare the FFT results with an independent spatial-domain Gaussian,
exercise edges and impulses, compare Humza's filtered tile cores with whole
images, and check CPU/GPU agreement with rtol=1e-4 plus scale-aware absolute
tolerance near zero. GPU requests never silently fall back to CPU.

## Integration scope

The branch starts from `main` (`d5ee8f5`) merged with Humza's original
`humza_dev` history (`e48f843`). The tiler and its tests are inherited from
that merge, rather than added by the filtering commit. Shared `interfaces.py`
remains version 2.
Only filtering is implemented here. Background estimation, detection, FITS I/O,
batch scheduling, and catalog validation still contain pipeline stubs.

The GPU call uploads pixels and the kernel, performs padding and convolution
on the device, then synchronously returns the result to host memory. Persistent
device batches and transfer/compute overlap are later performance work. This
baseline does not establish end-to-end source recovery or the 2x throughput
target, and filtering equivalence does not establish catalog equivalence across
tile boundaries.

Reference APIs: [SciPy FFT convolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html),
[CuPy FFT convolution](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.signal.fftconvolve.html),
and [CuPy FFT/cuFFT support](https://docs.cupy.dev/en/stable/user_guide/fft.html).
