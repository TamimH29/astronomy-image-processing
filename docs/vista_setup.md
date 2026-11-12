# Setting Up the Astronomy Pipeline on TACC Vista

This guide covers setting up the project on a Vista Grace-Hopper (GH) node for the first time and reconnecting for later sessions. Commands below assume the Vista Bash shell.

## Before You Start

- Replace `<TACC-username>` and `<repository-url>` with your account and repository details.
- Keep the project, virtual environment, and results under `$WORK` so they persist between sessions.
- Run and test compute work on an allocated compute node, not on a login node. A Vista GH node has one H200 GPU and 72 CPU cores.
- Vista's `/tmp` is local to a node and temporary. It is useful for staging input files during a job, but save results back under `$WORK`.

## First-Time Setup

### 1. Connect to Vista

From a terminal with SSH configured for TACC, connect using the Vista login address provided by TACC, for example:

```bash
ssh <TACC-username>@vista.tacc.utexas.edu
```

If your TACC account uses a different host alias or authentication setup, use the one supplied for your account.

### 2. Put the project in `$WORK`

If the repository is not already on Vista, clone it into `$WORK` and enter the project directory:

```bash
cd "$WORK"
git clone <repository-url> astronomy-project
cd "$WORK/astronomy-project"
```

If the project is already there, just `cd` to that directory. Check the branch before running:

```bash
git status --short --branch
```

Use the branch agreed on by your team; do not assume that the local branch name is the current shared integration branch.

### 3. Request a compute node

For an interactive GPU development session, request one GH development node:

```bash
idev -p gh-dev -N 1 -n 1 -t 01:00:00
```

Wait for `idev` to give you a compute-node prompt. When that session ends or disconnects, you must request another node before continuing compute work.

### 4. Load software modules

On the compute node, inspect available Python and CUDA modules, then load the required modules:

```bash
module spider python3
module spider cuda
module load gcc cuda python3
module list
nvidia-smi
nvcc --version
```

Record the CUDA major version from `nvcc --version`; the CuPy package must match it. If you need an exact installed version, use `module spider cuda/<version>` and load that version (for example, `module load cuda/12.6` if available).

### 5. Create and populate the Python environment

Create the environment once, under persistent `$WORK` storage:

```bash
mkdir -p "$WORK/venvs"
python3 -m venv "$WORK/venvs/astro-pipeline"
source "$WORK/venvs/astro-pipeline/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt
```

For the GPU path, install CuPy and its CUDA component libraries into this same environment. Choose the command that matches the CUDA major version loaded above:

```bash
# CUDA 12.x
python -m pip install 'cupy-cuda12x[ctk]'

# CUDA 13.x (use this instead if the node's CUDA Toolkit is 13.x)
python -m pip install 'cupy-cuda13x[ctk]'
```

Do not install both variants or the separate unversioned `cupy` package in the same environment. The `[ctk]` extra supplies CUDA component libraries that can otherwise be missing at runtime (for example, `libcublasLt.so`). CuPy's package names and optional CUDA component installation are documented in its [official installation guide](https://docs.cupy.dev/en/stable/install.html).

The environment is persistent, so the package installs above are first-time setup. Modules, however, must be loaded again in each new shell/session.

### 6. Verify the environment

Confirm Python is using the `$WORK` environment and the main CPU dependencies import:

```bash
which python
python -c "import numpy, scipy, astropy; print('CPU dependencies OK')"
```

For GPU work, verify CuPy, its SciPy signal module, and the visible device:

```bash
python -c "import cupy as cp; import cupyx.scipy.signal; print('CuPy', cp.__version__); print('GPU count:', cp.cuda.runtime.getDeviceCount()); print(cp.arange(5).get())"
```

Expected output includes a GPU count of `1` and `[0 1 4 9 16]`. If CuPy cannot find `libcublasLt`, check that you installed the `[ctk]` package variant matching the node's CUDA major version and that you are testing in the activated environment.

### 7. Run project checks

From the project root, run tests on the allocated node:

```bash
python -m pytest -q
```

Try one image on CPU:

```bash
python -m astro_pipeline.run_pipeline \
  --input fits/<one-real-image>.fits \
  --output "$WORK/astro-test/one.csv" \
  --workers 1 \
  --backend cpu
```

Then, once the CuPy check succeeds, try one image with the GPU backend:

```bash
python -m astro_pipeline.run_pipeline \
  --input fits/<one-real-image>.fits \
  --output "$WORK/astro-test/one-gpu.csv" \
  --workers 1 \
  --backend gpu
```

## Returning for a Later Session

1. SSH to Vista and enter the project directory under `$WORK`.
2. Request a new compute node with `idev`.
3. Load the modules again.
4. Activate the existing virtual environment. Do not recreate it or reinstall packages unless dependencies changed.
5. Check the GPU and Python environment if you will use the GPU backend.

```bash
cd "$WORK/astronomy-project"
idev -p gh-dev -N 1 -n 1 -t 01:00:00

# Run these after idev gives you the compute-node prompt:
module load gcc cuda python3
source "$WORK/venvs/astro-pipeline/bin/activate"
module list
which python
nvidia-smi
```

For a GPU run, verify CuPy before launching the pipeline:

```bash
python -c "import cupy as cp; import cupyx.scipy.signal; print(cp.cuda.runtime.getDeviceCount())"
```

If you need the CPU path only, CuPy and `nvidia-smi` are not required; the CPU dependencies still need to be active in the venv.

## Example: Medium Synthetic Set

From the project root, quote the glob so Bash passes it to the pipeline for expansion. Do not wrap it in backticks:

```bash
python -m astro_pipeline.run_pipeline \
  --input 'gen_fits/medium/*.fits' \
  --output "$WORK/astro-test/one_medium.csv" \
  --workers 1 \
  --tile-size 1024 \
  --backend cpu
```

## References

- [Vista User Guide](https://docs.tacc.utexas.edu/hpc/vista/)
- [Vista launching guide](https://docs.tacc.utexas.edu/hpc/vista/launching/)
- [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
