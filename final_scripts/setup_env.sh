# setup_env.sh -- source this file (do not execute it):   source setup_env.sh
#
# 1. Loads CUDA FIRST.  On Vista, loading cuda changes MODULEPATH and Lmod
#    reloads gcc as the newest version (15.1.0), so the compiler has to be
#    chosen after cuda, not before.
# 2. Loads a gcc that nvcc accepts.  CUDA 12.8's nvcc stops with
#      "unsupported GNU version! gcc versions later than 14 are not supported"
#    so this loads the newest gcc module whose major version is <= 14
#    (override with   PS4_GCC=gcc/13.2.0 source setup_env.sh).
#    Fallbacks: the system g++ via -ccbin, then -allow-unsupported-compiler.
# 3. Finds Nsight Compute (ncu) and Nsight Systems (nsys). With the Vista
#    NVHPC modules, the Arm SBSA executable may be directly under a
#    target-linux-sbsa-armv8 directory rather than a bin/ directory.

_ps4_major() { "$1" -dumpversion 2>/dev/null | cut -d. -f1; }

# ---- 1. cuda first ----
module load gcc cuda/12.8

# ---- 2. a gcc <= 14, loaded after cuda ----
if [ -n "${PS4_GCC:-}" ]; then
    _ps4_gcc_mod=$PS4_GCC
else
    _ps4_gcc_mod=$(module -t avail gcc 2>&1 | grep -oE '^gcc/[0-9]+(\.[0-9]+)*' \
                   | sort -uV | awk -F'[/.]' '$2 <= 14' | tail -1)
fi
echo "[setup_env] gcc module chosen: ${_ps4_gcc_mod:-gcc/13 (none found in the list)}"
module load ${_ps4_gcc_mod:-gcc/13}

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[setup_env] WARNING: nvcc not found after loading cuda"
fi

if [ -z "${NVCC:-}" ] && [ "$(_ps4_major gcc)" -gt 14 ] 2>/dev/null; then
    if [ -x /usr/bin/g++ ] && [ "$(_ps4_major /usr/bin/g++)" -le 14 ] 2>/dev/null; then
        export NVCC="nvcc -ccbin /usr/bin/g++"
        echo "[setup_env] gcc on PATH is still newer than 14; nvcc will use /usr/bin/g++ instead"
    else
        export NVCC="nvcc -allow-unsupported-compiler"
        echo "[setup_env] WARNING: no gcc <= 14 found; using -allow-unsupported-compiler (may fail)"
    fi
fi

# ---- 3. Nsight tools (search the CUDA/NVHPC tree if they are not on PATH) ----
_ps4_nvcc=$(command -v nvcc 2>/dev/null)
if [ -n "$_ps4_nvcc" ]; then
    _ps4_cudadir=$(dirname "$(dirname "$_ps4_nvcc")")               # .../cuda/12.8
    _ps4_sdk=$(dirname "$(dirname "$_ps4_cudadir")")                # .../25.3
    for _ps4_tool in ncu nsys; do
        command -v "$_ps4_tool" >/dev/null 2>&1 && continue
        # The Nsight Systems Arm SBSA launcher is commonly named `nsys` at
        # target-linux-sbsa-armv8/nsys (no bin/ component). Search by filename
        # so both that layout and the conventional bin/nsys layout work.
        _ps4_found=$(find "$_ps4_sdk/profilers" "$_ps4_sdk/cuda" "$_ps4_cudadir" \
                          -maxdepth 8 -type f -name "$_ps4_tool" -executable 2>/dev/null \
                     | awk -F/ '{print NF" "$0}' | sort -k1,1n -k2,2Vr | head -1 | cut -d' ' -f2-)
        if [ -n "$_ps4_found" ]; then
            export PATH="$(dirname "$_ps4_found"):$PATH"
            echo "[setup_env] found $_ps4_tool: $_ps4_found"
        else
            echo "[setup_env] WARNING: $_ps4_tool not found; its profiling steps will be skipped"
        fi
    done
fi

echo "[setup_env] host compiler: $(gcc --version 2>&1 | head -1)"
echo "[setup_env] nvcc command : ${NVCC:-nvcc}"
echo "[setup_env] ncu: $(command -v ncu || echo missing)   nsys: $(command -v nsys || echo missing)"
unset _ps4_gcc_mod _ps4_nvcc _ps4_cudadir _ps4_sdk _ps4_tool _ps4_found
