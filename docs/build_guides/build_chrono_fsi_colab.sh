#!/bin/bash
# Build Chrono from source with FSI + Python + Irrlicht enabled, on Colab.
# STAGE 1 of 2: pre-flight checks, Linux prerequisites, clone, configure.
# See build_chrono_fsi_colab_stage2.sh for the actual (expensive) build
# step -- split into two scripts/cells specifically so there's a real
# checkpoint between them where you can save progress to your laptop
# before the long compile starts. A single script can't offer that: while
# `!bash script.sh` is running in one cell, Colab's kernel is busy with
# it and you cannot run a separate cell (like a checkpoint/download
# snippet) at the same time without interrupting -- and killing -- the
# whole script. Two scripts, run as two separate cells, is the only way
# to get a real pause point.
#
# REFERENCE SCRIPT -- not runnable on this project's own development
# machine (Intel Arc GPU, no CUDA). Meant to be run inside a Google Colab
# session with a GPU runtime attached, per documentation/colab_fsi_build.md
# (read that document's "Real risks specific to Colab" section first).
# This is a Linux/Colab port of build_chrono_fsi.ps1 -- same steps, same
# structure, adapted for bash + apt instead of PowerShell + cmake-gui.
#
# Uses /content directly (NOT Google Drive) -- Drive's drive.mount() AND
# the alternative google.colab.auth.authenticate_user() both failed
# identically with "credential propagation was unsuccessful", even after
# a runtime restart and browser troubleshooting (see the development log) --
# pointing to a browser/network-level communication issue, not something
# fixable from inside the notebook. Checkpointing instead uses a plain
# zip + google.colab.files.download(), which needs no Google auth at all.
#
# Usage (from a Colab cell, after uploading this file):
#   !bash build_chrono_fsi_colab.sh
# Then, once this finishes, see the checkpoint instructions it prints,
# and run build_chrono_fsi_colab_stage2.sh next (same upload approach).

set -e

# ============================================================
# EDIT THESE PATHS IF NEEDED
# ============================================================
CHRONO_ROOT="/content/chrono-fsi-build"
CHRONO_SOURCE_DIR="$CHRONO_ROOT/chrono"
CHRONO_BUILD_DIR="$CHRONO_ROOT/chrono-build"
CHRONO_INSTALL_DIR="$CHRONO_ROOT/chrono-install"

# Colab's own Python -- NOT necessarily the same as this project's conda
# env, since this build runs inside the Colab runtime, not this project's
# `sloshing` conda env. Match this project's own convention (the project notes'
# "Environment setup" section) of never assuming a bare `python` resolves
# to the right interpreter -- resolve it explicitly instead.
PYTHON_EXECUTABLE="$(which python3)"

mkdir -p "$CHRONO_ROOT"

# ============================================================
# Sanity checks before spending time on a build that can't succeed
# ============================================================
echo "=== Pre-flight checks ==="

if ! command -v nvidia-smi &> /dev/null; then
    echo "FATAL: nvidia-smi not found. Runtime -> Change runtime type -> GPU, then reconnect this session."
    exit 1
fi
nvidia-smi | grep -i "NVIDIA-SMI" || true

if ! command -v nvcc &> /dev/null; then
    echo "FATAL: nvcc not found. This should ship with Colab's GPU runtime image -- if it's missing, the runtime may not actually have a GPU attached."
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep -i "release" || true)
echo "CUDA: $CUDA_VERSION"
if [[ "$CUDA_VERSION" != *"12.9"* ]]; then
    echo "WARNING: expected CUDA 12.9 specifically -- found something else (Colab commonly ships 12.8). Chrono::FSI may still build/run fine, or may not -- this run is the actual test of that. Continuing anyway."
fi

if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "FATAL: could not resolve python3 via 'which python3'."
    exit 1
fi
echo "Using Python: $PYTHON_EXECUTABLE"

# ============================================================
# Already built? Don't redo hours of work by accident.
# Re-running this script (editing a cell, a stray re-run, anything
# short of an actual session reset) should never repeat a finished
# build -- this check makes that true regardless of WHY it got re-run.
# Delete $CHRONO_INSTALL_DIR yourself if you genuinely want to force a
# rebuild from scratch.
#
# NOT a plain `import pychrono.fsi` -- confirmed live (the development log
# Entry 34) that every built .so has its runtime search path stripped to
# empty by `cmake --install`, and LD_LIBRARY_PATH set via the environment
# does not reliably help (the dynamic linker parses it once, early in the
# process's life). ctypes.CDLL with an explicit path and RTLD_GLOBAL
# sidesteps both problems. Real install location is share/chrono/python,
# NOT bin/ -- an earlier version of this script assumed bin/ and was wrong.
# ============================================================
if "$PYTHON_EXECUTABLE" <<PYEOF &> /dev/null
import ctypes
for lib in ["libyaml-cpp.so.0.8", "libChrono_core.so", "libChrono_fsi.so", "libChrono_fsisph.so", "libChrono_irrlicht.so"]:
    ctypes.CDLL("$CHRONO_INSTALL_DIR/lib/" + lib, mode=ctypes.RTLD_GLOBAL)
import sys
sys.path.insert(0, "$CHRONO_INSTALL_DIR/share/chrono/python")
import pychrono.fsi
import pychrono.irrlicht
PYEOF
then
    echo "Already built and importable at $CHRONO_INSTALL_DIR -- skipping everything below."
    echo "Delete that folder first if you actually want a fresh rebuild."
    exit 0
fi

# ============================================================
# Install Linux prerequisites (idempotent -- safe to re-run)
# ============================================================
echo ""
echo "=== Installing Linux prerequisites ==="
apt-get update -qq
apt-get install -y -qq \
    build-essential cmake git swig \
    libeigen3-dev \
    libirrlicht-dev \
    libx11-dev libxrandr-dev libxxf86vm-dev libgl1-mesa-dev
pip install -q six

# ============================================================
# Clone Chrono if not already present
# ============================================================
if [ ! -d "$CHRONO_SOURCE_DIR" ]; then
    echo ""
    echo "=== Cloning Chrono source ==="
    git clone https://github.com/projectchrono/chrono.git "$CHRONO_SOURCE_DIR"
else
    echo "Chrono source already present at $CHRONO_SOURCE_DIR, skipping clone."
fi

# ============================================================
# Configure
# ============================================================
echo ""
echo "=== Configuring (cmake -S -B) ==="
mkdir -p "$CHRONO_BUILD_DIR"

cmake -S "$CHRONO_SOURCE_DIR" -B "$CHRONO_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CHRONO_INSTALL_DIR" \
    -DCH_ENABLE_MODULE_PYTHON=ON \
    -DCH_ENABLE_MODULE_IRRLICHT=ON \
    -DCH_ENABLE_MODULE_FSI=ON \
    -DCH_ENABLE_MODULE_FSI_TDPF=OFF \
    -DCHRONO_GPU_BACKEND=CUDA \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE"

# Note: CMAKE_BUILD_TYPE=Release is set here (not via --config at build
# time) because Unix Makefiles/Ninja are single-config generators, unlike
# the Windows script's Visual Studio generator -- --config Release would
# silently do nothing here if it were used the same way.

echo ""
echo "=== Stage 1 complete: source cloned, build configured ==="
echo "This is a real, safe pause point -- the kernel is free now."
echo ""
echo "Optional: save a checkpoint before the long build starts (run in a"
echo "new Python cell -- cheap to skip and redo since nothing expensive"
echo "has happened yet):"
echo "  import os"
echo "  os.system('cd /content && zip -r -q chrono-fsi-build-checkpoint.zip chrono-fsi-build')"
echo "  from google.colab import files"
echo "  files.download('/content/chrono-fsi-build-checkpoint.zip')"
echo ""
echo "When ready, run build_chrono_fsi_colab_stage2.sh next (upload it the"
echo "same way you uploaded this one, then: !bash build_chrono_fsi_colab_stage2.sh)."
