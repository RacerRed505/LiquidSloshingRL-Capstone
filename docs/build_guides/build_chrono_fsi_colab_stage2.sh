#!/bin/bash
# Build Chrono from source with FSI + Python + Irrlicht enabled, on Colab.
# STAGE 2 of 2: the actual (expensive) build + install + verify, run
# AFTER build_chrono_fsi_colab.sh (stage 1) has cloned and configured.
# See that file's header for why this is split into two scripts/cells --
# short version: it's the only way to get a real checkpoint pause, since
# Colab can't run a second cell while a `!bash script.sh` cell is still
# executing in the same kernel.
#
# REFERENCE SCRIPT -- not runnable on this project's own development
# machine (Intel Arc GPU, no CUDA). Same paths as stage 1 -- must be run
# in the SAME Colab session stage 1 ran in (or after re-uploading a
# checkpoint zip that restores the same /content/chrono-fsi-build layout).
#
# Usage (from a Colab cell, after uploading this file):
#   !bash build_chrono_fsi_colab_stage2.sh

set -e

# ============================================================
# EDIT THESE PATHS IF NEEDED -- must match stage 1's paths exactly
# ============================================================
CHRONO_ROOT="/content/chrono-fsi-build"
CHRONO_BUILD_DIR="$CHRONO_ROOT/chrono-build"
CHRONO_INSTALL_DIR="$CHRONO_ROOT/chrono-install"

PYTHON_EXECUTABLE="$(which python3)"

if [ ! -d "$CHRONO_BUILD_DIR" ]; then
    echo "FATAL: $CHRONO_BUILD_DIR not found -- run build_chrono_fsi_colab.sh (stage 1) first."
    exit 1
fi

# Verification helper -- NOT a plain `import pychrono.fsi`. Confirmed live
# (see the development log Entry 34) that every built .so has its runtime search
# path stripped to empty by `cmake --install`, so nothing finds its sibling
# libraries by default, and LD_LIBRARY_PATH set via the environment doesn't
# help either (the dynamic linker parses it once, early in the process's
# life -- setting it later, even before this script's own python3 call,
# does not reliably override that). ctypes.CDLL with an explicit path and
# RTLD_GLOBAL sidesteps both problems. Also note the real install location
# is share/chrono/python, NOT bin/ -- an earlier version of this script
# assumed bin/ and was wrong.
verify_fsi_import() {
    "$PYTHON_EXECUTABLE" <<PYEOF
import ctypes
for lib in ["libyaml-cpp.so.0.8", "libChrono_core.so", "libChrono_fsi.so", "libChrono_fsisph.so", "libChrono_irrlicht.so"]:
    ctypes.CDLL("$CHRONO_INSTALL_DIR/lib/" + lib, mode=ctypes.RTLD_GLOBAL)
import sys
sys.path.insert(0, "$CHRONO_INSTALL_DIR/share/chrono/python")
import pychrono.fsi as fsi
print("FSI import OK:", fsi)
import pychrono.irrlicht
print("Irrlicht import OK")
PYEOF
}

# Already built? Don't redo hours of work if this gets re-run for any reason.
if verify_fsi_import &> /dev/null; then
    echo "Already built and importable at $CHRONO_INSTALL_DIR -- skipping the build."
    verify_fsi_import
    exit 0
fi

# ============================================================
# Build + Install (Release only -- see the build guide's own note on why)
# ============================================================
echo "=== Building (this will take a while) ==="
cmake --build "$CHRONO_BUILD_DIR" --target install -- -j"$(nproc)"

# ============================================================
# Verify
# ============================================================
echo ""
echo "=== Verifying the build actually has FSI ==="
verify_fsi_import

echo ""
echo "If both lines above printed OK, the build is usable. If this instead"
echo "fails naming a missing library, apt-get install it if it's a system"
echo "package (e.g. libirrlicht1.8), or add it to the ctypes.CDLL preload"
echo "list in verify_fsi_import() above if it's one of Chrono's own .so"
echo "files -- see the project notes' Phase 13-14 section / the development log Entry 34"
echo "for the full explanation. If it instead fails to import at all, re-check"
echo "CH_ENABLE_MODULE_FSI was actually ON during configure (CMake silently"
echo "leaving a module off due to an unmet dependency is the most common"
echo "failure mode here, not a loud error) -- check stage 1's configure log"
echo "for any 'FSI module disabled' or similar notice."

# ============================================================
# Auto-checkpoint: zip the finished build so it can be downloaded
# straight to your laptop -- no Google auth needed. files.download()
# only works from a Python cell, not from inside this bash script, so
# this just prepares the zip; run the printed Python snippet next to
# actually save it off the ephemeral Colab session.
# ============================================================
echo ""
echo "=== Zipping finished build for download ==="
cd /content && zip -r -q chrono-fsi-build-checkpoint.zip chrono-fsi-build
echo "Zip ready at /content/chrono-fsi-build-checkpoint.zip"
echo ""
echo "RUN THIS IN A NEW PYTHON CELL NOW to save it to your laptop:"
echo "  from google.colab import files"
echo "  files.download('/content/chrono-fsi-build-checkpoint.zip')"
