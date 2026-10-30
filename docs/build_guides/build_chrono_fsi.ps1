<#
Build Chrono from source with FSI + Python + Irrlicht enabled.

Originally written as a REFERENCE SCRIPT for a real NVIDIA GPU machine
(the college RTX 3090 desktop) since the ORIGINAL dev machine (Intel Arc
GPU, no CUDA) could never run it. As of the development log Entries 50-51,
this project has a SECOND machine (real NVIDIA GPU, RTX 5060) where every
prerequisite below is confirmed installed and working -- the path values
below are filled in for THAT machine, and this is now genuinely about to
be run for real, not just a plan. See the project notes' "Second machine note"
if you're not sure which machine a session is on.

EDIT THE PATHS BELOW before running (on any machine other than the one
they're currently filled in for). Everything after that section is a
straight, non-interactive `cmake -S -B ... -D ...` + `cmake --build`
sequence -- deliberately scripted instead of clicking through cmake-gui,
so this is a "run once, wait, check the end" step rather than a manual
walkthrough every time you rebuild.

Usage (PowerShell, from a "Developer PowerShell for VS" prompt matching
your installed VS generation, so the MSVC toolchain is on PATH):
    .\build_chrono_fsi.ps1
#>

# ============================================================
# EDIT THESE PATHS FOR YOUR MACHINE
# ============================================================
# Values below are filled in for the "second machine" documented in
# the project notes' "Second machine note" and the development log Entries 50-51 --
# confirmed real paths on that machine, not placeholders. If running on
# a DIFFERENT machine (e.g. the original dev machine, or the college RTX
# 3090 desktop), replace every value below with that machine's own real
# paths -- don't assume these carry over.
$ChronoSourceDir = "C:\Chrono\chrono"              # where you cloned github.com/projectchrono/chrono
$ChronoBuildDir  = "C:\Chrono\chrono-build"          # a SEPARATE build dir, not inside the source tree
$ChronoInstallDir = "C:\Chrono\chrono-install"       # where the INSTALL step stages the built bindings

$IrrlichtRoot = "C:\Tools\irrlicht-1.8.5"         # unzipped Irrlicht SDK 1.8.5 (matches this project's existing version)

# Eigen3: Chrono's own src/CMakeLists.txt calls find_package(Eigen3 5.0),
# falling back to 3.3+ -- there is NO CMake auto-fetch despite an earlier
# (wrong) claim in chrono_fsi_build_guide.md's Architecture note, see
# the development log Entry 51. cmake/FindEigen3.cmake supports a plain
# EIGEN3_INCLUDE_DIR pointed at Eigen's extracted header-only source --
# no Eigen3Config.cmake/install step needed at all. Point this at the
# folder that directly CONTAINS the "Eigen" subfolder.
$Eigen3IncludeDir = "C:\Tools\eigen-5.0.1"

# Point these at the SAME conda env's python.exe you intend to import
# pychrono.fsi from later -- NOT a bare system Python. Match this
# project's own convention of never using a bare `python` (see the project notes'
# "Environment setup" section) -- the built bindings only do anything
# useful if PYTHONPATH + this python.exe end up matched to each other.
$PythonExecutable = "C:\path\to\envs\sloshing\python.exe"
$SwigExecutable   = "C:\Tools\swigwin-4.5.1\swig.exe"  # SWIG 4.0.0+, confirmed working at 4.5.1 (see Entry 50)

# CMake generator string -- MUST match your installed Visual Studio's
# real generator name, confirmed via `cmake --help` (listed under
# "Generators", marked with a name like "Visual Studio <N> <year>") --
# don't guess this from the marketing year alone. Confirmed on the
# second machine (VS Community 2026, 18.9.2): "Visual Studio 18 2026".
# If you're on VS 2022 instead, this would be "Visual Studio 17 2022".
$CMakeGenerator = "Visual Studio 18 2026"

# ============================================================
# Sanity checks before spending an hour on a build that can't succeed
# ============================================================
Write-Host "=== Pre-flight checks ===" -ForegroundColor Cyan

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "FATAL: nvidia-smi not found. This script requires a real NVIDIA GPU with drivers installed." -ForegroundColor Red
    Write-Host "This is exactly the check that failed on the original dev machine (Intel Arc GPU) -- if it fails here too, stop now." -ForegroundColor Red
    exit 1
}
& nvidia-smi | Select-String "NVIDIA-SMI"

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Write-Host "FATAL: nvcc not found. Install CUDA Toolkit 13.3+ (see the development log Entries 51-52 -- 12.9 predates VS 2026 support and crashes with ACCESS_VIOLATION when forced via -allow-unsupported-compiler; 13.3 officially supports MSVC 195x/VS 2026 18.x per NVIDIA's own install guide)." -ForegroundColor Red
    exit 1
}
$cudaVersion = & nvcc --version | Select-String "release"
Write-Host "CUDA: $cudaVersion"
if ($cudaVersion -notmatch "13\.3") {
    Write-Host "WARNING: expected CUDA 13.3 specifically -- found something else. If this is 12.9, expect a CUDA compiler ACCESS_VIOLATION crash on this VS 2026 machine (confirmed, Entry 52) -- upgrade to 13.3 first rather than debugging further." -ForegroundColor Yellow
}

foreach ($path in @($IrrlichtRoot, $Eigen3IncludeDir, $PythonExecutable, $SwigExecutable, $ChronoSourceDir)) {
    if (-not (Test-Path $path)) {
        Write-Host "FATAL: path does not exist, edit the variables at the top of this script: $path" -ForegroundColor Red
        exit 1
    }
}
if (-not (Test-Path (Join-Path $Eigen3IncludeDir "Eigen"))) {
    Write-Host "FATAL: `$Eigen3IncludeDir ($Eigen3IncludeDir) does not directly contain an 'Eigen' subfolder -- point it at the folder that does, not a parent or child of it." -ForegroundColor Red
    exit 1
}

# ============================================================
# Configure
# ============================================================
Write-Host "`n=== Configuring (cmake -S -B) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $ChronoBuildDir | Out-Null

cmake -S $ChronoSourceDir -B $ChronoBuildDir `
    -G "$CMakeGenerator" -A x64 -T "cuda=13.3" `
    -DCMAKE_INSTALL_PREFIX="$ChronoInstallDir" `
    -DCH_ENABLE_MODULE_PYTHON=ON `
    -DCH_ENABLE_MODULE_IRRLICHT=ON `
    -DCH_ENABLE_MODULE_FSI=ON `
    -DCH_ENABLE_MODULE_FSI_TDPF=OFF `
    -DCHRONO_GPU_BACKEND=CUDA `
    -DIrrlicht_ROOT="$IrrlichtRoot" `
    -DEIGEN3_INCLUDE_DIR="$Eigen3IncludeDir" `
    -DPython3_EXECUTABLE="$PythonExecutable" `
    -DSWIG_EXECUTABLE="$SwigExecutable"

if ($LASTEXITCODE -ne 0) {
    Write-Host "cmake configure failed -- fix the reported error before re-running. Common cause: one of the *_DIR/*_ROOT/*_INCLUDE_DIR paths above doesn't actually contain what CMake expects (e.g. EIGEN3_INCLUDE_DIR needs the folder that directly contains an 'Eigen' subfolder, not a parent or child of it)." -ForegroundColor Red
    exit 1
}

# ============================================================
# Build + Install (Release only -- see the build guide's own note on why)
# ============================================================
Write-Host "`n=== Building (this will take a while) ===" -ForegroundColor Cyan
cmake --build $ChronoBuildDir --config Release --target INSTALL -- /m

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed. Check the FIRST error in the log, not the last -- MSBuild often cascades failures from one missing dependency into many downstream module failures." -ForegroundColor Red
    exit 1
}

# ============================================================
# Verify
# ============================================================
# Two real, DIFFERENT failure modes to expect here, both already hit and
# fixed elsewhere in this project's history -- fixed here proactively
# rather than waiting to rediscover them on this machine too:
#
# 1. WRONG BINDINGS PATH: an earlier version of this script hardcoded
#    "$ChronoInstallDir\bin\Release" as the pychrono bindings location --
#    that exact assumption was already proven WRONG on the Linux/Colab
#    build (the real location there was share/chrono/python/, not bin/;
#    see the project notes' Phase 13-14 section). Windows' own CMake install
#    layout for this project is genuinely unverified -- don't trust
#    either guess blindly. Find the REAL location instead:
Write-Host "`n=== Locating the actual pychrono bindings (don't trust a guessed path) ===" -ForegroundColor Cyan
$PyChronoInit = Get-ChildItem -Path $ChronoInstallDir -Recurse -Filter "__init__.py" -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -eq "pychrono" } | Select-Object -First 1
if (-not $PyChronoInit) {
    Write-Host "FATAL: could not find a pychrono/__init__.py anywhere under $ChronoInstallDir -- the install step may not have completed. Check the build log above." -ForegroundColor Red
    exit 1
}
$PyChronoBindingsDir = $PyChronoInit.Directory.Parent.FullName
Write-Host "Found bindings at: $PyChronoBindingsDir"
$env:PYTHONPATH = "$PyChronoBindingsDir;$env:PYTHONPATH"

# 2. MISSING DLL SEARCH PATH: pychrono.irrlicht's compiled extension
#    depends on native DLLs (the Irrlicht engine itself, and likely CUDA
#    runtime DLLs for the FSI/GPU modules) that Python's import system
#    does NOT find via PYTHONPATH alone -- PYTHONPATH only helps Python
#    locate .py/.pyd files to import, not the DLLs those .pyd files then
#    need loaded from the OS's own DLL search path. This is the EXACT
#    bug that crashed main.py on this project's own dev machine
#    (`ImportError: cannot import name '_irrlicht' from 'pychrono'`,
#    the development log Entry 47) -- fixed there by prepending the relevant
#    native library directories to PATH, not PYTHONPATH.
#    CONFIRMED (Entry 54, not guessed): `cmake --install` does NOT copy
#    the raw Irrlicht.dll (the third-party SDK's own engine library)
#    into the install tree at all -- only Chrono's own built DLLs
#    (Chrono_core.dll, Chrono_irrlicht.dll, etc.) land in
#    $ChronoInstallDir\bin. The real Irrlicht.dll only ever exists at
#    its original SDK location, $IrrlichtRoot\bin\Win64-VisualStudio\.
#    Both directories are needed on PATH.
$env:PATH = "$ChronoInstallDir\bin;$IrrlichtRoot\bin\Win64-VisualStudio;$env:PATH"

Write-Host "`n=== Verifying the build actually has FSI ===" -ForegroundColor Cyan
& $PythonExecutable -c "import pychrono.fsi as fsi; print('FSI import OK:', fsi)"
& $PythonExecutable -c "import pychrono.irrlicht; print('Irrlicht import OK')"

Write-Host "`nIf both lines above printed OK, the build is usable. If the FSI import" -ForegroundColor Cyan
Write-Host "fails, re-check CH_ENABLE_MODULE_FSI was actually ON during configure" -ForegroundColor Cyan
Write-Host "(CMake silently leaving a module off due to an unmet dependency is the" -ForegroundColor Cyan
Write-Host "most common failure mode here, not a loud error). If the IRRLICHT import" -ForegroundColor Cyan
Write-Host "specifically fails with 'cannot import name _irrlicht', that's the PATH/" -ForegroundColor Cyan
Write-Host "DLL issue above, NOT a bad build -- find and add the missing DLL directory," -ForegroundColor Cyan
Write-Host "don't rebuild." -ForegroundColor Cyan
