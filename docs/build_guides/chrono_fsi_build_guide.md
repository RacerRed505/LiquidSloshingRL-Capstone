# Building PyChrono with Chrono::FSI (real SPH fluid physics)

**Status: prerequisites confirmed working for real, on real Windows/CUDA
hardware, as of the development log Entry 50 — the actual `cmake -S -B`
configure/build has not been attempted yet.** This project has (at least)
two development machines now: the *original* one this doc was first written
for (Intel Arc GPU, no CUDA at all — every prerequisite below is genuinely
unrunnable there), and a **second machine** (confirmed 2026-09-05) with a
real NVIDIA GPU (GeForce RTX 5060, Blackwell/`sm_120`) where every
prerequisite in the list below has now actually been installed and verified
working, not just planned. **If you're setting up a new/different machine
for this, follow the "Confirmed working, exact versions" callouts below
first** — they're real installs with real gotchas already paid for, not
guesses. Written on the `feature/sph-fluid-physics` branch, kept separate
from `main` until there's a real, tested backend to show for it.

Sourced from Project Chrono's own current documentation (see "Sources" at
the bottom), not written from memory — the same discipline this project has
applied to every PyChrono API claim so far.

## Why this is a from-scratch build, not a `conda install`

The `sloshing` env's existing `pychrono` package (8.0.0, `projectchrono`
channel) has no `fsi` submodule at all (confirmed earlier this session via
`import pychrono.fsi` → `ModuleNotFoundError`). Chrono::FSI's SPH solver is
GPU-only by design — there is no CPU fallback mode within Chrono::FSI
itself. The public conda-forge/projectchrono binary builds do not ship it
enabled. Getting it means compiling Chrono from source with the FSI module
turned on, on a machine with an actual NVIDIA GPU.

## Hardware/software prerequisites

- **NVIDIA GPU required, no exceptions.** AMD GPUs can use Chrono::FSI via
  a HIP backend instead of CUDA, but that's a separate, less-traveled path
  not considered here since every target machine so far (RTX 3090, and the
  second machine's RTX 5060) is NVIDIA.
- **CUDA Toolkit — 12.9 is the confirmed choice, though Chrono's own current
  docs say "12.8 or newer" with no stated upper bound.** This doc originally
  said "12.9 specifically, 13 not supported yet" from an earlier read of
  Chrono's docs; re-checked live against `api.projectchrono.org` on
  2026-09-05 and the current wording is looser. Stuck with **12.9** anyway
  (matches `build_chrono_fsi.ps1`'s own pre-flight check, no reason to be
  the first to test 13.x here). **Confirmed working, exact version:** CUDA
  Toolkit **12.9.2** (`nvcc` reports `release 12.9, V12.9.86`), installed on
  the second machine's RTX 5060 (Blackwell, `sm_120`) — separately confirmed
  via NVIDIA's own CUDA 12.8 release notes that this architecture is
  supported starting at 12.8, so 12.9 covers it too. **Real gotcha, already
  hit and avoided:** the CUDA installer bundles a display driver by default
  — if your existing driver is newer than what the installer ships (very
  likely on new hardware), do a **Custom** install and **uncheck "Driver
  components" entirely**, or you risk a silent driver downgrade. Confirmed
  on the second machine: driver stayed at `610.88` throughout.
- **C++ compiler, C++17 minimum**: MSVC via Visual Studio 2022 or newer on
  Windows; GCC 11+ or Clang 14+ on Linux. **Confirmed working:** Visual
  Studio **Community 2026** (18.9.2) — a full generation past "2022," and
  CUDA 12.9's Visual-Studio-integration installer still recognized it fine
  (confirmed via its own Repair step reporting "Visual Studio Integration
  ... Installed" after VS was already present). Install the **"Desktop
  development with C++"** workload; nothing else from VS is needed. If
  you're unsure whether a very new VS generation will be recognized, the
  "MSVC v143 - VS 2022 C++ x64/x86 build tools" component is a cheap,
  harmless hedge to check alongside the default toolset — turned out
  unnecessary on the second machine, but cost nothing to have.
- **CMake** (added to `PATH` on Windows). **Confirmed: no separate install
  needed if you already installed VS 2022+'s "Desktop development with
  C++" workload** — it bundles its own "C++ CMake tools for Windows"
  component (confirmed CMake 4.3.1 on the second machine, comfortably above
  whatever Chrono's own `CMakeLists.txt` requires), at a path under the VS
  install tree (e.g. `...\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`)
  rather than on `PATH` by default — add that specific folder to `PATH`
  rather than installing a second, separate CMake.
- **Git** (a GUI client is fine).
- **Eigen3** — the one mandatory dependency for Chrono's core module,
  header-only, no separate install needed if fetched via CMake. **Not yet
  resolved as of the development log Entry 50**: this claim (from Chrono's own
  docs) conflicts with `build_chrono_fsi.ps1` itself, which hardcodes a
  manual `$Eigen3Dir` pointing at a pre-installed `Eigen3Config.cmake` —
  check Chrono's real current CMake configuration before trusting either
  claim, don't guess.
- **SWIG 4.0.0 or newer** — needed for the Python bindings (this project
  needs `CH_ENABLE_MODULE_PYTHON` too, not just FSI, or `main.py`/the
  training stack won't run against this build at all). Tested versions:
  4.0.2, 4.1.0, 4.2.1. **Confirmed working with a newer release too: 4.5.1**
  (current stable as of 2026-09-05), a plain portable zip
  (`swigwin-4.5.1.zip` from SourceForge, no installer) — extract anywhere
  and add that folder to `PATH`. **Real gotcha hit on the second machine:**
  writing directly to `C:\` (the system drive root) failed silently with a
  permission error under a non-admin session — use a folder you actually
  have write access to (e.g. `C:\Tools`) instead.
- **Irrlicht SDK 1.8.5** — needed for `CH_ENABLE_MODULE_IRRLICHT` so
  `main.py`'s visual demo keeps working against this build too. Notably,
  1.8.5 is exactly the version already in use in this project's current
  conda-installed PyChrono (confirmed via this project's own console logs)
  — so this isn't a new/different dependency, just the same one built from
  source instead of pre-packaged. **Confirmed working**, downloaded from
  SourceForge (`irrlicht-1.8.5.zip`, ~21.8 MB) — extract and point
  `Irrlicht_ROOT` at the resulting folder (`bin\Win64-VisualStudio\` /
  `lib\Win64-visualStudio\` are the subfolders that matter on 64-bit
  Windows). **Real gotcha hit and worth repeating for any future
  download of this same file: verify the downloaded size against a HEAD
  request before trusting a "successful" download.** The first attempt on
  the second machine reported `HTTP 200`/exit code 0 from `curl` but was
  silently truncated (13.9 MB instead of the real 21.8 MB) — caught by
  comparing byte counts, confirmed corrupt via `unzip -t`
  (`End-of-central-directory signature not found`), fixed by a clean
  retry. A successful `curl` exit code alone is NOT sufficient evidence a
  download completed correctly.
- **Python 3** (this project uses 3.10 in the `sloshing` env; Chrono's docs
  confirm testing up to 3.12) plus `numpy` and `six` (`pip3 install numpy
  six`) as runtime dependencies for the built bindings. **Not yet resolved
  on the second machine as of Entry 50**: that machine doesn't have this
  project's `sloshing` conda env (or necessarily any conda install at all)
  — needs a real decision (set one up, or use a plain Python 3.10-3.12 with
  `numpy`/`six`) before `build_chrono_fsi.ps1`'s `$PythonExecutable` can be
  set correctly.

## Quick start (once you actually have the GPU machine)

`build_chrono_fsi.ps1` (same folder as this document) scripts steps 1-6
below as a single non-interactive `cmake -S -B` + `cmake --build`
sequence, instead of clicking through `cmake-gui` by hand. Edit the path
variables at its own top (repo clone location, Irrlicht/Eigen/SWIG/Python
locations) to match the actual machine, then run it from a "Developer
PowerShell for VS 2022" prompt. It also runs the same
`import pychrono.fsi` verification from this document automatically at
the end, and pre-flight-checks for `nvidia-smi`/`nvcc`/CUDA 12.9 before
spending an hour on a build that can't possibly succeed. It's a starting
point, not guaranteed to run clean on the first try — real build systems
rarely do; treat its error output as more trustworthy than this guide's
own paraphrasing if the two disagree.

## Alternative: testing the build on Google Colab

Before scheduling time on the college RTX 3090 desktop, the whole build
above can be rehearsed on a free/cheap NVIDIA GPU via Google Colab — see
[`colab_fsi_build.md`](colab_fsi_build.md) (same folder) and its
companion script `build_chrono_fsi_colab.sh`. This is a stand-in for
proving the build plan works, not a replacement for real desktop time —
Colab sessions are time-limited and its GPU model isn't guaranteed
run-to-run, both explained in that document's "Real risks specific to
Colab" section. Same untested-reference status as this document and
`build_chrono_fsi.ps1`.

## Build steps (Windows-focused; Linux notes called out where they differ)

1. **Clone the Chrono repository** (the full source tree, not PyChrono
   alone — PyChrono is generated FROM a from-source Chrono build, not a
   separate project).
2. **Install prerequisites** from the list above. On Windows, make sure
   CMake is on `PATH`. On Linux, install Irrlicht via the system package
   manager (`irrlicht`/`libirrlicht-dev`/`irrlicht-devel` depending on
   distro) instead of downloading the SDK manually.
3. **Configure via `cmake-gui`** (or `cmake` on the command line), pointing
   at the cloned source directory and a **separate** build directory (not
   in-source). Set these CMake variables, then press Configure:
   - `CH_ENABLE_MODULE_PYTHON` = `on`
   - `CH_ENABLE_MODULE_IRRLICHT` = `on`, `Irrlicht_ROOT` = wherever the SDK
     was unzipped (e.g. `C:/Packages/irrlicht-1.8.5`)
   - `CH_ENABLE_MODULE_FSI` = `on` (enables both FSI-SPH and FSI-TDPF by
     default; this project only needs FSI-SPH — `CH_ENABLE_MODULE_FSI_TDPF`
     can be turned off since TDPF additionally requires HDF5, which this
     project has no other use for)
   - `EIGEN3_INCLUDE_DIR` pointed at the folder that directly contains
     Eigen's `Eigen` subfolder — **confirmed real requirement, not
     auto-fetched by CMake at all** (Entry 51): Chrono's own
     `src/CMakeLists.txt` calls `find_package(Eigen3 5.0)`, falling back
     to `3.3`, and prints an ERROR + aborts configure if neither is
     found. Don't bother with `Eigen3_DIR`/`Eigen3Config.cmake` — the
     plain include-dir variable is simpler and doesn't need an install
     step.
   - `SWIG_EXECUTABLE` if not auto-detected (common on Windows)
   - **`Python3_EXECUTABLE`** (confirmed, Entry 51 — NOT the older
     `PYTHON_EXECUTABLE`/`PYTHON_LIBRARY`/`PYTHON_INCLUDE_DIR` trio this
     doc originally said, which is CMake's now-deprecated
     `FindPythonInterp` convention). Chrono's
     `src/chrono_swig/chrono_python/CMakeLists.txt` uses the modern
     `find_package(Python3 REQUIRED COMPONENTS Interpreter Development)`
     — point `Python3_EXECUTABLE` at the **conda env's own `python.exe`**,
     not a system Python, so the built bindings land somewhere importable
     from that env.
4. Press **Configure** again (CUDA should now be auto-detected if the
   Toolkit is on `PATH`; there's a `CHRONO_GPU_BACKEND` variable, default
   `AUTO`, that should resolve to `CUDA` automatically on an NVIDIA
   machine — explicitly set it to `CUDA` if it doesn't auto-resolve rather
   than leaving it ambiguous), then **Generate**.

   **Real gotcha, confirmed on the second machine (Entry 51-52): if CMake
   reports `Looking for a CUDA compiler - NOTFOUND` even with `nvcc`
   working fine from the command line, check whether your Visual Studio
   installation actually HAS the CUDA MSBuild integration files** — the
   CUDA installer's own "Visual Studio Integration" component can report
   success (via its Repair step) without actually placing anything, if
   your VS version postdates what that CUDA release's installer knows
   how to target (confirmed: CUDA 12.9 + VS 2026 hit exactly this — zero
   `CUDA*.props`/`.targets` files existed in ANY of VS's toolset
   `BuildCustomizations` folders, `v150` through `v180`, despite the
   installer's own claim). **Fix, using files NVIDIA already ships inside
   the CUDA Toolkit itself** (`<CUDA install>\extras\visual_studio_integration\MSBuildExtensions\CUDA <version>.{props,targets,xml}`)
   — copy those three files into
   `<VS install>\MSBuild\Microsoft\VC\<toolset>\BuildCustomizations\`
   (find the right `<toolset>` folder by checking which one matches your
   detected MSVC toolset version, e.g. via `cmake --help` / the configure
   log's own `MSVC ...` line). **This needs an elevated (Administrator)
   terminal** — `Program Files` isn't writable otherwise. Also check
   `CUDA_PATH` is actually visible to whatever process is running CMake —
   it's normally set as a system environment variable by the installer,
   but a process started before that variable was set (or before a fresh
   login) won't see it; a brand new terminal window usually resolves this.
5. **Build**: open the generated `Chrono.sln` in Visual Studio, pick
   `Release` (not `Debug` — GPU/SPH work is not something you want to run
   unoptimized), build the whole solution, then build the `INSTALL`
   project specifically to stage everything into the configured install
   location.
6. **Set `PYTHONPATH`** to the generated bindings' output directory (CMake
   prints the exact path during configuration — don't guess it; an earlier
   version of `build_chrono_fsi.ps1` guessed `bin\Release` and that guess
   was already proven wrong on the Linux/Colab build, where the real
   location turned out to be `share/chrono/python/`, not `bin/` — see
   the project notes' Phase 13-14 section. Windows' own install layout for this
   project is still genuinely unverified; `build_chrono_fsi.ps1` now
   searches for the real `pychrono/__init__.py` location automatically
   instead of hardcoding a path — prefer that over guessing by hand).

## Verifying the build actually has FSI (before writing any integration code)

**Before running the import test, also add the install directory's native
library folder(s) to `PATH` — not just `PYTHONPATH`.** This is a second,
DIFFERENT requirement from the bindings path above: `PYTHONPATH` only
helps Python *find* the `.pyd` extension files; it does nothing for the
native DLLs those extensions then need loaded from the OS's own DLL search
path (the actual Irrlicht engine DLL, and likely CUDA runtime DLLs for the
FSI/GPU modules). This is confirmed, not theoretical — it's the exact bug
that crashed `main.py` on this project's own local dev machine
(`ImportError: cannot import name '_irrlicht' from 'pychrono'`,
the development log Entry 47), fixed there by prepending the relevant library
directories to `PATH` before invoking Python, mirroring `the project notes`'s own
"Environment setup" section for this project's local conda env. Locate the
real DLL directory the same "don't guess" way as the bindings path (e.g.
search the install tree for `Irrlicht*.dll`) rather than assuming a path.

```cmd
python -c "import pychrono.fsi; print('FSI import OK:', pychrono.fsi)"
python -c "import pychrono.irrlicht; print('Irrlicht import OK')"
```

Both must succeed. If `pychrono.fsi` fails to import, recheck
`CH_ENABLE_MODULE_FSI` was actually `ON` at Generate time (CMake silently
leaving a module off because a dependency wasn't found is the most common
failure mode here, not a real error) — re-run `cmake-gui`, click Configure,
and check the FSI-related variables are still checked before assuming
anything else is wrong. If instead `pychrono.irrlicht` specifically fails
with `cannot import name '_irrlicht'`, that is almost certainly the
missing-`PATH` issue above, not a bad build — find and add the missing DLL
directory rather than rebuilding.

## What this document deliberately does NOT cover yet

Writing the actual `SPHBackend` implementation against `FluidBackend`
(`physics/sph_backend.py`'s existing stub) — that's real integration work
that should happen only after a working FSI-enabled build is confirmed on
real hardware via the verification step above, not designed blind against
documentation. Two things worth planning for before that work starts,
flagged here so they aren't a surprise later:

- **This is very unlikely to be usable inside RL training at parallel
  speed.** SPH is dramatically more expensive per step than
  `PendulumBackend`'s closed-form pendulum integration — the same reason
  `ChronoTankDynamics` (Phase 10, real ChSystem tank) was validated but
  never wired into `training/train_ppo.py`. Expect `SPHBackend` to be a
  validation/visualization backend (like `scripts/validate_real_dynamics.py`
  already does for the tank), not a training-time replacement for
  `PendulumBackend`, unless real numbers say otherwise.
- **This build is tied to one specific machine.** Unlike this project's
  conda-installed PyChrono (portable via `environment.yml`/conda channels),
  a from-source FSI build is compiled against that machine's specific CUDA
  Toolkit/driver/compiler versions. Moving between the college desktop and
  a temporary cloud GPU instance means rebuilding, not copying files over.

## Real FSI Python API surface, grounded in Chrono's own demo source (not guessed)

Rather than write speculative `SPHBackend` code against the C++ Doxygen
class reference alone (`chrono::fsi::ChSystemFsi`, which turns out not to
be the whole story — see below), the actual demo shipped in Chrono's own
repository was pulled and read directly:
[`src/demos/python/fsi/demo_FSI_ObjectDrop.py`](https://github.com/projectchrono/chrono/blob/main/src/demos/python/fsi/demo_FSI_ObjectDrop.py)
(fetched from the `main` branch — re-check this against whatever specific
commit/release actually gets built, since `main` can move ahead of any
pinned version). **This is still unverified by actually running it** —
reading a demo is not the same as confirming it works on real hardware —
but it's grounded in real, current, working source rather than assumed
from a class reference page.

Key finding: the real Python entry point is a higher-level convenience
wrapper, `pychrono.fsi.ChFsiProblemCartesian`, not the raw `ChSystemFsi`
class the Doxygen page documents directly — and its method names differ
from that page in at least one place already caught (the page's
`DoStepDynamics_FSI()` vs. the wrapper's actual `DoStepDynamics(dt)`).
That mismatch is itself the reason to trust the demo source over the
class reference page when the two disagree.

The demo's actual shape, for reference when `SPHBackend` design starts:

```python
import pychrono as chrono
import pychrono.fsi as fsi

sysMBS = chrono.ChSystemNSC()                       # the ordinary multibody system
problem = fsi.ChFsiProblemCartesian(initial_spacing, sysMBS)
problem.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.8))
problem.SetStepSizeCFD(step_size)
problem.SetStepsizeMBD(step_size)

fluid_props = fsi.FluidProperties()                  # density, viscosity
fluid_props.density = 1000.0
problem.SetCfdSPH(fluid_props)

sph_params = fsi.SPHParameters()                     # integration scheme, EOS, boundary/viscosity method, ...
problem.SetSPHParameters(sph_params)

# A rigid body dropped INTO the fluid, coupled via geometry + AddRigidBody:
body = chrono.ChBody()
# ... SetPos/SetMass/SetInertia/etc., same as this project's own builders/ pattern ...
problem.AddRigidBody(body, geometry, True, True)     # couples it to the SPH particles

problem.Construct(fluid_size, position, fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS)  # builds STATIC domain walls + fluid particles
problem.Initialize()

while ...:
    problem.DoStepDynamics(meta_time_step)           # the actual FSI step
    # sysMBS bodies (e.g. `body` above) now have updated positions from the coupled solve
```

### Two real open design questions this raises for `SPHBackend` — not resolved here

1. **The demo's container is STATIC — our tank rocks.** `Construct()`
   builds fixed domain walls once; it's the right tool for a container
   that never moves. This project's tank is driven by a real motor joint
   and rotates continuously. The likely answer is to make the TANK ITSELF
   an FSI-coupled rigid body (`AddRigidBody`/`AddFsiBody`, the same
   mechanism the demo uses for the object dropped into the fluid) with its
   own geometry matching `TankBuilder`'s box components, rather than using
   `Construct()`'s static-wall convenience helper at all — but this is a
   design hypothesis based on reading one demo, not confirmed by building
   anything. Verify this is actually how a moving container is meant to
   work (Chrono's own docs/forum, or another demo, before assuming) once
   FSI is actually installed and inspectable.

   **New lead, from a second demo — not yet fully read, but stronger
   precedent than the above.** Chrono ships a second, more directly
   relevant FSI demo:
   [`src/demos/fsi/sph/demo_FSI-SPH_WaveTank.cpp`](https://github.com/projectchrono/chrono/blob/main/src/demos/fsi/sph/demo_FSI-SPH_WaveTank.cpp)
   (C++ only — confirmed no Python translation exists in the repo; two
   guessed Python paths 404'd, found the real path by listing
   `src/demos/fsi/sph/` via GitHub's contents API rather than guessing
   further). Unlike `demo_FSI_ObjectDrop.py`'s fully static container,
   this demo builds a wave tank with an *actuated* wall:

   ```cpp
   class WaveFunction : public ChFunction {
       // GetVal(t) returns a prescribed displacement -- same
       // subclass-ChFunction pattern this project's own
       // motor_builder.py already uses for _SineAngleFunction.
   };
   auto fun = chrono_types::make_shared<WaveFunction>(0.25, 0.2, 1);
   auto body = fsi.ConstructWaveTank(ChFsiProblemWavetank::WavemakerType::PISTON,
                                      ChVector3d(0, 0, 0), csize, depth, fun);
   ```

   `ChFsiProblemWavetank` is a distinct high-level problem class from
   `ChFsiProblemCartesian` above, purpose-built for containers with a
   moving wall, with its own `ConstructWaveTank()` convenience method and
   a `WavemakerType` enum (`PISTON` is one option — others unread). The
   wavemaker's motion is driven directly by a custom `ChFunction`, handed
   straight into the FSI construction call. **Why this matters:** it's
   official, working precedent that "a boundary driven by a
   hand-subclassed `ChFunction`, wired in at FSI setup time" is a real,
   supported Chrono pattern — using the exact same technique this
   project's motor already relies on — not something that would need to
   be reverse-engineered from the static-container demo alone.

   **What this does NOT resolve, so don't overclaim it:** the file was
   read only up through the `ConstructWaveTank()` call itself — not read
   yet is what `body` (the object the call returns) actually is, whether
   its state is queryable the way `AddRigidBody`'s coupled bodies are,
   or how `WaveTankRampBeach`/the other `WavemakerType` options factor
   in. It's also a single *translating* piston wall generating waves,
   not a whole-container *rotation* like this project's tank rocking —
   so `ConstructWaveTank` is not a drop-in replacement for `Construct()`,
   just the strongest precedent found so far for "moving boundary via
   `ChFunction`" as a first-class, intended pattern. Confirming the rest
   is future work, not done here.

   **Connection to design question #3 below:** the wavemaker's motion is
   *prescribed* (a `ChFunction` dictates its displacement directly), not
   solved for — which lines up with path **(a)** in question #3 (tank as
   a kinematic puppet inside FSI) far more than path (b). That the
   kinematic-puppet approach is apparently how Chrono's own official
   wave-tank demo handles an actuated boundary is a real point in its
   favor, not just a compromise for preserving this project's existing
   architecture — though whether the wavemaker body still gets pushed
   back on by the fluid (partial two-way coupling despite prescribed
   motion) is exactly the unread part above, so this is a lead, not a
   verified answer.

   **Partial prep done ahead of time**, without touching anything
   FSI-specific: `AddRigidBody()`'s own signature (per the demo) needs a
   `chrono.ChBodyGeometry` built from `chrono.BoxShape` entries — checked
   directly and confirmed **neither class exists in this project's
   currently-installed `pychrono.core`** (they're presumably part of
   whatever module the demo's own dependencies pull in, not shipped in
   the base conda package). So `utilities/fsi_tank_geometry.py` was
   written instead: a PyChrono-free translation layer
   (`tank_box_geometry_for_fsi()`) that reuses
   `utilities/tank_geometry.py::tank_component_boxes()` (the SAME source
   of truth `TankBuilder`'s real `ChBody` already uses) and produces
   plain `FsiBoxGeometry` records — center, full dimensions, base/wall
   flag — in exactly the shape `BoxShape(position, rotation, ChBox(size))`
   will want. Wrapping each record into the real FSI classes once
   available is a mechanical one-line-per-component step, not a redesign.
   Tested (`tests/test_fsi_tank_geometry.py`, 5 tests, no pychrono import
   at all) and confirmed to exactly match `tank_component_boxes()`'s own
   numbers, not transform them. This does NOT resolve the open question
   above (whether `AddRigidBody`/`AddFsiBody` is actually the right
   mechanism for a moving container) — it only removes the geometry-data
   extraction as one less thing to figure out live on the GPU machine.
2. **The demo visualizes via VSG3D (`pychrono.vsg3d`), not Irrlicht.**
   Everything else in this project (`main.py`, all three `--fluid-viz*`
   flags) uses `pychrono.irrlicht`. Whether Irrlicht can render FSI
   particles at all in this version, or whether a real FSI demo requires
   switching to VSG for that specific visualization, is **genuinely
   unknown** — not checked, because there's nothing to check against
   without the module installed. This could mean either building a
   separate VSG-based viewer specifically for `SPHBackend`, or discovering
   Irrlicht handles it fine once actually tried. Don't assume either
   answer before testing on real hardware.

   **Partially de-risked, without any FSI at all.** Confirmed directly
   that `chrono.ChParticleCloud` (a real, already-installed core class —
   not FSI-specific) supports individual per-particle position mutation
   (`cloud.Particle(i).SetPos(...)`, confirmed to write through) and
   renders via `AddVisualShape()` the same way any other body does.
   Built a real prototype: 300 particles, each individually moved every
   frame via a fake synthetic wave (no real fluid physics at all) and
   rendered through the SAME `ChVisualSystemIrrlicht` this project's
   `main.py` already uses — **measured 668 Hz with a real stopwatch**, no
   crash. This confirms the underlying *rendering mechanism* (a moving
   particle cloud, rendered via Irrlicht) works fine on this hardware.
   **What this does NOT confirm**: whether Chrono::FSI's own internal SPH
   particle representation (once installed) can actually be exposed
   through `ChParticleCloud`, or through `GetParticlePositions()`/
   `GetParticleVelocities()` fed into a `ChParticleCloud` built
   separately, or whether FSI keeps its particle data in a GPU-resident
   form that needs real plumbing to bridge into anything Irrlicht can
   see. Narrowed from "completely unknown" to "the rendering technology
   itself works; the FSI-specific data bridge is still unverified."

   **Further resolved via research (not guessed): Chrono::FSI has no
   first-party Irrlicht visualization plugin at all.** `ChFsiVisualization`
   (FSI's own visualization base class) has exactly ONE documented
   implementation across Chrono's API reference —
   [`ChFsiVisualizationVSG`](https://api.projectchrono.org/classchrono_1_1fsi_1_1_ch_fsi_visualization.html).
   No Irrlicht equivalent exists to plug into. This means the choice
   isn't "does Irrlicht happen to support FSI automatically" (it
   doesn't, confirmed) — it's "switch to VSG for this specific
   visualization" vs. "manually bridge FSI's `GetParticlePositions()`/
   `GetParticleVelocities()` into a separately-built `ChParticleCloud`,
   rendered via Irrlicht the same way the prototype above already
   confirmed works." The manual-bridge path is the one consistent with
   keeping `main.py` on Irrlicht throughout — real engineering work
   (get FSI's particle arrays out each step, feed positions into the
   cloud), but not blocked by anything found so far.

3. **A real architectural mismatch, not yet resolved: `FluidBackend`'s
   contract assumes one-way coupling; Chrono::FSI's `AddRigidBody()`
   pattern is two-way.** `FluidBackend.step(dt, tank_motion)` (see
   `physics/backend_interface.py`) hands the fluid backend the tank's
   *already-computed* motion as a read-only forcing input each step —
   this fits `PendulumBackend` perfectly (a pure ODE integrator that
   never affects the tank) and is why the tank and fluid are stepped as
   two independent objects throughout this project. Chrono::FSI's own
   demo pattern doesn't work that way: the rigid body added via
   `AddRigidBody()` lives *inside* the same `sysMBS` the FSI problem
   owns, and `problem.DoStepDynamics(dt)` advances the fluid particles
   and the coupled rigid body **together, with the fluid genuinely
   pushing back on the body** — that mutual feedback is the entire point
   of "fluid-structure interaction," not an optional extra.

   This means naively wiring the tank into FSI as `AddRigidBody(tank_body,
   ...)` would NOT fit today's `step(dt, tank_motion)` signature at all —
   there'd be no separate "tank motion" to hand in, because the tank's
   motion would now be an *output* of the same coupled step, not an
   input to it. Two real paths, neither exercised, both worth stating
   plainly rather than picking blind:
   - **(a) Keep the tank as a kinematic puppet inside FSI**, driven each
     step to match whatever `TankDynamicsModel`/`ChronoTankDynamics`
     already computed independently — preserves today's one-way-coupling
     architecture and `FluidBackend`'s existing contract untouched, but
     means "real" FSI would still behave exactly as one-way as the
     current decorative ripple as far as the RL environment is
     concerned. The fluid would never actually feel the tank pushing on
     it any differently than `PendulumBackend` already approximates,
     which raises a fair question of how much genuine fidelity gain this
     buys over the existing surrogate, for a much larger engineering
     cost.
   - **(b) Let FSI actually own the tank, two-way for real** — the
     physically meaningful option, but breaks the "swap the backend
     without touching `environment/`" promise `backend_interface.py`'s
     own docstring makes today, since the tank and fluid could no longer
     be stepped as two independent objects behind separate interfaces.
     Would need a real redesign of how `sloshing_env.py` orchestrates
     tank + fluid stepping, not a drop-in `SPHBackend`.

   **Not resolved here** — flagged so this decision gets made
   deliberately once FSI is actually installed and inspectable, not
   discovered as a surprise mid-implementation.

## Sources

- [Project Chrono: Install the FSI module](https://api.projectchrono.org/module_fsi_installation.html)
- [Project Chrono: Install Chrono](https://api.projectchrono.org/tutorial_install_chrono.html)
- [Project Chrono: Install the PYTHON module](https://api.projectchrono.org/module_python_installation.html)
- [Project Chrono: Install the IRRLICHT module](https://api.projectchrono.org/module_irrlicht_installation.html)
- [Project Chrono: `chrono::fsi::ChSystemFsi` class reference](https://api.projectchrono.org/classchrono_1_1fsi_1_1_ch_system_fsi.html) (the raw C++ API — kept for reference, but see above for where the real Python demo's actual usage differs from this page)
- [`demo_FSI_ObjectDrop.py`](https://github.com/projectchrono/chrono/blob/main/src/demos/python/fsi/demo_FSI_ObjectDrop.py), Chrono's own real, current Python FSI demo (`main` branch) — the actual source the API sketch above was read from, not paraphrased from memory
