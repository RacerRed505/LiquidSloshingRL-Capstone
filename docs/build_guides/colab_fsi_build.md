# Building PyChrono with Chrono::FSI on Google Colab

**Status: reference document only — not yet executed anywhere, same status
as `chrono_fsi_build_guide.md` and `build_chrono_fsi.ps1`.** This adapts
that guide's Linux build path to Google Colab specifically, as a
**temporary stand-in for the RTX 3090 desktop** when it isn't available —
not a replacement for it. Written on `feature/sph-fluid-physics`, kept off
`main` until there's a real, tested backend to show for it.

This is genuinely untested. Treat every command below the way this
project's own convention already treats `build_chrono_fsi.ps1`: a
starting point grounded in real research (Chrono's own docs/demos and
this project's own already-verified prerequisite list), not a guarantee
of a clean first run. Colab in particular has failure modes the Windows
guide doesn't need to worry about — see "Real risks specific to Colab"
below before spending an hour on this.

## Why Colab, and why it's a stand-in, not a replacement

Colab gives free (or cheap, on Pro tiers) access to a real NVIDIA GPU —
exactly what this dev machine's Intel Arc GPU cannot provide, and exactly
what `chrono_fsi_build_guide.md` says is required with no exceptions.
That makes it a legitimate way to test **whether this project's build
plan actually works at all** before relying on scheduled time on the
college RTX 3090 desktop.

It is not a good place to *live* — see the session-limit risks below —
so the intended use is: prove the build succeeds and `import pychrono.fsi`
works here first, so that whenever real desktop time is available, it's
spent running an already-debugged process instead of discovering problems
live.

## Real risks specific to Colab — read before starting

1. **Sessions disconnect.** Free-tier Colab disconnects after roughly 90
   minutes of inactivity and has a hard ceiling around 12 hours even with
   activity. A full Chrono build (FSI + Python + Irrlicht modules, C++17,
   on Colab's typically 2-core CPU allocation) can plausibly take over an
   hour — there is a real chance of losing an in-progress build mid-way.
2. **Local storage is ephemeral.** Anything written to `/content` (the
   default working directory) is wiped when the session ends. **Mount
   Google Drive and do all real work under
   `/content/drive/MyDrive/chrono-fsi-build/` instead of `/content`
   directly** (Step 2 below) so a disconnect doesn't erase hours of
   compiled object files — CMake/Make can resume an interrupted build from
   persisted object files, but only if they still exist somewhere.
3. **The exact CUDA Toolkit version isn't known ahead of time.** Colab
   pre-installs a CUDA Toolkit version that changes over time as Google
   updates the runtime image. `chrono_fsi_build_guide.md` requires
   **CUDA 12.9 specifically** (12.9, not 13.x). Step 1 below checks this
   directly — if Colab's preinstalled version doesn't match, a separate
   CUDA Toolkit install step is needed before continuing (not written
   here, since whether it's even needed depends on what Step 1 actually
   reports — don't preemptively install a second CUDA Toolkit if the
   first one already matches).
4. **The GPU model varies.** Free tier is usually a T4; Colab Pro/Pro+ can
   assign better GPUs but neither is guaranteed run-to-run. Any NVIDIA GPU
   satisfies Chrono::FSI's "NVIDIA GPU required" prerequisite, so this
   doesn't block the build — it just means performance numbers from a
   Colab run are **not comparable** to whatever the RTX 3090 eventually
   produces, and shouldn't be reported as if they were.
5. **The Linux package list below is a best-effort port of the Windows
   guide's prerequisites, not independently verified against Colab's
   actual Ubuntu image.** Irrlicht in particular has real Linux windowing
   dependencies (X11/Mesa libraries) that the Windows SDK path doesn't
   surface — if the build fails specifically inside the Irrlicht module,
   check for a missing `libx11-dev`/`libgl1-mesa-dev`-style package before
   assuming something more fundamental is wrong.
6. **`google.colab.drive.mount()` may not work at all, and neither may
   the API-based alternative — confirmed the hard way, not assumed.** In
   one real session, `drive.mount()` failed repeatedly with `MessageError:
   credential propagation was unsuccessful`, surviving a runtime restart
   and standard browser troubleshooting (extensions, third-party cookies,
   incognito). A structurally different mechanism,
   `google.colab.auth.authenticate_user()` (used for Drive-API access
   instead of a filesystem mount), failed with the **identical** error —
   strong evidence the failure is in Colab's underlying browser-to-kernel
   message channel for that specific account/browser/network
   combination, not in Drive's permissions or which API is called. If you
   hit this, more retries of `drive.mount()` itself are unlikely to help;
   see "Skipping Google Drive entirely" below for the approach that
   sidesteps it completely.

## Step 0 — Start the notebook with a GPU runtime

In Colab: **Runtime → Change runtime type → Hardware accelerator → GPU**
(pick the best tier available to you), then **Save**. Do this *before*
running anything else, or the notebook will boot without a GPU attached
at all.

## Step 1 — Verify the GPU and CUDA version (run this first, every session)

```python
!nvidia-smi
!nvcc --version
```

Read the `nvcc --version` output for the `release` line. If it says
`release 12.9`, continue as-is. If it says anything else (very possible —
Colab's image changes over time), **stop and reconcile the version before
building** — either look for a matching CUDA 12.9 install method for
Colab specifically, or check `chrono_fsi_build_guide.md`'s sources for
whether a newer/older CUDA version has since become supported by
Chrono::FSI. Do not just proceed and hope; version mismatches here are
exactly the kind of failure that surfaces confusingly deep inside the
build log instead of clearly at the start.

## Step 2 — Skipping Google Drive entirely (current recommended path)

Given risk #6 above, this workflow defaults to **not** using
`drive.mount()` at all — everything builds in plain `/content`. The
trade-off (a session disconnect mid-build loses everything, with no
resume) is real, but is handled differently: instead of a live Drive
mount, `build_chrono_fsi_colab_stage2.sh` automatically zips the
finished build at the end, and both build scripts print a manual
checkpoint snippet using `google.colab.files.download()` — a plain
browser file download that needs **no Google authentication at all**,
since it doesn't go through the same message channel that
`drive.mount()`/`authenticate_user()` failed on:

```python
import os
os.system('cd /content && zip -r -q chrono-fsi-build-checkpoint.zip chrono-fsi-build')
from google.colab import files
files.download('/content/chrono-fsi-build-checkpoint.zip')
```

This downloads straight to your laptop's Downloads folder. To resume in
a fresh session after a disconnect, re-upload that same zip (same
drag-and-drop or upload-widget approach as uploading the scripts
themselves) and extract it before re-running the build:

```python
from google.colab import files
uploaded = files.upload()  # pick the checkpoint zip from your Downloads folder
!unzip -q -o chrono-fsi-build-checkpoint.zip -d /content
```

Because zip preserves file modification times, CMake/Make should
recognize already-compiled object files as up to date and resume rather
than recompile everything — this is unverified against a real multi-hour
build's full object count, but is the mechanism the two-stage scripts
below are built around.

**If you'd rather try Drive again anyway** (e.g. on a different
network/browser where risk #6 doesn't reproduce): mount it the standard
way, then just point `CHRONO_ROOT` in both build scripts at
`/content/drive/MyDrive/chrono-fsi-build` instead of `/content/chrono-fsi-build`
before running them — no other changes needed, since both scripts read
that variable throughout rather than hardcoding `/content` in multiple
places.

## Step 3 — Install Linux prerequisites

```bash
!apt-get update -qq
!apt-get install -y -qq \
    build-essential cmake git swig \
    libeigen3-dev \
    libirrlicht-dev \
    libx11-dev libxrandr-dev libxxf86vm-dev libgl1-mesa-dev
!pip install -q six
```

(`numpy` is already present in the standard Colab image.) Per
`chrono_fsi_build_guide.md`'s own Linux note, this installs Irrlicht via
the system package manager rather than the Windows SDK download — if this
resolves to an Irrlicht version other than 1.8.5, note the actual version
installed; it's called out in the Windows guide as a nice-to-have version
match, not a hard requirement.

## Step 4 — Run the build scripts (two stages, two separate cells)

Split into two scripts specifically so there's a real pause point to
checkpoint at — Colab can't run a second cell while a `!bash script.sh`
cell is still executing in the same kernel, so a single all-in-one
script gives you no safe place to interrupt and save progress without
killing the whole thing.

**Stage 1** (pre-flight checks, prerequisites, clone, configure) — upload
`build_chrono_fsi_colab.sh` (drag-and-drop into the file browser panel),
then:
```bash
!bash build_chrono_fsi_colab.sh
```
This ends with a real checkpoint: the cell returns control, and you can
optionally run the download snippet from Step 2 above before continuing.

**Stage 2** (the actual expensive build + install + verify) — upload
`build_chrono_fsi_colab_stage2.sh` the same way, then:
```bash
!bash build_chrono_fsi_colab_stage2.sh
```
This automatically zips the finished build at the end and prints the
`files.download()` snippet to run next.

Read both scripts' own header comments before running — the path
variables at their top may need adjusting if this guide's directory
layout wasn't followed exactly, and both must agree with each other
(stage 2 expects stage 1's exact `CHRONO_ROOT`).

**Stage 2 is the one most likely to need real debugging on first run.**
Read the actual error Colab prints, not this guide's description of what
should happen — this document and both scripts are grounded in research,
not a confirmed working run.

## Step 5 — Verify (repeat this in any NEW session, after reconnecting)

Because a fresh Colab session doesn't remember the prior session's
`PYTHONPATH`, re-run this in any new session that reconnects to an
already-built install (after restoring a checkpoint zip, per Step 2 —
skip Stages 1-2 entirely if the install already exists and just needs
re-verifying):

```python
import sys
sys.path.insert(0, '/content/chrono-fsi-build/chrono-install/bin')  # adjust to match wherever CHRONO_ROOT actually pointed

import pychrono.fsi as fsi
print("FSI import OK:", fsi)
import pychrono.irrlicht
print("Irrlicht import OK")
```

Both must succeed. If they do, this is the point where
`chrono_fsi_build_guide.md`'s "Real FSI Python API surface" section and
its two open design questions stop being unverified research and become
things that can actually be tried against a live `pychrono.fsi` — but per
this project's own "don't design blind" rule, that integration work
(`physics/sph_backend.py`) should still happen deliberately, not rushed
into the same session that just got the import working.

## What this does NOT cover

- Installing this project's own dependencies (`gymnasium`,
  `stable-baselines3`, this repo's own source) into the Colab runtime —
  that's a separate step, only needed once `SPHBackend` integration work
  actually starts, not for proving the build alone works.
- Running this project's existing test suite against a Colab-built
  PyChrono — the existing conda-env-based test setup
  (`tests/test_chrono_tank_dynamics.py`'s `pytest.importorskip` guard,
  etc.) assumes the Windows conda environment described in `the project notes`,
  not a Colab runtime. Out of scope until there's real `SPHBackend` code
  to test.
