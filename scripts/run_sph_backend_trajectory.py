"""
Phase 13-14 -- Drive physics/sph_backend.py::SPHBackend over a recorded
tank_motion trajectory, producing the --sph-csv scripts/compare_fluid_backends.py
expects.

This is the piece that never existed as a committed local script: every
prior SPHBackend comparison (the development log Entries 43-46) drove the
backend via ad hoc Colab notebook cells, since SPHBackend could only run
there (no local CUDA GPU). Now that a real local Chrono::FSI build exists
(the development log Entries 50-54), this script closes that gap -- the same
comparison can run entirely on this machine, no Colab/Drive bridge needed.

Reads a CSV written by scripts/record_fluid_trajectory.py (columns
step, tank_angle, tank_angular_velocity, ...) and steps SPHBackend with
each row's (tank_angle, tank_angular_velocity) at the given control_dt,
recording SPHBackend's own get_state() each step.

Interruption/resume (added 2026-09-18, ahead of first access to a third
real machine -- see the project notes' "RTX 3090 machine note" -- some of this
project's own remaining SPH work is estimated in HOURS, e.g. a full
dx=0.003 resolution run, so a workstation closing/restarting mid-run is
a real risk, not hypothetical): two honest, real protections, NOT a
claim that the SPH particle physics itself can resume mid-flight --
`SPHBackend`/Chrono::FSI expose no API to serialize/restore the full
particle state (positions+velocities for thousands of markers), and
this file does not pretend otherwise.
  1. **Skip already-fully-completed runs.** If `--output` already exists
     with exactly as many rows as the input trajectory, this run is a
     no-op (logged, not silent) -- re-launching after a machine restart
     never redoes work that's already fully on disk. Pass --force to
     override.
  2. **Save progress every --save-every-n-steps steps, not just at the
     end.** If interrupted partway, at most --save-every-n-steps of
     compute is lost, not the whole run -- but the run still has to be
     RESTARTED FROM STEP 0 to produce a complete file (the physics
     itself cannot resume from row N; only the file-level "already done
     or not" check above avoids wasted GPU time on top of that).

Run:
    python -m scripts.run_sph_backend_trajectory tank_motion_local_seed0.csv --output sph_output_local_seed0.csv

    # Also capture raw particle positions for a visual render (e.g. via
    # scripts/render_publication_quality.py) -- off by default, opt in:
    python -m scripts.run_sph_backend_trajectory tank_motion_local_seed0.csv \\
        --output sph_output_local_seed0.csv --particles-npz sph_particles_local_seed0.npz
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from typing import List

import numpy as np

from physics.sph_backend import SPHBackend
from scripts._gpu_info import log_gpu_info

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _read_tank_motion(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _already_complete(output_path: str, expected_rows: int) -> bool:
    """True only if output_path exists and has exactly expected_rows data
    rows -- a partial file from an interrupted run does NOT count (it
    must be redone from scratch; see module docstring)."""
    if not os.path.exists(output_path):
        return False
    with open(output_path, newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1  # minus the header row
    return n == expected_rows


def _expected_particle_frames(n_rows: int, snapshot_every_n_steps: int) -> int:
    """Frame 0 is the initial state (0 steps taken); a new frame is then
    captured every snapshot_every_n_steps control steps, including the
    final step if it lands on the interval -- matches the frame-i-means-
    step-(i*snapshot_every_n_steps) convention render_sph_particle_animation.py
    and render_publication_quality.py already assume when reading a
    particles .npz (confirmed by reading their own real frame-to-step
    lookup, not guessed)."""
    return n_rows // snapshot_every_n_steps + 1


def _particles_npz_already_complete(particles_npz: str, expected_frames: int) -> bool:
    if not particles_npz or not os.path.exists(particles_npz):
        return False
    data = np.load(particles_npz, allow_pickle=True)
    return len(data["times"]) == expected_frames


def write_particles_npz(path: str, times: List[float], positions: List[np.ndarray]) -> None:
    # positions is a ragged-safe object array (particle count is constant
    # per run here, but object dtype matches the format the existing
    # committed captures -- e.g. sph_particles_trained_seed0.npz -- already
    # use, and what the render scripts already know how to load).
    np.savez(path, times=np.array(times), positions=np.array(positions, dtype=object))


def run(
    tank_motion_path: str,
    control_dt: float,
    log_every: int,
    initial_spacing: float = 0.01,
    step_size: float = 1.0e-4,
    max_velocity: float = 4.5,
    viscosity: float = 1.0,
    num_bce_layers: int = 4,
    output_path: str = "sph_output_local.csv",
    save_every_n_steps: int = 25,
    force: bool = False,
    particles_npz: str = None,
    snapshot_every_n_steps: int = 5,
) -> List[dict]:
    rows = _read_tank_motion(tank_motion_path)
    if not rows:
        raise ValueError(f"No rows found in {tank_motion_path}")

    expected_frames = _expected_particle_frames(len(rows), snapshot_every_n_steps)
    csv_done = _already_complete(output_path, len(rows))
    npz_done = (not particles_npz) or _particles_npz_already_complete(particles_npz, expected_frames)
    if not force and csv_done and npz_done:
        logger.info(
            "%s already has all %d rows%s -- skipping (this run is already fully done, "
            "not redoing it after a restart). Pass force=True / --force to override.",
            output_path, len(rows),
            f" and {particles_npz} already has all {expected_frames} frames" if particles_npz else "",
        )
        with open(output_path, newline="") as f:
            return list(csv.DictReader(f))

    log_gpu_info(logger)
    backend = SPHBackend(
        initial_spacing=initial_spacing, step_size=step_size,
        max_velocity=max_velocity, viscosity=viscosity,
        num_bce_layers=num_bce_layers,
    )
    output_rows = []
    particle_times: List[float] = []
    particle_frames: List[np.ndarray] = []
    if particles_npz:
        # Frame 0 is the pre-stepping initial state -- same convention the
        # already-committed captures (e.g. sph_particles_trained_seed0.npz)
        # use, confirmed by inspecting that file directly (times[0] == 0.0).
        particle_times.append(0.0)
        particle_frames.append(backend.get_fluid_particle_positions())
    t_start = time.perf_counter()
    for i, row in enumerate(rows):
        tank_angle = float(row["tank_angle"])
        tank_angular_velocity = float(row["tank_angular_velocity"])
        backend.step(control_dt, (tank_angle, tank_angular_velocity))
        state = backend.get_state()
        output_rows.append(
            {
                "step": i,
                "sph_fluid_com_x": state.com_position[0],
                "sph_fluid_com_y": state.com_position[1],
                "sph_fluid_com_vx": state.com_velocity[0],
                "sph_fluid_com_vy": state.com_velocity[1],
                "sph_slosh_energy": state.slosh_energy,
            }
        )
        if particles_npz and (i + 1) % snapshot_every_n_steps == 0:
            particle_times.append((i + 1) * control_dt)
            particle_frames.append(backend.get_fluid_particle_positions())
        if (i + 1) % save_every_n_steps == 0 or (i + 1) == len(rows):
            write_csv(output_path, output_rows)
            if particles_npz:
                write_particles_npz(particles_npz, particle_times, particle_frames)
        if (i + 1) % log_every == 0 or (i + 1) == len(rows):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (len(rows) - (i + 1)) / rate if rate > 0 else float("inf")
            logger.info(
                "step %d/%d  com=(%.5f, %.5f)  slosh_energy=%.5f  elapsed=%.1fs  eta=%.1fs  "
                "(saved to %s)",
                i + 1, len(rows), state.com_position[0], state.com_position[1],
                state.slosh_energy, elapsed, eta, output_path,
            )
    total_elapsed = time.perf_counter() - t_start
    sim_time = len(rows) * control_dt
    logger.info(
        "Done: %d steps in %.1fs real time (%.1fs simulated, %.2fx slower than real-time). "
        "Wrote %s%s.",
        len(rows), total_elapsed, sim_time, total_elapsed / sim_time, output_path,
        f" and {particles_npz} ({len(particle_frames)} frames)" if particles_npz else "",
    )
    return output_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tank_motion_csv", help="CSV written by scripts/record_fluid_trajectory.py")
    parser.add_argument("--output", default="sph_output_local.csv")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Must match the recorded trajectory's own control_dt (EnvironmentConfig default: 0.02s).")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every-n-steps", type=int, default=25, help="Write --output to disk this often, not just once at the end, so a crash/interruption loses at most this many steps' worth of GPU compute (see module docstring's honest resume/interruption note).")
    parser.add_argument("--force", action="store_true", help="Re-run even if --output already has all the expected rows (by default, an already-fully-completed run is skipped, not redone).")
    parser.add_argument("--initial-spacing", type=float, default=0.01, help="SPH particle spacing (m). Default (0.01, ~1,846 particles) matches every historical the development log comparison. An external CFD-advisor review (SPH_Enhancement_Guide_18Sept2026.zip, 2026-09-18) flagged this as too coarse -- gives a 'pearl'/bead look in scripts/render_sph_particle_animation.py -- and recommended 0.003 (~68,000 particles). Exposed here for a real GPU run to test that; not applied as a silent default change.")
    parser.add_argument("--step-size", type=float, default=1.0e-4, help="SPHBackend's own internal substep size (s). Default matches SPHBackend's own default.")
    parser.add_argument("--max-velocity", type=float, default=4.5, help="SPHParameters.max_velocity (sets the weakly-compressible EOS sound speed, c_s = 10x this). Same 2026-09-18 CFD-advisor review recommended 2.0 for this tank's real peak fluid velocity -- untested locally (no GPU), exposed for a controlled comparison.")
    parser.add_argument("--viscosity", type=float, default=1.0, help="FluidProperties.viscosity (Pa*s, dynamic). Default (1.0) is a real, longstanding placeholder -- 1000x real water's ~0.001 Pa*s at 20C -- flagged by the same review. Every historical comparison used this default, so it is not silently changed; pass --viscosity 0.001 for a real physical-water comparison.")
    parser.add_argument("--num-bce-layers", type=int, default=4, help="Number of ADAMI boundary-marker layers. Default (4) matches every historical comparison. The boundary layer's real physical thickness is num_bce_layers * --initial-spacing -- at a finer --initial-spacing, the same 4 layers give a THINNER real boundary buffer unless scaled up to compensate (the development log Entry 96 found a real dx=0.005 SPH solver crash that a smaller --step-size alone did not fix, pointing at this).")
    parser.add_argument("--particles-npz", default=None, help="If given, also captures raw per-particle positions (physics/sph_backend.py::SPHBackend.get_fluid_particle_positions()) every --snapshot-every-n-steps steps, written in the same times/positions .npz format scripts/render_publication_quality.py and scripts/render_sph_particle_animation.py already expect (confirmed by inspecting both the format those scripts read and an already-committed real capture, sph_particles_trained_seed0.npz, before writing this). Off by default -- every existing call site (the resolution-convergence comparison runs) only ever needed the summary CSV; this is for producing the actual visual renders per resolution.")
    parser.add_argument("--snapshot-every-n-steps", type=int, default=5, help="How often to capture a particle-position frame when --particles-npz is given. Default (5) matches the existing committed captures (e.g. sph_particles_trained_seed0.npz: 100 frames over a 500-step/10s run).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.tank_motion_csv, args.control_dt, args.log_every,
        args.initial_spacing, args.step_size, args.max_velocity, args.viscosity,
        args.num_bce_layers,
        args.output, args.save_every_n_steps, args.force,
        args.particles_npz, args.snapshot_every_n_steps,
    )
