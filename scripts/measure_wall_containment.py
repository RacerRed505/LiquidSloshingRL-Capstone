"""
Real, reusable measurement of the transient wall-penetration issue first
found in the development log Entry 61/64: some fluid markers in a captured
ClosedLoopSPHTank run transiently sit outside the tank's real interior
walls during a rapid angle reversal. Prior measurements (Entries 64/73)
were done in now-deleted scratch scripts -- this is the first committed,
reusable version, written to support a real A/B comparison between
Chrono::FSI boundary methods (the development log Entry 81).

Reuses the SAME world->tank-local rotation convention already proven
correct inside physics/closed_loop_sph_tank.py::ClosedLoopSPHTank.
_rotate_to_tank_frame() (used there for real fluid_com/slosh_energy
computation, not invented fresh here) -- a de-rotation about the pivot
at the sim-frame origin, since the pivot never translates.

Interior wall bounds come directly from config/simulation_config.py::
TankConfig (length/width/wall_thickness/height), the SAME dataclass
ClosedLoopSPHTank.reset() uses to size the real FSI container
(interior_length/interior_width there, matched here exactly) -- not an
independently-guessed geometry.

Run:
    python -m scripts.measure_wall_containment \
        --particles-npz closed_loop_particles.npz \
        --tank-motion-csv closed_loop_output.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math

import numpy as np

from config.simulation_config import TankConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _rotate_to_tank_frame(positions: np.ndarray, tank_angle: float) -> np.ndarray:
    """Vectorized version of ClosedLoopSPHTank._rotate_to_tank_frame --
    same formula, applied to an (N, 3) array of world-frame positions
    instead of a single vector."""
    cos_t, sin_t = math.cos(tank_angle), math.sin(tank_angle)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    return np.column_stack([x * cos_t - z * sin_t, y, x * sin_t + z * cos_t])


def load_tank_angle_by_step(tank_motion_csv: str) -> dict:
    with open(tank_motion_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(row["step"]): float(row["tank_angle"]) for row in rows}


def measure(
    particles_npz: str,
    tank_motion_csv: str,
    snapshot_every_n_steps: int,
    tank_config: TankConfig,
    tolerance_m: float = 1e-6,
) -> list:
    data = np.load(particles_npz, allow_pickle=True)
    times = data["times"]
    raw_positions = data["positions"]
    n_frames = len(times)

    tank_angle_by_step = load_tank_angle_by_step(tank_motion_csv)

    half_length = (tank_config.length - 2 * tank_config.wall_thickness) / 2.0
    half_width = (tank_config.width - 2 * tank_config.wall_thickness) / 2.0

    rows = []
    for i in range(n_frames):
        step = i * snapshot_every_n_steps
        if step not in tank_angle_by_step:
            break  # same truncation convention as render_sph_particle_animation.py
        angle = tank_angle_by_step[step]
        pos = np.asarray(raw_positions[i], dtype=np.float64)
        local = _rotate_to_tank_frame(pos, angle)
        x_local, y_local, z_local = local[:, 0], local[:, 1], local[:, 2]

        viol_x = np.maximum(x_local - half_length, -half_length - x_local)
        viol_y = np.maximum(y_local - half_width, -half_width - y_local)
        viol_floor = -z_local  # positive if below the floor (z_local < 0)
        worst = np.maximum(np.maximum(viol_x, viol_y), viol_floor)

        outside = worst > tolerance_m
        n_outside = int(outside.sum())
        max_depth = float(worst.max()) if len(worst) else 0.0

        rows.append({
            "step": step,
            "time": float(times[i]),
            "tank_angle": angle,
            "n_markers": len(pos),
            "n_outside": n_outside,
            "fraction_outside": n_outside / len(pos) if len(pos) else 0.0,
            "max_penetration_depth_m": max(0.0, max_depth),
        })

    return rows


def summarize(rows: list) -> None:
    if not rows:
        logger.warning("No valid frames to measure.")
        return
    worst_row = max(rows, key=lambda r: r["fraction_outside"])
    deepest_row = max(rows, key=lambda r: r["max_penetration_depth_m"])
    logger.info(
        "Measured %d frames (t=0.00s to %.2fs).", len(rows), rows[-1]["time"],
    )
    logger.info(
        "Worst fraction-outside: %.1f%% at step %d (t=%.2fs, tank_angle=%.3f rad)",
        worst_row["fraction_outside"] * 100, worst_row["step"], worst_row["time"], worst_row["tank_angle"],
    )
    logger.info(
        "Deepest penetration: %.4f m (%.2f cm) at step %d (t=%.2fs)",
        deepest_row["max_penetration_depth_m"], deepest_row["max_penetration_depth_m"] * 100,
        deepest_row["step"], deepest_row["time"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles-npz", required=True)
    parser.add_argument("--tank-motion-csv", required=True)
    parser.add_argument("--snapshot-every-n-steps", type=int, default=5)
    parser.add_argument("--output-csv", default=None, help="Optional: write the full per-frame containment CSV here.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = measure(
        args.particles_npz, args.tank_motion_csv, args.snapshot_every_n_steps, TankConfig(),
    )
    summarize(rows)
    if args.output_csv and rows:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %s", args.output_csv)
