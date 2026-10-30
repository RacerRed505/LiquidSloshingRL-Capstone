"""
Phase 13-14 -- Diff PendulumBackend's predicted fluid response against
SPHBackend's real Chrono::FSI response, for the SAME recorded
tank_motion trajectory.

This is the fluid-model analogue of scripts/compare_trajectories.py
(which compares TankDynamicsModel's surrogate tank against
ChronoTankDynamics's real tank) -- see the development log Entry 41 for why
this trajectory-file-based comparison, rather than a live shared
process, is the right bridge: SPHBackend can only run on a Colab GPU,
not on this project's local machine.

Expects two CSVs:
    --pendulum-csv: written by scripts/record_fluid_trajectory.py
        locally (columns: step, tank_angle, tank_angular_velocity,
        pendulum_fluid_com_x, pendulum_fluid_com_y,
        pendulum_fluid_com_vx, pendulum_fluid_com_vy,
        pendulum_slosh_energy).
    --sph-csv: produced by running SPHBackend against that SAME CSV's
        tank_angle/tank_angular_velocity columns on Colab (columns:
        step, sph_fluid_com_x, sph_fluid_com_y, sph_fluid_com_vx,
        sph_fluid_com_vy, sph_slosh_energy) -- see the project notes' Phase
        13-14 section for the Colab script that produces this.

IMPORTANT caveat, not a bug in this script: PendulumBackend's
com_position is measured from the pendulum's own pivot; SPHBackend's is
reported as a displacement from the fluid's own rest-state COM (see
physics/sph_backend.py::SPHBackend.get_state()'s own docstring) --
these two origins have no reason to coincide, since one is an idealized
point-mass's hanging position and the other is a real fluid centroid's
own rest configuration. Absolute values are therefore NOT expected to
match closely; what matters is whether the two curves' RESPONSE to the
same tank_motion (direction, rough magnitude, timing) agrees -- this
script reports correlation alongside raw differences specifically so
that distinction stays visible, rather than only reporting a raw
mean-absolute-error that would overstate disagreement.

Produces:
    <output_prefix>.csv -- merged per-step comparison
    <output_prefix>.png -- comparison plot (requires matplotlib;
        skipped with a logged warning, CSV still written, if absent)

Run:
    python -m scripts.compare_fluid_backends --pendulum-csv tank_motion_trajectory.csv --sph-csv sph_output.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def merge_by_step(pendulum_rows: List[Dict[str, str]], sph_rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    sph_by_step = {int(row["step"]): row for row in sph_rows}
    merged = []
    missing = 0
    for row in pendulum_rows:
        step = int(row["step"])
        sph_row = sph_by_step.get(step)
        if sph_row is None:
            missing += 1
            continue
        merged.append(
            {
                "step": step,
                "tank_angle": float(row["tank_angle"]),
                "pendulum_fluid_com_x": float(row["pendulum_fluid_com_x"]),
                "pendulum_fluid_com_y": float(row["pendulum_fluid_com_y"]),
                "pendulum_slosh_energy": float(row["pendulum_slosh_energy"]),
                "sph_fluid_com_x": float(sph_row["sph_fluid_com_x"]),
                "sph_fluid_com_y": float(sph_row["sph_fluid_com_y"]),
                "sph_slosh_energy": float(sph_row["sph_slosh_energy"]),
            }
        )
    if missing:
        logger.warning(
            "%d pendulum-CSV steps had no matching SPH-CSV row (SPH run may "
            "have ended early or used a different step count) -- excluded "
            "from the comparison.",
            missing,
        )
    return merged


def _correlation(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / (var_a * var_b) ** 0.5


def summarize(merged: List[Dict[str, float]]) -> None:
    if not merged:
        logger.warning("No overlapping steps -- nothing to summarize.")
        return
    pend_x = [r["pendulum_fluid_com_x"] for r in merged]
    sph_x = [r["sph_fluid_com_x"] for r in merged]
    pend_y = [r["pendulum_fluid_com_y"] for r in merged]
    sph_y = [r["sph_fluid_com_y"] for r in merged]
    pend_e = [r["pendulum_slosh_energy"] for r in merged]
    sph_e = [r["sph_slosh_energy"] for r in merged]

    mae_x = sum(abs(a - b) for a, b in zip(pend_x, sph_x)) / len(merged)
    mae_y = sum(abs(a - b) for a, b in zip(pend_y, sph_y)) / len(merged)
    mae_e = sum(abs(a - b) for a, b in zip(pend_e, sph_e)) / len(merged)

    logger.info(
        "Compared %d steps.\n"
        "  fluid_com_x: MAE=%.5f, correlation=%.3f\n"
        "  fluid_com_y: MAE=%.5f, correlation=%.3f\n"
        "  slosh_energy: MAE=%.6f, correlation=%.3f\n"
        "  (see this script's own module docstring: absolute values are NOT "
        "expected to match -- correlation/response-shape agreement is the "
        "meaningful signal here, not raw MAE alone.)",
        len(merged),
        mae_x, _correlation(pend_x, sph_x),
        mae_y, _correlation(pend_y, sph_y),
        mae_e, _correlation(pend_e, sph_e),
    )


def _write_csv(path: str, merged: List[Dict[str, float]]) -> None:
    if not merged:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)
    logger.info("Wrote merged comparison CSV to %s", path)


def _write_plot(path: str, merged: List[Dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed -- skipping %s (CSV was still written). "
            "Install with: pip install matplotlib",
            path,
        )
        return

    steps = [r["step"] for r in merged]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(steps, [r["pendulum_fluid_com_x"] for r in merged], label="pendulum")
    axes[0].plot(steps, [r["sph_fluid_com_x"] for r in merged], label="sph")
    axes[1].plot(steps, [r["pendulum_fluid_com_y"] for r in merged], label="pendulum")
    axes[1].plot(steps, [r["sph_fluid_com_y"] for r in merged], label="sph")
    axes[2].plot(steps, [r["pendulum_slosh_energy"] for r in merged], label="pendulum")
    axes[2].plot(steps, [r["sph_slosh_energy"] for r in merged], label="sph")

    axes[0].set_ylabel("fluid_com_x (m)")
    axes[1].set_ylabel("fluid_com_y (m)")
    axes[2].set_ylabel("slosh_energy proxy (J)")
    axes[2].set_xlabel("control step")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("PendulumBackend vs. SPHBackend, same tank_motion trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote comparison plot to %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pendulum-csv", required=True, help="CSV from scripts/record_fluid_trajectory.py")
    parser.add_argument("--sph-csv", required=True, help="CSV produced by running SPHBackend on Colab")
    parser.add_argument("--output-prefix", default="fluid_backend_comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pendulum_rows = _read_csv(args.pendulum_csv)
    sph_rows = _read_csv(args.sph_csv)
    merged = merge_by_step(pendulum_rows, sph_rows)
    summarize(merged)
    _write_csv(f"{args.output_prefix}.csv", merged)
    _write_plot(f"{args.output_prefix}.png", merged)
