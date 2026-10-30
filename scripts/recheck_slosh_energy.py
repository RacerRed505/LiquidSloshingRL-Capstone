"""
Offline re-check of the "slosh energy" numbers already saved in this repo's
CSVs, in response to the second-review question "convective/sloshing energy:
check if valid". No GPU and no PyChrono needed: it only reads existing CSVs.

What it checks:

1. Pendulum (Housner convective-mass) trajectories -- tank_motion_*.csv.
   Every saved `pendulum_slosh_energy` value is KINETIC energy only
   (0.5*m_c*v^2). The convective mode's full mechanical energy also has a
   gravitational potential term, m_c*g*L*(1 - cos theta). The pendulum's
   saved COM height is y = -L*cos(theta), so that term can be recovered
   exactly from the CSV as m_c*g*(y + L). This script adds it and reports
   kinetic vs. potential vs. total for each file.

2. SPH trajectories -- sph_output_*.csv.
   SPH saves the same kind of kinetic-only bulk-COM value. The obvious
   potential term, M*g*dy from the COM's height change, turns out NOT to be
   a valid sloshing energy here: the SPH fluid's COM drifts steadily
   downward as the weakly compressible fluid settles and as markers leak
   past the ADAMI walls (the development log Entry 64). This script measures
   that drift and compares M*g*drift with the real slosh kinetic energy,
   which is the evidence for leaving SPH's potential term unreported
   (FluidState.slosh_potential_energy = None) rather than reporting a
   misleading number. The fluid mass M is recovered from each file itself
   (saved energy / 0.5*|v|^2), so no marker count is needed.

Outputs slosh_energy_recheck.csv (one row per input file) and a log summary.

    python -m scripts.recheck_slosh_energy
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
from typing import Dict, List, Optional

from config.simulation_config import FluidConfig, TankConfig, WorldConfig
from utilities.housner_utils import compute_housner_parameters

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pendulum_energy_split(rows: List[Dict[str, str]], mass: float, length: float, gravity: float) -> Dict[str, List[float]]:
    """Kinetic (as saved), potential (recovered from COM height) and total
    energy per step for a PendulumBackend trajectory."""
    kinetic = [float(r["pendulum_slosh_energy"]) for r in rows]
    # y = -L*cos(theta)  =>  m*g*L*(1 - cos theta) = m*g*(y + L)
    potential = [mass * gravity * (float(r["pendulum_fluid_com_y"]) + length) for r in rows]
    total = [k + p for k, p in zip(kinetic, potential)]
    return {"kinetic": kinetic, "potential": potential, "total": total}


def sph_fluid_mass(rows: List[Dict[str, str]]) -> Optional[float]:
    """Recover the fluid mass SPH used from energy = 0.5*M*|v|^2."""
    for r in rows:
        v2 = float(r["sph_fluid_com_vx"]) ** 2 + float(r["sph_fluid_com_vy"]) ** 2
        if v2 > 1e-10:
            return float(r["sph_slosh_energy"]) / (0.5 * v2)
    return None


def recheck_pendulum_file(path: str, mass: float, length: float, gravity: float) -> Optional[Dict[str, object]]:
    rows = _read_rows(path)
    if not rows or "pendulum_fluid_com_y" not in rows[0]:
        return None
    e = pendulum_energy_split(rows, mass, length, gravity)
    peak_total = max(e["total"])
    return {
        "file": os.path.basename(path),
        "model": "pendulum",
        "steps": len(rows),
        "peak_kinetic_j": max(e["kinetic"]),
        "peak_potential_j": max(e["potential"]),
        "peak_total_j": peak_total,
        "final_kinetic_j": e["kinetic"][-1],
        "final_total_j": e["total"][-1],
        # Share of the mode's energy the saved kinetic-only number misses,
        # averaged over the run (0 = nothing missed).
        "mean_fraction_missed": (
            sum(p for p in e["potential"]) / sum(e["total"]) if sum(e["total"]) > 0 else 0.0
        ),
        "com_height_drift_mm": "",
        "drift_pe_over_peak_ke": "",
        "verdict": "potential term added (kinetic-only value was incomplete)",
    }


def recheck_sph_file(path: str, gravity: float) -> Optional[Dict[str, object]]:
    rows = _read_rows(path)
    if not rows or "sph_fluid_com_y" not in rows[0]:
        return None
    mass = sph_fluid_mass(rows)
    kinetic = [float(r["sph_slosh_energy"]) for r in rows]
    heights = [float(r["sph_fluid_com_y"]) for r in rows]
    # Largest excursion of the COM height from its first saved value.
    drift = max((h - heights[0] for h in heights), key=abs)
    drift_pe = abs(mass * gravity * drift) if mass is not None else float("nan")
    return {
        "file": os.path.basename(path),
        "model": "sph",
        "steps": len(rows),
        "peak_kinetic_j": max(kinetic),
        "peak_potential_j": "",
        "peak_total_j": "",
        "final_kinetic_j": kinetic[-1],
        "final_total_j": "",
        "mean_fraction_missed": "",
        "com_height_drift_mm": drift * 1000.0,
        "drift_pe_over_peak_ke": drift_pe / max(kinetic) if max(kinetic) > 0 else float("nan"),
        "verdict": "potential not reported: COM-height drift is a numerical artifact",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=_REPO_ROOT)
    parser.add_argument("--output", default=os.path.join(_REPO_ROOT, "slosh_energy_recheck.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tank, fluid, world = TankConfig(), FluidConfig(), WorldConfig()
    gravity = abs(world.gravity[1])
    params = compute_housner_parameters(
        tank_length=tank.length,
        tank_breadth=tank.width,
        fill_height=fluid.fill_height,
        fluid_density=fluid.density,
        gravity=gravity,
    )
    logger.info(
        "Housner convective mass m_c=%.3f kg, pendulum length L=%.4f m",
        params.convective_mass, params.pendulum_length,
    )

    results: List[Dict[str, object]] = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "tank_motion_*.csv"))):
        r = recheck_pendulum_file(path, params.convective_mass, params.pendulum_length, gravity)
        if r is not None:
            results.append(r)
    for path in sorted(glob.glob(os.path.join(args.data_dir, "sph_output_*.csv"))):
        r = recheck_sph_file(path, gravity)
        if r is not None:
            results.append(r)

    for r in results:
        if r["model"] == "pendulum":
            logger.info(
                "%-45s peak KE=%.3e J  peak PE=%.3e J  peak total=%.3e J  PE share of energy=%.0f%%",
                r["file"], r["peak_kinetic_j"], r["peak_potential_j"], r["peak_total_j"],
                100.0 * float(r["mean_fraction_missed"]),
            )
        else:
            logger.info(
                "%-45s peak KE=%.3e J  max COM-height drift=%+.2f mm  M*g*drift / peak KE = %.1fx",
                r["file"], r["peak_kinetic_j"], r["com_height_drift_mm"], r["drift_pe_over_peak_ke"],
            )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
