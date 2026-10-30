"""
SPH Convergence Study Script
==============================
Runs the SAME trajectory at 3 particle resolutions and plots the
convergence of key sloshing metrics (COM displacement, natural frequency).

This is Table 1 / Figure 2 of your paper.

Provenance: contributed via an external CFD-advisor review package
(SPH_Enhancement_Guide_18Sept2026.zip, 2026-09-18). NOT yet run for
real -- this project's local machine has no CUDA GPU (see the project notes),
and no real .npz data exists yet at dx=5mm/3mm to postprocess (only the
existing dx=10mm captures). Committed as prep work for whenever a real
GPU session (the second machine, or Colab) produces those resolution
variants -- do not treat this as "a convergence study has been run."

Only the `postprocess` subcommand below is real/implemented. The
package's own QUICK_START.md additionally described a "--simulate"
flag for a full-simulation mode -- that mode does NOT exist in this
file (checked directly, not assumed): there is no --simulate argument
and no code path that drives SPHBackend/ClosedLoopSPHTank at multiple
resolutions. A real implementation of that mode would need writing
(loop scripts.run_sph_backend_trajectory's new --initial-spacing flag,
see physics/sph_backend.py, at dx = 10/5/3mm) and can only be tested on
a GPU-capable session.

Usage (post-processing mode — no GPU needed, the only mode that works):
  python -m scripts.run_convergence_study postprocess \\
      --coarse sph_particles_dx10mm.npz \\
      --medium sph_particles_dx5mm.npz \\
      --fine sph_particles_dx3mm.npz \\
      --output convergence_study.png
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def extract_com_timeseries(npz_path: str):
    """Extract center-of-mass x-displacement time series from particle data."""
    data = np.load(npz_path, allow_pickle=True)
    times = data["times"]
    positions = data["positions"]

    com_x = []
    for frame in positions:
        frame = np.asarray(frame)
        com_x.append(frame[:, 0].mean())  # x = tank length axis

    com_x = np.array(com_x)
    # Report as displacement from initial
    com_x -= com_x[0]
    return times, com_x


def estimate_natural_frequency(times, com_x):
    """Estimate sloshing natural frequency from COM displacement signal."""
    dt = np.mean(np.diff(times))
    N = len(com_x)

    # Zero-pad and FFT
    fft_vals = np.abs(np.fft.rfft(com_x - com_x.mean()))
    freqs = np.fft.rfftfreq(N, d=dt)

    # Find dominant frequency (ignore DC)
    fft_vals[0] = 0
    peak_idx = np.argmax(fft_vals)
    f_peak = freqs[peak_idx]
    omega_peak = 2 * np.pi * f_peak

    return f_peak, omega_peak


def plot_convergence(results: dict, output_path: str):
    """Plot convergence study results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # Subplot 1: COM displacement time history
    ax1 = axes[0]
    for label, data in results.items():
        ax1.plot(data["times"], data["com_x"] * 1000, label=label, linewidth=1.5)
    ax1.set_xlabel("Time (s)", fontweight="bold")
    ax1.set_ylabel("COM Displacement (mm)", fontweight="bold")
    ax1.set_title("Fluid COM Response", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Frequency convergence
    ax2 = axes[1]
    labels = list(results.keys())
    omegas = [results[k]["omega"] for k in labels]
    n_particles = [results[k]["n_particles"] for k in labels]

    # Analytical reference
    L = 0.388  # interior length
    h0 = 0.20
    g = 9.81
    k1 = np.pi / L
    omega_analytical = np.sqrt(g * k1 * np.tanh(k1 * h0))

    ax2.semilogx(n_particles, omegas, "bo-", markersize=10, linewidth=2, label="SPH")
    ax2.axhline(omega_analytical, color="r", linestyle="--", linewidth=1.5,
                label=f"Analytical (ω₁ = {omega_analytical:.2f} rad/s)")
    ax2.set_xlabel("Number of Particles", fontweight="bold")
    ax2.set_ylabel("Natural Frequency (rad/s)", fontweight="bold")
    ax2.set_title("Frequency Convergence", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Error convergence
    ax3 = axes[2]
    errors = [abs(o - omega_analytical) / omega_analytical * 100 for o in omegas]
    dx_vals = [results[k].get("dx_mm", 0) for k in labels]

    ax3.loglog(n_particles, errors, "rs-", markersize=10, linewidth=2)
    for i, (n, e, d) in enumerate(zip(n_particles, errors, dx_vals)):
        ax3.annotate(f"dx={d}mm", (n, e), textcoords="offset points",
                     xytext=(10, 5), fontsize=9)
    ax3.set_xlabel("Number of Particles", fontweight="bold")
    ax3.set_ylabel("Frequency Error (%)", fontweight="bold")
    ax3.set_title("Convergence Rate", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("SPH Particle Resolution Convergence Study",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved convergence study to %s", output_path)


def postprocess_mode(coarse_npz, medium_npz, fine_npz, output_path):
    """Analyze existing simulation outputs at different resolutions."""
    results = {}

    for label, path, dx_mm in [("Coarse (dx=10mm)", coarse_npz, 10),
                                ("Medium (dx=5mm)", medium_npz, 5),
                                ("Fine (dx=3mm)", fine_npz, 3)]:
        if path is None:
            continue
        logger.info("Processing %s from %s", label, path)
        times, com_x = extract_com_timeseries(path)
        f_hz, omega = estimate_natural_frequency(times, com_x)

        data = np.load(path, allow_pickle=True)
        n_particles = data["positions"][0].shape[0]

        results[label] = {
            "times": times,
            "com_x": com_x,
            "omega": omega,
            "f_hz": f_hz,
            "n_particles": n_particles,
            "dx_mm": dx_mm,
        }
        logger.info("  Particles: %d, Frequency: %.3f Hz (%.3f rad/s)",
                     n_particles, f_hz, omega)

    if len(results) >= 2:
        plot_convergence(results, output_path)
    else:
        logger.warning("Need at least 2 resolution levels for convergence study.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    # Post-processing mode
    pp = sub.add_parser("postprocess", help="Analyze existing .npz files")
    pp.add_argument("--coarse", help="Coarse resolution .npz (dx=10mm)")
    pp.add_argument("--medium", help="Medium resolution .npz (dx=5mm)")
    pp.add_argument("--fine", help="Fine resolution .npz (dx=3mm)")
    pp.add_argument("--output", default="convergence_study.png")

    args = parser.parse_args()
    if args.command == "postprocess":
        postprocess_mode(args.coarse, args.medium, args.fine, args.output)
    else:
        parser.print_help()
