"""
Phase 13-14 -- Render the real captured SPH particle snapshots
(physics/sph_backend.py::SPHBackend.get_fluid_particle_positions(), saved
by the Colab run as a .npz -- see the development log Entry 44) into an actual
animation of the fluid sloshing.

This is the first real visualization of Chrono::FSI fluid in this
project -- everything else FSI-related so far (Entries 33-44) has been
numbers/console output only. It is entirely separate from main.py's
--fluid-viz* flags, which render a CPU-only DECORATIVE ripple driven by
PendulumBackend, not real SPH particles (see the project notes' Architecture
section) -- this script renders the real thing, offline, from data
already captured on Colab (no live PyChrono/FSI needed to run this
script itself, since it only reads the saved .npz + CSV).

Renders a 2D side view (tank length vs. height -- the plane of rocking
motion, matching PendulumBackend's own 2D model) since the captured
particle data is inherently a subsampled 3D point cloud and a 2D
projection reads far more clearly as "water" than a 3D scatter would at
GIF resolution. The tank outline (floor + two side walls, rotated by the
tank's own angle at each snapshot) is drawn for visual reference so the
fluid's motion relative to the rocking container is legible.

Run:
    python -m scripts.render_sph_particle_animation \
        --particles-npz sph_particles_decaying_sine.npz \
        --tank-motion-csv tank_motion_decaying_sine.csv \
        --output decaying_sine_sph_animation.gif
"""

from __future__ import annotations

import argparse
import csv
import logging

import numpy as np

from config.simulation_config import EnvironmentConfig, TankConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_tank_angle_by_step(tank_motion_csv: str) -> dict:
    with open(tank_motion_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(row["step"]): float(row["tank_angle"]) for row in rows}


def _tank_outline_local(tank_config: TankConfig) -> np.ndarray:
    """Interior tank walls (floor + two sides) in the tank's own local
    frame (x=length, z=height, matching SPHBackend's sim-frame axes),
    as a single polyline: left-wall-top -> floor-left -> floor-right ->
    right-wall-top."""
    half_length = (tank_config.length - 2 * tank_config.wall_thickness) / 2.0
    wall_height = tank_config.height
    return np.array(
        [
            [-half_length, wall_height],
            [-half_length, 0.0],
            [half_length, 0.0],
            [half_length, wall_height],
        ]
    )


def _rotate_xz(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate points from the tank's own LOCAL frame into WORLD frame by
    tank_angle -- the forward rotation, needed here because the tank
    outline is defined in local coordinates (_tank_outline_local) but
    must be drawn alongside the particle data, which Chrono already
    reports in world frame.

    NOT the same transform as SPHBackend._rotate_to_tank_frame(), even
    though it looks similar -- that one deliberately does the INVERSE
    (world -> local, undoing a forward rotation) for a different purpose
    (reporting fluid_com in the tank's own frame). Confusing the two is a
    real bug that was caught visually: reusing the inverse-rotation
    formula here made the hand-drawn tank outline rotate the wrong way,
    so it stopped matching the real (correctly world-frame) fluid data --
    the fluid particles were never actually leaking outside the tank,
    the outline was drawn in the wrong place. Confirmed fixed by checking
    multiple frames at both positive and negative tank angles."""
    cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
    x, z = points[:, 0], points[:, 1]
    return np.column_stack([x * cos_t + z * sin_t, -x * sin_t + z * cos_t])


def render(
    particles_npz: str,
    tank_motion_csv: str,
    output_path: str,
    snapshot_every_n_steps: int,
    fps: int,
    disable_boundary_filter: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    data = np.load(particles_npz, allow_pickle=True)
    times = data["times"]
    raw_positions = data["positions"]  # object array, one (N_i, 3) array per snapshot
    n_frames = len(times)
    logger.info("Loaded %d particle snapshots from %s", n_frames, particles_npz)

    # Real bug found 2026-09-16 (the development log Entry 71): the .npz capture
    # can run past the point tank_motion_csv actually has real tank_angle
    # for -- traced to scripts/run_closed_loop_sph_policy.py continuing to
    # log/capture for dozens of extra steps after a mid-step safety trip
    # instead of stopping immediately (now fixed there too). Silently
    # defaulting a missing step's angle to 0.0 (the old behavior) doesn't
    # just draw the tank wrong -- it renders whatever the capture recorded
    # AFTER the run should have stopped, i.e. an already-destabilized
    # simulation, as if it were valid data. Truncate to the longest
    # CONTIGUOUS prefix of frames whose step actually has a real CSV row,
    # rather than trusting frames past the first gap.
    _tank_angle_probe = load_tank_angle_by_step(tank_motion_csv)
    _valid_n_frames = n_frames
    for _i in range(n_frames):
        if (_i * snapshot_every_n_steps) not in _tank_angle_probe:
            _valid_n_frames = _i
            break
    if _valid_n_frames < n_frames:
        logger.warning(
            "%s covers steps up to %d, but %s captured frames up to step %d -- "
            "truncating from %d to %d frames (dropping frames with no real "
            "tank_angle, which also reflect an already-destabilized run past "
            "where it should have stopped; see the development log Entry 71).",
            tank_motion_csv, max(_tank_angle_probe.keys()) if _tank_angle_probe else -1,
            particles_npz, (n_frames - 1) * snapshot_every_n_steps,
            n_frames, _valid_n_frames,
        )
        times = times[:_valid_n_frames]
        raw_positions = raw_positions[:_valid_n_frames]
        n_frames = _valid_n_frames

    # A tiny number of markers in the raw snapshot are boundary/reference
    # points that go completely and exactly static partway through the
    # run and never move again (confirmed directly: one such point moved
    # once, in a single jump between t=0.0s and t=0.5s, then stayed
    # BYTE-IDENTICAL for the remaining ~19.5s of captured frames -- a
    # "total displacement across the whole run" check misses this, since
    # that one early jump is enough range to look like real motion).
    # Genuine fluid particles keep moving (however slightly) throughout;
    # something that goes and stays perfectly frozen for a long trailing
    # stretch of the run is not real fluid, most likely a BCE/rigid-body
    # marker or a numerically-stuck artifact that ended up in what should
    # be the fluid-only slice. Negligible effect on SPHBackend.get_state()'s
    # own COM/slosh_energy numbers (roughly 1 in 14,763 total markers) but
    # visually wrong in a rendered scatter plot -- filtered out here for
    # the visualization specifically, by checking movement across the
    # LATTER HALF of the run only (past any early transient), dropped by
    # original array index so the filter is consistent across every frame
    # of the animation.
    if disable_boundary_filter:
        # Real finding (LiquidSloshingRL the development log Entry 60): the
        # "static in the second half = boundary artifact" heuristic below
        # was tuned against SPHBackend runs where under 0.1% of markers
        # ever matched it. A ClosedLoopSPHTank run ending in the same
        # torque_limit=7.0/artificial_viscosity=0.1 regime Entry 56 already
        # flagged as marginally stable showed ~34% of markers going
        # perfectly static partway through and staying frozen -- verified
        # directly to be real fluid markers (GetNumFluidMarkers()-sliced,
        # same convention SPHBackend uses), not leaked rigid-body/BCE
        # markers, and spatially spread across the whole fluid volume, not
        # confined near a wall. The freeze-vs-real-artifact signature
        # (moves briefly, then goes byte-identical) is indistinguishable
        # from the original rare-artifact case by position data alone, so
        # blindly applying that filter here would hide real data rather
        # than clean up a genuine artifact. Pass this flag to show the
        # true, complete captured data instead -- if a run really does
        # have this many markers freeze, that's worth seeing, not masking.
        positions = raw_positions
        logger.info(
            "Boundary-marker filter disabled -- showing all %d markers "
            "as captured, including any that freeze mid-run.",
            raw_positions[0].shape[0] if n_frames else 0,
        )
    elif n_frames >= 4:
        stacked = np.stack([np.asarray(p, dtype=np.float64) for p in raw_positions])  # (n_frames, N, 3)
        second_half = stacked[n_frames // 2 :]
        late_range = np.linalg.norm(second_half.max(axis=0) - second_half.min(axis=0), axis=1)  # (N,)
        moved = late_range > 1e-6
        n_dropped = int((~moved).sum())
        if n_dropped:
            logger.info(
                "Dropping %d marker(s) out of %d that go and stay exactly "
                "static for the run's second half -- boundary/reference "
                "artifacts, not real fluid.",
                n_dropped, len(moved),
            )
        positions = np.array([frame[moved] for frame in raw_positions], dtype=object)
    else:
        positions = raw_positions

    tank_angle_by_step = _tank_angle_probe  # already loaded above, for the truncation check
    env_config = EnvironmentConfig()
    tank_config = TankConfig()
    tank_outline_local = _tank_outline_local(tank_config)

    # snapshot i corresponds to control step i * snapshot_every_n_steps
    steps = [i * snapshot_every_n_steps for i in range(n_frames)]
    tank_angles = [tank_angle_by_step.get(s, 0.0) for s in steps]

    half_length = tank_config.length / 2.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-half_length * 1.3, half_length * 1.3)
    ax.set_ylim(-0.02, tank_config.height * 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("tank length axis (m)")
    ax.set_ylabel("tank height axis (m)")

    scatter = ax.scatter([], [], s=3, c="#1f77b4", alpha=0.6)
    (outline_line,) = ax.plot([], [], color="black", linewidth=1.5)
    title = ax.set_title("")

    def update(frame_idx: int):
        pos = positions[frame_idx]  # (N, 3) sim-frame: x=length, y=width, z=height
        xz = pos[:, [0, 2]]  # 2D side-view projection, drop the width axis
        scatter.set_offsets(xz)

        angle = tank_angles[frame_idx]
        rotated_outline = _rotate_xz(tank_outline_local, angle)
        outline_line.set_data(rotated_outline[:, 0], rotated_outline[:, 1])

        title.set_text(
            f"t={times[frame_idx]:.2f}s  tank_angle={angle:+.3f} rad "
            f"({np.degrees(angle):+.1f} deg)"
        )
        return scatter, outline_line, title

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    logger.info("Wrote animation to %s (%d frames @ %d fps)", output_path, n_frames, fps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles-npz", required=True)
    parser.add_argument("--tank-motion-csv", required=True)
    parser.add_argument("--output", default="sph_particle_animation.gif")
    parser.add_argument("--snapshot-every-n-steps", type=int, default=5, help="Must match the SNAPSHOT_EVERY_N_STEPS used when the .npz was captured on Colab.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--disable-boundary-filter",
        action="store_true",
        help="Show all captured markers as-is, skipping the static-in-second-half "
             "boundary/reference-marker filter. See the development log Entry 60 -- that "
             "filter was tuned against SPHBackend runs (<0.1% of markers ever "
             "matched it) and can't reliably distinguish real frozen fluid markers "
             "from true artifacts when a much larger fraction matches, as seen on "
             "some ClosedLoopSPHTank runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    render(
        args.particles_npz, args.tank_motion_csv, args.output,
        args.snapshot_every_n_steps, args.fps,
        disable_boundary_filter=args.disable_boundary_filter,
    )
