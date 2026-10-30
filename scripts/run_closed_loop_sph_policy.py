"""
Phase 15 -- run the trained policy in a genuine closed loop against
physics/closed_loop_sph_tank.py::ClosedLoopSPHTank: a real, torque-
driven, FSI-coupled tank, not a replay of a pre-recorded trajectory
(scripts/record_fluid_trajectory.py + scripts/compare_fluid_backends.py,
Entries 43-44) and not the fast analytic surrogate
(main_trained_policy.py). Each control step, this script reads the
tank's REAL resulting state, feeds it to the policy, and applies the
resulting torque back into the same live FSI scene -- see
the development log Entry 48 for the full design decision.

Cannot run on this project's local machine at all (no CUDA, no
pychrono.fsi) -- written for a real CUDA-capable machine (the RTX
3090-class desktop this was built for, or Colab as a fallback). Same
ctypes preload sequence as every other FSI script in this project (see
the project notes' Phase 13-14 section) is assumed to have already run before
this module is imported.

NOT real-time and NOT interactive -- see this project's own measured
SPHBackend performance (the development log Entries 40, 43-46, ~35-46x
slower than real-time on a Tesla T4). This produces a CSV +
periodically-saved particle-snapshot .npz to review afterward, the
same way Entries 43-46 did, via scripts/render_sph_particle_animation.py
-- not a live window.

Run (on the real FSI-capable machine, after the ctypes preload):
    python -m scripts.run_closed_loop_sph_policy models/ppo_sloshing \
        --n-steps 500 --output-csv closed_loop_output.csv \
        --particles-npz closed_loop_particles.npz
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig
from scripts._gpu_info import log_gpu_info
from physics.closed_loop_sph_tank import ClosedLoopSPHTank

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_normalizer(vecnormalize_path):
    if vecnormalize_path is None:
        return None
    import environment  # noqa: F401 -- gym.register() side effect
    import gymnasium as gym

    throwaway = DummyVecEnv([lambda: gym.make("LiquidSloshingRL/SloshingSuppression-v0")])
    normalizer = VecNormalize.load(vecnormalize_path, throwaway)
    normalizer.training = False
    return normalizer


def build_observation(tank_angle: float, tank_angular_velocity: float, fluid_state) -> np.ndarray:
    return np.array(
        [
            tank_angle, tank_angular_velocity,
            fluid_state.com_position[0], fluid_state.com_position[1],
            fluid_state.com_velocity[0], fluid_state.com_velocity[1],
            fluid_state.slosh_energy,
        ],
        dtype=np.float32,
    )


# Real, evidence-based safety limit (the development log Entry 57): every observed
# SPH solver crash across many real trials (Entries 55-57) was preceded by
# angular velocity reaching roughly 1.0-1.2 rad/s. Checking the PREVIOUS
# step's resulting angular velocity before attempting another step catches
# the danger zone before the solver is pushed into a crash-inducing regime
# -- a real, tested mitigation (environment/closed_loop_sph_env.py::
# ClosedLoopSPHEnv uses the identical check), not a guess.
_ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S = 1.0


def run(
    model_path: str,
    vecnormalize_path,
    n_steps: int,
    output_csv: str,
    particles_npz: str,
    snapshot_every_n_steps: int,
    save_every_n_steps: int,
    particle_subsample: int,
    torque_limit: float = None,
    artificial_viscosity: float = 0.1,
    step_size: float = 1.0e-4,
    num_bce_layers: int = 4,
    boundary_method: str = "adami",
    initial_spacing: float = 0.01,
    max_velocity: float = 4.5,
    viscosity: float = 1.0,
    force: bool = False,
) -> None:
    # Interruption/resume (added 2026-09-18, ahead of first access to a
    # third real machine -- see the project notes' "RTX 3090 machine note").
    # Real, honest protection, NOT a claim that the FSI particle state
    # itself can resume mid-run -- Chrono::FSI exposes no API to
    # serialize/restore thousands of particles' positions+velocities.
    # If output_csv already has n_steps rows, this run is already fully
    # done -- skip it (and skip paying for a fresh ClosedLoopSPHTank
    # build, by checking BEFORE that below) rather than redoing it after
    # a restart. A run interrupted PARTWAY must still restart from step
    # 0 -- save_every_n_steps below just bounds how much is lost when
    # that happens, it does not make the physics itself resumable.
    if not force and os.path.exists(output_csv):
        with open(output_csv, newline="") as f:
            existing_rows = sum(1 for _ in csv.reader(f)) - 1
        if existing_rows == n_steps:
            logger.info(
                "%s already has all %d steps -- skipping (already fully done, "
                "not redoing it after a restart). Pass force=True / --force to override.",
                output_csv, n_steps,
            )
            return

    import pychrono.fsi as fsi

    log_gpu_info(logger)
    model = PPO.load(model_path)
    normalizer = _load_normalizer(vecnormalize_path)
    logger.info("Loaded policy from %s", model_path)

    boundary_method_value = {
        "adami": fsi.BoundaryMethod_ADAMI,
        "holmes": fsi.BoundaryMethod_HOLMES,
    }[boundary_method]

    env_config = EnvironmentConfig(max_motor_torque_nm=torque_limit) if torque_limit else EnvironmentConfig()
    tank = ClosedLoopSPHTank(
        tank_config=TankConfig(fixed=False), fluid_config=FluidConfig(), env_config=env_config,
        artificial_viscosity=artificial_viscosity, step_size=step_size, num_bce_layers=num_bce_layers,
        boundary_method=boundary_method_value, initial_spacing=initial_spacing,
        max_velocity=max_velocity, viscosity=viscosity,
    )
    logger.info(
        "ClosedLoopSPHTank built and Initialize()'d (torque_limit=%.2f, artificial_viscosity=%.3f, "
        "num_bce_layers=%d, boundary_method=%s, initial_spacing=%.4f, max_velocity=%.2f, viscosity=%.4f).",
        env_config.max_motor_torque_nm, artificial_viscosity, num_bce_layers, boundary_method,
        initial_spacing, max_velocity, viscosity,
    )

    out_rows = []
    snapshot_times = []
    snapshot_positions = []

    def save_progress():
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        np.savez_compressed(
            particles_npz,
            times=np.array(snapshot_times),
            positions=np.array(snapshot_positions, dtype=object),
        )

    obs = build_observation(tank.angle, tank.angular_velocity, tank.get_fluid_state())

    t0 = time.time()
    for step in range(n_steps):
        if step > 0 and abs(tank.angular_velocity) > _ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S:
            logger.warning(
                "step %d: angular_velocity=%.4f exceeds safety limit (%.2f) -- "
                "stopping BEFORE attempting another step, not after a solver crash "
                "(the development log Entry 57).",
                step, tank.angular_velocity, _ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S,
            )
            break

        model_input = normalizer.normalize_obs(obs) if normalizer is not None else obs
        action, _ = model.predict(model_input, deterministic=True)
        torque_nm = float(np.clip(action, -1.0, 1.0).reshape(-1)[0]) * env_config.max_motor_torque_nm
        tank.set_commanded_torque(torque_nm)

        # the development log Entry 58: pass the safety limit into step() itself
        # so a mid-step blowup (found in a real 16k fine-tune run -- the
        # before-step-only check above missed it) stops the internal
        # substep loop immediately instead of running to a crash.
        completed = tank.step(
            env_config.control_dt,
            angular_velocity_safety_limit=_ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S,
        )
        if not completed:
            logger.warning(
                "step %d: mid-step safety trip -- angular_velocity=%.4f exceeded "
                "safety limit (%.2f) WITHIN this control step's substeps "
                "(the development log Entry 58). Stopping now, not after another step -- "
                "a real bug (found 2026-09-16) let this fall through to the next "
                "iteration instead, so the before-step check above was the only "
                "thing that ever actually stopped the run, letting capture continue "
                "for dozens of extra steps against an already-destabilized simulation.",
                step, tank.angular_velocity, _ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S,
            )
            fluid_state = tank.get_fluid_state()
            tank_angle, tank_angular_velocity = tank.angle, tank.angular_velocity
            out_rows.append({
                "step": step,
                "tank_angle": tank_angle,
                "tank_angular_velocity": tank_angular_velocity,
                "torque_nm": torque_nm,
                "fluid_com_x": fluid_state.com_position[0],
                "fluid_com_y": fluid_state.com_position[1],
                "slosh_energy": fluid_state.slosh_energy,
            })
            if step % snapshot_every_n_steps == 0:
                positions = tank.get_fluid_particle_positions()
                snapshot_times.append(step * env_config.control_dt)
                snapshot_positions.append(positions[::particle_subsample].copy())
            break

        fluid_state = tank.get_fluid_state()
        tank_angle, tank_angular_velocity = tank.angle, tank.angular_velocity
        obs = build_observation(tank_angle, tank_angular_velocity, fluid_state)

        out_rows.append({
            "step": step,
            "tank_angle": tank_angle,
            "tank_angular_velocity": tank_angular_velocity,
            "torque_nm": torque_nm,
            "fluid_com_x": fluid_state.com_position[0],
            "fluid_com_y": fluid_state.com_position[1],
            "slosh_energy": fluid_state.slosh_energy,
        })

        if step % snapshot_every_n_steps == 0:
            positions = tank.get_fluid_particle_positions()
            snapshot_times.append(step * env_config.control_dt)
            snapshot_positions.append(positions[::particle_subsample].copy())

        if step % save_every_n_steps == 0:
            elapsed = time.time() - t0
            logger.info(
                "step %d: t=%.2fs tank_angle=%.4f torque=%.4f "
                "fluid_com_x=%.5f slosh_energy=%.6f (elapsed %.1fs real)",
                step, step * env_config.control_dt, tank_angle, torque_nm,
                fluid_state.com_position[0], fluid_state.slosh_energy, elapsed,
            )
            save_progress()

    # Real bug fixed 2026-09-16: this used to always report n_steps (the
    # REQUESTED total from --n-steps), even when a safety trip broke the
    # loop early -- len(out_rows) is the actual number of steps that ran,
    # which can be far fewer (confirmed directly: a real eval survived
    # only 24 of a requested 250 steps, but this line would have claimed
    # "Ran 250 steps").
    actual_steps = len(out_rows)
    elapsed = time.time() - t0
    logger.info(
        "Ran %d of %d requested steps (%.2fs simulated) in %.2fs real time",
        actual_steps, n_steps, actual_steps * env_config.control_dt, elapsed,
    )
    save_progress()
    logger.info("Wrote %s and %s", output_csv, particles_npz)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", default="models/ppo_sloshing")
    parser.add_argument("--vecnormalize-path", default=None)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--output-csv", default="closed_loop_output.csv")
    parser.add_argument("--particles-npz", default="closed_loop_particles.npz")
    parser.add_argument("--snapshot-every-n-steps", type=int, default=5)
    parser.add_argument("--save-every-n-steps", type=int, default=25)
    parser.add_argument("--particle-subsample", type=int, default=8)
    parser.add_argument("--torque-limit", type=float, default=None, help="Overrides EnvironmentConfig.max_motor_torque_nm. Default (None) uses the training-time value (2.0 Nm) -- the development log Entry 56 found 7.0 Nm gives much better closed-loop survival.")
    parser.add_argument("--artificial-viscosity", type=float, default=0.1, help="Entry 56 found 0.1 (not ClosedLoopSPHTank's own 0.03 default) prevents the SPH solver crash outright, though the real fix is the velocity-based safety stop above, not this parameter alone.")
    parser.add_argument("--step-size", type=float, default=1.0e-4, help="ClosedLoopSPHTank's own internal substep size (s). Default matches ClosedLoopSPHTank's own default. Exposed for controlled re-runs investigating the transient wall-penetration issue (the development log Entry 64) -- a smaller value tests whether penetration is a discretization artifact.")
    parser.add_argument("--num-bce-layers", type=int, default=4, help="Number of boundary-condition-enforcement marker layers the ADAMI boundary method draws from. Default (4) matches ClosedLoopSPHTank's own prior default -- unchanged unless explicitly overridden. Exposed for a real, controlled test of whether more layers reduce the transient wall-penetration issue (the development log Entry 61/73/75) -- the one legitimate idea from the external reference package's own (never-functional) baffled_tank_geometry.py, tried here for real via Chrono's actual SPHParameters.num_bce_layers instead.")
    parser.add_argument("--boundary-method", choices=["adami", "holmes"], default="adami", help="Chrono::FSI boundary-marker method (fsi.BoundaryMethod_*). 'adami' matches ClosedLoopSPHTank's prior default -- unchanged unless explicitly overridden. 'holmes' is a genuinely different boundary-marker approach (not a parameter tweak around ADAMI), confirmed to exist in this installed build by inspecting pychrono.fsi directly -- exposed for a real, controlled A/B test of whether it reduces the transient wall-penetration issue (the development log Entry 61/64), since three separate tuning attempts around ADAMI itself (viscosity, step_size, num_bce_layers -- Entries 73/77) were all closed negative.")
    parser.add_argument("--initial-spacing", type=float, default=0.01, help="SPH particle spacing (m). Default (0.01, ~1,846 particles in this tank) matches every existing checkpoint/finding in the development log. An external CFD-advisor review (SPH_Enhancement_Guide_18Sept2026.zip, 2026-09-18) flagged this as too coarse for publication-quality rendering ('pearl' artifact) and recommended 0.003 (~68,000 particles) -- exposed here for a real GPU run to test that claim, not applied as a silent default change; ~37x more particles means substantially longer runtime (guide estimates 15-25 min per 2s of sim on an RTX 4050 class GPU).")
    parser.add_argument("--max-velocity", type=float, default=4.5, help="SPHParameters.max_velocity -- sets the weakly-compressible EOS's sound speed (c_s = 10x this). Default (4.5) matches ClosedLoopSPHTank's own prior default. The same 2026-09-18 CFD-advisor review recommended 2.0 as closer to this tank's real peak fluid velocity, allowing a larger stable timestep -- untested locally (no GPU), exposed for a real controlled comparison.")
    parser.add_argument("--viscosity", type=float, default=1.0, help="FluidProperties.viscosity (Pa*s, dynamic). Default (1.0) is a real, longstanding placeholder -- 1000x real water's dynamic viscosity (~0.001 Pa*s at 20C) -- flagged by the same 2026-09-18 CFD-advisor review. Every existing checkpoint/comparison in the development log was produced at this honey-thick default, so it is NOT silently changed here; pass --viscosity 0.001 for a real, controlled physical-water comparison.")
    parser.add_argument("--force", action="store_true", help="Re-run even if --output-csv already has all --n-steps rows (by default, an already-fully-completed run is skipped, not redone -- see the interruption/resume note in run()'s own docstring/comments).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        default_stats_path = args.model_path.rstrip(".zip") + "_vecnormalize.pkl"
        if os.path.exists(default_stats_path):
            vecnormalize_path = default_stats_path
            logger.info("Found VecNormalize stats at %s, using them.", vecnormalize_path)

    run(
        args.model_path, vecnormalize_path, args.n_steps, args.output_csv,
        args.particles_npz, args.snapshot_every_n_steps, args.save_every_n_steps,
        args.particle_subsample, args.torque_limit, args.artificial_viscosity,
        args.step_size, args.num_bce_layers, args.boundary_method,
        args.initial_spacing, args.max_velocity, args.viscosity, args.force,
    )
