"""
Phase 13-14 -- Generate a synthetic, controlled tank_motion trajectory
that returns to level, for a second PendulumBackend-vs-SPHBackend
comparison.

Why not just reuse a trained-policy rollout: the development log Entry 43
found the trained policy (models/ppo_sloshing.zip) settles the tank at
a steady ~-0.1786 rad tilt, not back to level -- a direct consequence
of PendulumBackend's own slosh_energy metric being blind to whatever
angle the tank stops at (it only depends on velocity, which goes to
zero regardless of the resting angle). That trajectory was exactly the
right one to expose that blind spot, but it can't isolate whether
PendulumBackend's OSCILLATORY/transient response agrees with SPH any
better, since the back half of that run was all spent in the
static-tilt regime the pendulum model structurally can't represent.

This script sidesteps needing a policy or real tank dynamics at all --
tank_angle/tank_angular_velocity are a directly-specified, exactly
analytic decaying sine:
    tank_angle(t) = amplitude_rad * exp(-decay_rate * t) * sin(2*pi*f*t)
driven at the fluid's own natural frequency (read from FluidBuilder's
real Housner computation, not hardcoded) so there's a rich, resonant
sloshing response to compare, decaying to near-zero tilt by the end of
the run -- guaranteed to return to level, unlike the trained-policy
trajectory.

Writes the same CSV schema scripts/record_fluid_trajectory.py produces,
so scripts/compare_fluid_backends.py works on it unchanged, and the
SAME Colab script/procedure used for Entry 43 replays it through
SPHBackend.

Run:
    python -m scripts.generate_decaying_sine_trajectory --output tank_motion_decaying_sine.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math

from builders.fluid_builder import FluidBuilder
from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig, WorldConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate(
    amplitude_rad: float,
    decay_rate_per_s: float,
    n_steps: int,
    control_dt: float,
    frequency_hz: float = None,
) -> list:
    tank_config = TankConfig()
    fluid_config = FluidConfig()
    world_config = WorldConfig()
    fluid_builder = FluidBuilder(tank_config, fluid_config, world_config)
    if frequency_hz is None:
        # Defaults to the fluid's own natural (resonant) frequency -- but
        # NOTE (added after a real Colab crash): resonance drives peak
        # angular velocity ~3.4x higher than MotorConfig's real 0.4 Hz
        # default for the same amplitude, and was confirmed to numerically
        # blow up the SPH solver (`derivVel is NAN`) partway through a real
        # run. Every FSI test that has actually worked end to end (Entries
        # 37-40, 43) used 0.4 Hz -- pass --frequency 0.4 explicitly for any
        # trajectory that's actually going into SPHBackend, not the
        # resonance default.
        frequency_hz = fluid_builder.compute_backend_parameters().natural_frequency_hz
    omega = 2.0 * math.pi * frequency_hz
    backend = fluid_builder.build()
    backend.reset()

    logger.info(
        "Decaying sine: amplitude=%.4f rad, frequency=%.4f Hz (fluid's own natural "
        "frequency), decay_rate=%.4f /s, %d steps @ %.3fs = %.2fs simulated",
        amplitude_rad, frequency_hz, decay_rate_per_s, n_steps, control_dt,
        n_steps * control_dt,
    )

    rows = []
    for step in range(n_steps):
        t = step * control_dt
        envelope = amplitude_rad * math.exp(-decay_rate_per_s * t)
        tank_angle = envelope * math.sin(omega * t)
        # d/dt[envelope * sin(omega t)], envelope = A*exp(-decay_rate*t)
        tank_angular_velocity = envelope * (
            omega * math.cos(omega * t) - decay_rate_per_s * math.sin(omega * t)
        )

        backend.step(control_dt, (tank_angle, tank_angular_velocity))
        state = backend.get_state()

        rows.append(
            {
                "step": step,
                "tank_angle": tank_angle,
                "tank_angular_velocity": tank_angular_velocity,
                "pendulum_fluid_com_x": state.com_position[0],
                "pendulum_fluid_com_y": state.com_position[1],
                "pendulum_fluid_com_vx": state.com_velocity[0],
                "pendulum_fluid_com_vy": state.com_velocity[1],
                "pendulum_slosh_energy": state.slosh_energy,
            }
        )

    logger.info(
        "Final tank_angle=%.6f rad (%.2f%% of initial amplitude) -- returns to level.",
        rows[-1]["tank_angle"], 100.0 * abs(rows[-1]["tank_angle"]) / amplitude_rad,
    )
    return rows


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote decaying-sine tank_motion + pendulum-prediction CSV to %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=0.15, help="Initial tank tilt amplitude, rad (matches MotorConfig's default).")
    parser.add_argument("--decay-rate", type=float, default=0.23, help="Envelope decay rate, 1/s (default decays to ~10%% of amplitude over 10s).")
    parser.add_argument("--n-steps", type=int, default=None, help="Defaults to EnvironmentConfig.max_episode_steps (500).")
    parser.add_argument("--frequency", type=float, default=None, help="Hz. Defaults to the fluid's own natural (resonant) frequency -- see this module's own docstring/generate()'s note on why 0.4 (MotorConfig's real default) is the safer choice for anything actually run through SPHBackend.")
    parser.add_argument("--output", default="tank_motion_decaying_sine.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    env_config = EnvironmentConfig()
    n_steps = args.n_steps if args.n_steps is not None else env_config.max_episode_steps
    rows = generate(args.amplitude, args.decay_rate, n_steps, env_config.control_dt, args.frequency)
    write_csv(args.output, rows)
