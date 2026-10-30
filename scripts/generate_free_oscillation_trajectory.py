"""
Generate a tank_motion trajectory purpose-built for sloshing-frequency
validation (SPH_Enhancement_Guide_18Sept2026.zip's Phase D, the development log
Entry 87): a brief, smooth tilt "kick," then the tank held PERFECTLY
STILL for the rest of the run.

Why not scripts/generate_decaying_sine_trajectory.py for this: that
script (correctly, for its own purpose) drives the tank CONTINUOUSLY at
a chosen frequency with a decaying envelope -- an FFT of the resulting
fluid response is dominated by the DRIVING frequency, not the fluid's
own natural frequency (confirmed directly: postprocessing that file's
own data recovers 0.400 Hz, exactly its own driving frequency, not the
fluid's ~1.3 Hz Housner-pendulum frequency or whatever SPH's real
natural frequency turns out to be). To measure a natural frequency
instead of a forcing frequency, the tank must stop moving and the fluid
must be left to oscillate on its own.

The kick itself is a smoothstep ramp (angle(t) = A*(3u^2 - 2u^3), u =
t/kick_duration_s) rather than a step or a sine burst, specifically
because it has EXACTLY ZERO angular velocity at both t=0 and
t=kick_duration_s -- no discontinuous jump in tank_angular_velocity at
the ramp-to-hold boundary, which would otherwise inject an artificial
second impulsive disturbance right at that transition and contaminate
the "free" portion of the signal. This only holds if a sampled step
actually LANDS on u=1.0 -- a real bug in an earlier version of this
script (found on re-audit, not by a user report) used a plain t <
kick_duration_s comparison, which for a duration that isn't an exact
multiple of control_dt (0.625s / 0.02s = 31.25) never samples u=1.0 at
all, leaving a real ~0.011 rad/s velocity discontinuity between the
last ramp sample and the first held sample. Fixed by snapping
kick_duration_s to the nearest exact multiple of control_dt and
indexing the ramp by integer step count (step <= n_kick_steps) instead
of comparing floating-point time, so the ramp branch itself lands on
u=1.0 -- angle=A, velocity=0, both exact -- at a real sampled step,
matching the held branch's very next sample exactly rather than
approximately.

Default kick amplitude/duration are NOT arbitrary: amplitude matches
MotorConfig's own real default (0.15 rad), and duration (a quarter of
MotorConfig's real 0.4 Hz period, i.e. 1/(4*0.4) = 0.625s) is chosen so
the ramp's own peak angular velocity (1.5*amplitude/duration ~= 0.36
rad/s) lands close to the peak angular velocity of the already-
SPH-tested 0.15rad/0.4Hz sine motion used throughout Entries 40-43/46-49
(amplitude*omega = 0.15*2*pi*0.4 ~= 0.377 rad/s) -- reusing an
already-validated disturbance scale rather than introducing a new,
untested one.

Writes the same CSV schema as generate_decaying_sine_trajectory.py /
record_fluid_trajectory.py, so scripts/run_sph_backend_trajectory.py and
scripts/compare_fluid_backends.py work on it unchanged.

Run (local, PyChrono-core only, no FSI/CUDA needed for this generation
step -- only the later SPHBackend replay needs a GPU):
    python -m scripts.generate_free_oscillation_trajectory --output tank_motion_free_oscillation.csv
"""

from __future__ import annotations

import argparse
import csv
import logging

from builders.fluid_builder import FluidBuilder
from config.simulation_config import EnvironmentConfig, FluidConfig, MotorConfig, TankConfig, WorldConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate(
    kick_amplitude_rad: float,
    kick_duration_s: float,
    n_steps: int,
    control_dt: float,
) -> list:
    tank_config = TankConfig()
    fluid_config = FluidConfig()
    world_config = WorldConfig()
    fluid_builder = FluidBuilder(tank_config, fluid_config, world_config)
    backend = fluid_builder.build()
    backend.reset()

    # A real bug, found on re-audit and fixed here (not just described more
    # carefully): kick_duration_s is rarely an exact multiple of control_dt
    # (0.625s / 0.02s = 31.25), so the original t < kick_duration_s
    # comparison never actually SAMPLES u=1.0 -- the ramp branch's last
    # sample lands at u~0.992 with a small but nonzero velocity, then the
    # very next step jumps straight to the held branch's velocity=0,
    # producing exactly the kind of discontinuous jump this trajectory was
    # supposed to avoid (verified: it was really there, ~0.011 rad/s over
    # one 0.02s step, in the first version of this script). Fixed by
    # snapping kick_duration_s to an exact multiple of control_dt and
    # using an integer step comparison, so the ramp branch itself reaches
    # u=1.0 (angle=A, velocity=0, both exactly) at a real sampled step --
    # the held branch's very next sample matches it exactly, not
    # approximately.
    n_kick_steps = max(1, round(kick_duration_s / control_dt))
    kick_duration_s = n_kick_steps * control_dt

    logger.info(
        "Free-oscillation kick: amplitude=%.4f rad, ramp duration=%.4f s "
        "(%d steps, peak angular velocity ~%.4f rad/s), then held still for "
        "%d steps @ %.3fs = %.2fs total simulated.",
        kick_amplitude_rad, kick_duration_s, n_kick_steps,
        1.5 * kick_amplitude_rad / kick_duration_s,
        n_steps, control_dt, n_steps * control_dt,
    )

    rows = []
    for step in range(n_steps):
        t = step * control_dt
        if step <= n_kick_steps:
            u = step / n_kick_steps  # exactly 0 at step=0, exactly 1 at step=n_kick_steps
            tank_angle = kick_amplitude_rad * (3 * u * u - 2 * u * u * u)
            # d/dt[A*(3u^2 - 2u^3)], u = t/T -> A*(6u - 6u^2)/T
            tank_angular_velocity = kick_amplitude_rad * (6 * u - 6 * u * u) / kick_duration_s
        else:
            tank_angle = kick_amplitude_rad
            tank_angular_velocity = 0.0

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
        "Kick complete at step %d (t=%.3fs); tank held at %.4f rad, 0 rad/s "
        "for the remaining %d steps -- the fluid's OWN free response fills "
        "the rest of this trajectory.",
        n_kick_steps, kick_duration_s, kick_amplitude_rad,
        n_steps - n_kick_steps,
    )
    return rows


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote free-oscillation tank_motion + pendulum-prediction CSV to %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kick-amplitude", type=float, default=MotorConfig().amplitude_rad, help="Peak tank tilt reached at the end of the ramp, rad. Default matches MotorConfig's own real amplitude.")
    parser.add_argument("--kick-duration", type=float, default=0.625, help="Ramp duration, s. Default (a quarter of MotorConfig's real 0.4 Hz period) keeps peak angular velocity close to the already-SPH-validated 0.15rad/0.4Hz motion's own peak (~0.377 rad/s) -- see this module's docstring for the exact reasoning.")
    parser.add_argument("--n-steps", type=int, default=None, help="Defaults to EnvironmentConfig.max_episode_steps (500 = 10s at the default control_dt) -- plenty of hold-still time for a clean FFT frequency estimate.")
    parser.add_argument("--output", default="tank_motion_free_oscillation.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    env_config = EnvironmentConfig()
    n_steps = args.n_steps if args.n_steps is not None else env_config.max_episode_steps
    rows = generate(args.kick_amplitude, args.kick_duration, n_steps, env_config.control_dt)
    write_csv(args.output, rows)
