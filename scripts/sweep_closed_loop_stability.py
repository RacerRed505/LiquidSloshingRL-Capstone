"""
Phase 15 -- sweep torque-limit / reset-disturbance combinations against
physics/closed_loop_sph_tank.py::ClosedLoopSPHTank to find a configuration
that survives a full run without triggering the SPH solver's numerical
stability limit (the development log Entry 55).

NOT a permanent project script in the same sense as scripts/run_closed_loop_sph_policy.py
-- a diagnostic tool for this specific investigation. Runs one configuration
at a time (call repeatedly with different --torque-limit/--reset-velocity),
reports whether it completed or crashed, and where.

Run:
    python -m scripts.sweep_closed_loop_stability --torque-limit 1.0 --reset-velocity 0.05 --n-steps 200
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pychrono.fsi as fsi

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig
from physics.closed_loop_sph_tank import ClosedLoopSPHTank

_BOUNDARY_METHOD_VALUES = {
    "adami": fsi.BoundaryMethod_ADAMI,
    "holmes": fsi.BoundaryMethod_HOLMES,
}


def build_obs(angle, ang_vel, state):
    return np.array(
        [angle, ang_vel, state.com_position[0], state.com_position[1],
         state.com_velocity[0], state.com_velocity[1], state.slosh_energy],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/ppo_sloshing")
    parser.add_argument("--vecnormalize-path", default="models/ppo_sloshing_vecnormalize.pkl")
    parser.add_argument("--torque-limit", type=float, required=True, help="max_motor_torque_nm used both for action scaling AND ClosedLoopSPHTank's own clip.")
    parser.add_argument("--reset-velocity", type=float, required=True)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--artificial-viscosity", type=float, default=0.03)
    parser.add_argument("--boundary-method", choices=["adami", "holmes"], default="adami", help="Added 2026-09-22 (the development log Entry 124/125) so HOLMES-trained checkpoints can be evaluated under the boundary method they were actually trained with, not silently defaulted to ADAMI.")
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    import environment  # noqa: F401
    import gymnasium as gym
    throwaway = DummyVecEnv([lambda: gym.make("LiquidSloshingRL/SloshingSuppression-v0")])
    normalizer = VecNormalize.load(args.vecnormalize_path, throwaway)
    normalizer.training = False

    env_config = EnvironmentConfig(max_motor_torque_nm=args.torque_limit)
    tank = ClosedLoopSPHTank(
        tank_config=TankConfig(fixed=False), fluid_config=FluidConfig(),
        env_config=env_config, reset_angular_velocity_rad_s=args.reset_velocity,
        artificial_viscosity=args.artificial_viscosity,
        boundary_method=_BOUNDARY_METHOD_VALUES[args.boundary_method],
    )
    print(f"CONFIG: torque_limit={args.torque_limit} reset_velocity={args.reset_velocity} n_steps={args.n_steps} artificial_viscosity={args.artificial_viscosity} boundary_method={args.boundary_method}")

    obs = build_obs(tank.angle, tank.angular_velocity, tank.get_fluid_state())
    t0 = time.perf_counter()
    max_abs_angle = 0.0
    max_abs_ang_vel = 0.0
    try:
        for step in range(args.n_steps):
            model_input = normalizer.normalize_obs(obs)
            action, _ = model.predict(model_input, deterministic=True)
            torque = float(np.clip(action, -1.0, 1.0).reshape(-1)[0]) * args.torque_limit
            tank.set_commanded_torque(torque)
            tank.step(env_config.control_dt)
            state = tank.get_fluid_state()
            obs = build_obs(tank.angle, tank.angular_velocity, state)
            max_abs_angle = max(max_abs_angle, abs(tank.angle))
            max_abs_ang_vel = max(max_abs_ang_vel, abs(tank.angular_velocity))
            if step % args.log_every == 0:
                print(f"  step {step}: angle={tank.angle:.4f} ang_vel={tank.angular_velocity:.4f} torque={torque:.4f} slosh={state.slosh_energy:.5f}")
    except Exception as exc:  # noqa: BLE001 -- diagnostic script, report and exit cleanly
        elapsed = time.perf_counter() - t0
        print(f"RESULT: CRASHED after {step} steps ({elapsed:.1f}s real). max|angle|={max_abs_angle:.4f} max|ang_vel|={max_abs_ang_vel:.4f}")
        print(f"Exception: {exc}")
        return

    elapsed = time.perf_counter() - t0
    print(f"RESULT: COMPLETED all {args.n_steps} steps ({elapsed:.1f}s real). max|angle|={max_abs_angle:.4f} max|ang_vel|={max_abs_ang_vel:.4f} final_angle={tank.angle:.4f}")


if __name__ == "__main__":
    main()
