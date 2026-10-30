"""Fair, hardware-realistic controller benchmark -- Passive / PID / LQR / PPO,
all restricted to environment/observable_sloshing_env.py::ObservableSloshingEnv's
sensor-only observation (tank IMU + differential wall-pressure taps), unlike
benchmarks/run_controller_benchmarks.py.

Real motivation: that base benchmark's LQR row recovers the pendulum's phi/
phi_dot from SloshingSuppressionEnv's PRIVILEGED fluid_com observation --
information no real hardware controller would have -- while its PID row
only ever sees tank_angle/rate, with zero fluid information at all. That
comparison mostly measures "privileged state vs. none," not "classical
control vs. RL" -- it does not, by itself, support "why RL over classical
control" for this task, since LQR-with-privileged-state already handles it
cleanly. This script puts PID, LQR, and a real PPO policy trained against
ObservableSloshingEnv on the IDENTICAL hardware-realistic observation, so
the comparison is actually fair.

LQR's phi/phi_dot recovery here inverts
environment/observable_sloshing_env.py::wall_pressures() analytically
(delta_pressure is exactly linear in the fluid's lateral COM position --
the dynamic-head term cancels between the two walls) rather than reading
fluid_com directly -- see
control/lqr_controller.py::SloshLQRController.get_action_from_pressure_observation().

Reuses benchmarks/run_controller_benchmarks.py's run_episode()/
run_ppo_episode()/compute_metrics() unchanged (they're already environment-
and controller-agnostic) rather than duplicating that logic.

Usage:
    python -m benchmarks.run_observable_controller_benchmarks
    python -m benchmarks.run_observable_controller_benchmarks --ppo-model models/ppo_sloshing_observable
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

import numpy as np

from benchmarks.run_controller_benchmarks import compute_metrics, run_episode, run_ppo_episode
from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig
from control.lqr_controller import SloshLQRController
from control.pid_controller import SloshPIDController
from environment.observable_sloshing_env import ObservableSloshingEnv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_benchmark_campaign(
    seed: int = 0,
    ppo_model_path: Optional[str] = "models/ppo_sloshing_observable",
    ppo_vecnormalize_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    tank_config = TankConfig(fixed=False)
    fluid_config = FluidConfig()
    env_config = EnvironmentConfig()

    def env_fn() -> ObservableSloshingEnv:
        return ObservableSloshingEnv(
            tank_config=tank_config, fluid_config=fluid_config, env_config=env_config
        )

    torque_limit = env_config.max_motor_torque_nm
    histories: Dict[str, Dict[str, np.ndarray]] = {}

    logger.info("=" * 70)
    logger.info(
        "OBSERVABLE (SENSOR-ONLY) CONTROLLER BENCHMARK -- ObservableSloshingEnv, seed=%d",
        seed,
    )
    logger.info("=" * 70)

    # 1. Passive (uncontrolled) -- identical baseline meaning as the base
    # benchmark's passive row (same physics, only the observation differs,
    # and passive never reads it).
    env = env_fn()
    histories["passive"] = run_episode(
        env, lambda obs: np.array([0.0], dtype=np.float32), seed, torque_limit
    )
    env.close()

    # 2. PID -- unchanged from the base benchmark: obs[0]/obs[1] are still
    # tank_angle/tank_angular_velocity in ObservableSloshingEnv's
    # observation (same first two entries by design), so this is the SAME
    # controller seeing the SAME thing it always did. Included here purely
    # so all three rows come from one script/run, not because PID's own
    # inputs changed.
    pid = SloshPIDController(torque_limit=torque_limit)
    env = env_fn()

    def pid_controller(obs: np.ndarray) -> np.ndarray:
        torque = pid.get_action(current_angle=float(obs[0]), target_angle=0.0, dt=env_config.control_dt)
        return np.array([torque / torque_limit], dtype=np.float32)

    pid.reset()
    histories["pid"] = run_episode(env, pid_controller, seed, torque_limit)
    env.close()

    # 3. LQR -- the FAIR row: recovers phi/phi_dot from the pressure-tap
    # observation only, via an exact analytic inversion of wall_pressures()
    # (see control/lqr_controller.py), never from privileged fluid_com.
    lqr = SloshLQRController(tank_config=tank_config, fluid_config=fluid_config, env_config=env_config)
    env = env_fn()

    def lqr_controller(obs: np.ndarray) -> np.ndarray:
        torque = lqr.get_action_from_pressure_observation(obs)
        return np.array([torque / torque_limit], dtype=np.float32)

    histories["lqr"] = run_episode(env, lqr_controller, seed, torque_limit)
    env.close()

    # 4. Trained PPO policy -- models/ppo_sloshing_observable.zip, the
    # policy actually trained against ObservableSloshingEnv's own sensor
    # observation (NOT models/ppo_sloshing, which was trained against the
    # privileged base env and would be evaluated out-of-distribution here).
    if ppo_model_path is not None:
        try:
            if ppo_vecnormalize_path is None:
                candidate = ppo_model_path.rstrip(".zip") + "_vecnormalize.pkl"
                ppo_vecnormalize_path = candidate if os.path.exists(candidate) else None
            histories["ppo"] = run_ppo_episode(
                ppo_model_path, ppo_vecnormalize_path, env_fn, seed, env_config.control_dt
            )
        except ImportError:
            logger.warning(
                "stable-baselines3 not installed -- skipping the trained-PPO row. "
                "Run this on a machine with the real conda env (see the project notes) to include it."
            )

    metrics = {name: compute_metrics(hist, env_config.control_dt) for name, hist in histories.items()}

    labels = {
        "passive": "Uncontrolled Passive",
        "pid": "Classical Tuned PID (sensor-only, unchanged)",
        "lqr": "LQR (FAIR: pressure-tap observation only)",
        "ppo": f"Trained PPO Policy, observable ({ppo_model_path})",
    }
    for name, m in metrics.items():
        logger.info(
            "%-46s | Settling: %5.2fs | Motor Energy: %6.3f J | Final Slosh: %.6f J | Torque TV: %5.2f",
            labels.get(name, name), m["settling_time_s"], m["motor_energy_j"],
            m["final_slosh_energy_j"], m["torque_total_variation"],
        )

    csv_path = os.path.join(output_dir, "observable_benchmark_summary.csv")
    metric_keys = ["settling_time_s", "final_slosh_energy_j", "motor_energy_j", "torque_total_variation"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("controller," + ",".join(metric_keys) + "\n")
        for name, m in metrics.items():
            f.write(labels.get(name, name) + "," + ",".join(f"{m[k]:.6f}" for k in metric_keys) + "\n")
    logger.info("Saved %s", csv_path)

    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)
        colors = {"passive": "#64748b", "pid": "#0284c7", "lqr": "#059669", "ppo": "#dc2626"}

        ax = axs[0, 0]
        for name, hist in histories.items():
            ax.plot(hist["time"], np.degrees(hist["theta"]), color=colors.get(name), label=labels.get(name, name))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tank Tilt (deg)")
        ax.set_title("(a) Tank Angular Excursion")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.4, linestyle="--")

        ax = axs[0, 1]
        for name, hist in histories.items():
            ax.semilogy(hist["time"], np.maximum(1e-8, hist["slosh_energy"]), color=colors.get(name), label=labels.get(name, name))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Slosh Energy (J) [log]")
        ax.set_title("(b) Slosh Energy Decay")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.4, linestyle="--")

        ax = axs[1, 0]
        for name, hist in histories.items():
            if name == "passive":
                continue
            ax.plot(hist["time"], hist["torque"], color=colors.get(name), label=labels.get(name, name))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Commanded Torque (N*m)")
        ax.set_title("(c) Control Actuation")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.4, linestyle="--")

        ax = axs[1, 1]
        for name, m in metrics.items():
            ax.scatter(m["motor_energy_j"], m["final_slosh_energy_j"], color=colors.get(name), s=100, label=labels.get(name, name))
        ax.set_xlabel("Motor Energy (J)")
        ax.set_ylabel("Final Slosh Energy (J)")
        ax.set_title("(d) Pareto: Effort vs. Suppression")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.4, linestyle="--")

        plt.suptitle("Fair Controller Benchmark -- sensor-only ObservableSloshingEnv", fontweight="bold")
        plt.tight_layout()
        png_path = os.path.join(output_dir, "observable_benchmark_comparison.png")
        plt.savefig(png_path)
        plt.close()
        logger.info("Saved %s", png_path)
    except ImportError:
        logger.warning("matplotlib not installed -- CSV written, plot skipped (see the project notes).")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo-model", default="models/ppo_sloshing_observable")
    parser.add_argument("--ppo-vecnormalize", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark_campaign(
        seed=args.seed, ppo_model_path=args.ppo_model, ppo_vecnormalize_path=args.ppo_vecnormalize
    )
