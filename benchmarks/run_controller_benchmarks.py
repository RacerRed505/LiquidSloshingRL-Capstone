"""Controller benchmark suite -- Passive / PID / LQR / trained PPO, all run
against the SAME real environment/sloshing_env.py::SloshingSuppressionEnv,
same seed (so every controller starts from an identical fluid
disturbance), same reward/physics config.

This intentionally replaces (rather than adapts) the benchmark script in
an external reference package (see the project notes' log of that
package's review): that script benchmarked PID/LQR/a hand-tuned heuristic
against an independently-invented toy tank+pendulum ODE that never
imported anything from this project and never ran the actual trained
PPO policy -- so its numbers characterized a different problem, not this
one. Every number this script produces comes from the real Gym env this
project actually trains against, and the PPO row (when a model is given)
is the actual trained policy, not a stand-in.

Usage:
    python -m benchmarks.run_controller_benchmarks
    python -m benchmarks.run_controller_benchmarks --ppo-model models/ppo_sloshing
    python -m benchmarks.run_controller_benchmarks --seed 3 --n-seeds 5
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig
from control.lqr_controller import SloshLQRController
from control.pid_controller import SloshPIDController
from environment.sloshing_env import SloshingSuppressionEnv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ControllerFn = Callable[[np.ndarray], np.ndarray]


def _total_energy(info: Dict) -> float:
    """Kinetic + potential slosh energy from an env info dict, falling back
    to the kinetic-only value for backends that report no potential term."""
    total = info.get("slosh_total_energy")
    return float(info["slosh_energy"] if total is None else total)


def _settling_time(energy: np.ndarray, time: np.ndarray, control_dt: float) -> float:
    early_window = max(1, int(1.0 / control_dt))
    initial_peak = max(1e-8, float(np.max(energy[:early_window])))
    threshold = 0.02 * initial_peak
    for i in range(len(energy) - 1, 0, -1):
        if energy[i] > threshold:
            return float(time[min(len(time) - 1, i + 1)])
    return 0.0


def run_episode(
    env: SloshingSuppressionEnv,
    controller_fn: ControllerFn,
    seed: int,
    max_motor_torque_nm: float,
) -> Dict[str, np.ndarray]:
    """Runs one deterministic episode, returning per-step telemetry arrays.

    Arrays are all `num_steps + 1` long: index 0 is the post-reset state
    before any action (torque=0.0 placeholder there, since nothing has
    been commanded yet), index i>0 is the state that resulted from the
    i-th action.
    """
    obs, info = env.reset(seed=seed)
    control_dt = env._env_config.control_dt

    times: List[float] = [0.0]
    angles: List[float] = [info["tank_angle"]]
    slosh_energies: List[float] = [info["slosh_energy"]]
    total_energies: List[float] = [_total_energy(info)]
    torques: List[float] = [0.0]
    angular_velocities: List[float] = [info.get("tank_angular_velocity", 0.0)]

    t = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller_fn(obs)
        physical_torque = float(np.clip(action, -1.0, 1.0)[0]) * max_motor_torque_nm
        obs, _reward, terminated, truncated, info = env.step(action)
        t += control_dt

        times.append(t)
        angles.append(info["tank_angle"])
        slosh_energies.append(info["slosh_energy"])
        total_energies.append(_total_energy(info))
        torques.append(physical_torque)
        angular_velocities.append(info.get("tank_angular_velocity", 0.0))

    return {
        "time": np.array(times),
        "theta": np.array(angles),
        "slosh_energy": np.array(slosh_energies),
        "total_slosh_energy": np.array(total_energies),
        "torque": np.array(torques),
        "angular_velocity": np.array(angular_velocities),
    }


def run_ppo_episode(
    model_path: str,
    vecnormalize_path: Optional[str],
    env_fn: Callable[[], SloshingSuppressionEnv],
    seed: int,
    control_dt: float,
) -> Dict[str, np.ndarray]:
    """Runs one deterministic episode with the actual trained PPO policy.

    Reuses evaluation/evaluate_policy.py's own VecNormalize-handling
    pattern exactly (a DummyVecEnv-wrapped, frozen-statistics
    VecNormalize) -- a policy trained with observation normalization
    fed raw observations directly would be evaluated out-of-distribution,
    which would make this comparison meaningless, not just less fair.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(model_path)
    vec_env = DummyVecEnv([env_fn])
    if vecnormalize_path is not None and os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # DummyVecEnv doesn't expose per-env seeding through reset() directly
    # in older Gymnasium/SB3 versions -- set it on the underlying env
    # before the vectorized reset, matching evaluate_policy.py's own
    # non-vectorized seeding convention as closely as the VecEnv API allows.
    vec_env.seed(seed)
    obs = vec_env.reset()
    # vec_env.reset() doesn't return an info dict (older VecEnv API) --
    # read the pre-action tank state straight off the underlying env
    # instead, matching run_episode()'s own index-0-is-post-reset-state
    # convention exactly.
    underlying_env = vec_env.envs[0]
    torque_limit = underlying_env._env_config.max_motor_torque_nm

    times: List[float] = [0.0]
    angles: List[float] = [underlying_env._tank_dynamics.angle]
    initial_fluid_state = underlying_env._fluid_backend.get_state()
    slosh_energies: List[float] = [initial_fluid_state.slosh_energy]
    total_energies: List[float] = [
        initial_fluid_state.slosh_total_energy
        if initial_fluid_state.slosh_total_energy is not None
        else initial_fluid_state.slosh_energy
    ]
    torques: List[float] = [0.0]
    angular_velocities: List[float] = [underlying_env._tank_dynamics.angular_velocity]

    t = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        physical_torque = float(np.clip(action[0], -1.0, 1.0)[0]) * torque_limit
        obs, _reward, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        info = infos[0]

        t += control_dt
        times.append(t)
        angles.append(info["tank_angle"])
        slosh_energies.append(info["slosh_energy"])
        total_energies.append(_total_energy(info))
        torques.append(physical_torque)
        angular_velocities.append(info.get("tank_angular_velocity", 0.0))

    return {
        "time": np.array(times),
        "theta": np.array(angles),
        "slosh_energy": np.array(slosh_energies),
        "total_slosh_energy": np.array(total_energies),
        "torque": np.array(torques),
        "angular_velocity": np.array(angular_velocities),
    }


def compute_metrics(history: Dict[str, np.ndarray], control_dt: float) -> Dict[str, float]:
    """Real quantitative metrics -- same definitions as this project's own
    scripts/validate_real_dynamics.py-style reporting (return-free here,
    since these controllers don't optimize the RL reward directly): a
    2%-of-initial-peak settling threshold, final residual slosh energy,
    integrated |torque * angular_velocity| as a motor-work proxy, and
    total variation of torque as a chatter/smoothness proxy.
    """
    energy = history["slosh_energy"]
    torque = history["torque"]
    angular_velocity = history["angular_velocity"]
    time = history["time"]

    settling_time = _settling_time(energy, time, control_dt)
    # Kinetic + potential. Falls back to the kinetic-only trace for
    # histories recorded before the potential term existed.
    total_energy = history.get("total_slosh_energy", energy)

    return {
        "settling_time_s": settling_time,
        "final_slosh_energy_j": float(energy[-1]),
        "total_energy_settling_time_s": _settling_time(total_energy, time, control_dt),
        "final_total_slosh_energy_j": float(total_energy[-1]),
        "motor_energy_j": float(np.sum(np.abs(torque * angular_velocity)) * control_dt),
        "torque_total_variation": float(np.sum(np.abs(np.diff(torque)))),
    }


def run_benchmark_campaign(
    seed: int = 0,
    ppo_model_path: Optional[str] = None,
    ppo_vecnormalize_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    tank_config = TankConfig(fixed=False)
    fluid_config = FluidConfig()
    env_config = EnvironmentConfig()

    def env_fn() -> SloshingSuppressionEnv:
        return SloshingSuppressionEnv(
            tank_config=tank_config, fluid_config=fluid_config, env_config=env_config
        )

    torque_limit = env_config.max_motor_torque_nm
    histories: Dict[str, Dict[str, np.ndarray]] = {}

    logger.info("=" * 70)
    logger.info("CONTROLLER BENCHMARK -- real environment/sloshing_env.py, seed=%d", seed)
    logger.info("=" * 70)

    # 1. Passive (uncontrolled)
    env = env_fn()
    histories["passive"] = run_episode(
        env, lambda obs: np.array([0.0], dtype=np.float32), seed, torque_limit
    )
    env.close()

    # 2. PID
    # CONFIRMED REAL FINDING (not a bug in this script): with this
    # project's real EnvironmentConfig.reset_disturbance_impulse_range,
    # the reset-time disturbance is an impulse applied DIRECTLY TO THE
    # FLUID (see SloshingSuppressionEnv.reset() ->
    # FluidBackend.apply_disturbance()), never to the tank -- so the tank
    # starts, and stays, at angle=0 until something actually torques it.
    # A PID that only tracks tank_angle -> 0 therefore sees zero error for
    # the ENTIRE episode and applies exactly zero torque throughout --
    # verified directly (max|torque| == 0.0, max|theta| == 0.0 over a full
    # rollout), not assumed. Its slosh-energy trace is consequently
    # identical to the uncontrolled passive baseline's. This is a
    # legitimate, informative result, not a broken baseline: it shows a
    # naive angle-only classical controller cannot even see this task's
    # actual disturbance, which is exactly why LQR (which acts on the
    # pendulum's own phi/phi_dot state, recovered from the observation)
    # and the trained RL policy (which observes fluid_com/velocity
    # directly) are the meaningful comparisons here.
    pid = SloshPIDController(torque_limit=torque_limit)
    env = env_fn()

    def pid_controller(obs: np.ndarray) -> np.ndarray:
        torque = pid.get_action(current_angle=float(obs[0]), target_angle=0.0, dt=env_config.control_dt)
        return np.array([torque / torque_limit], dtype=np.float32)

    pid.reset()
    histories["pid"] = run_episode(env, pid_controller, seed, torque_limit)
    env.close()

    # 3. LQR (linearized from this project's own real config -- see
    # control/lqr_controller.py's own docstring)
    lqr = SloshLQRController(tank_config=tank_config, fluid_config=fluid_config, env_config=env_config)
    env = env_fn()

    def lqr_controller(obs: np.ndarray) -> np.ndarray:
        torque = lqr.get_action_from_observation(obs)
        return np.array([torque / torque_limit], dtype=np.float32)

    histories["lqr"] = run_episode(env, lqr_controller, seed, torque_limit)
    env.close()

    # 4. Trained PPO policy, if given -- the actual model, not a stand-in.
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
        "pid": "Classical Tuned PID",
        "lqr": "Linear Quadratic Regulator (LQR)",
        # Real bug fixed 2026-09-16: this used to hardcode "models/ppo_sloshing"
        # regardless of what --ppo-model actually pointed at, so any other
        # checkpoint's results were silently mislabeled as the surrogate-
        # trained baseline. Now reflects the real path passed in.
        "ppo": f"Trained PPO Policy ({ppo_model_path})",
    }
    for name, m in metrics.items():
        logger.info(
            "%-38s | Settling: %5.2fs (total-energy: %5.2fs) | Motor Energy: %6.3f J | "
            "Final Slosh KE: %.6f J | Final Slosh KE+PE: %.6f J | Torque TV: %5.2f",
            labels.get(name, name), m["settling_time_s"], m["total_energy_settling_time_s"],
            m["motor_energy_j"], m["final_slosh_energy_j"], m["final_total_slosh_energy_j"],
            m["torque_total_variation"],
        )

    # Write CSV
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    metric_keys = [
        "settling_time_s", "final_slosh_energy_j",
        "total_energy_settling_time_s", "final_total_slosh_energy_j",
        "motor_energy_j", "torque_total_variation",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("controller," + ",".join(metric_keys) + "\n")
        for name, m in metrics.items():
            f.write(labels.get(name, name) + "," + ",".join(f"{m[k]:.6f}" for k in metric_keys) + "\n")
    logger.info("Saved %s", csv_path)

    # Optional plot (matplotlib is optional project-wide, see the project notes'
    # known-risks note -- CSV is always written regardless).
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
        ax.legend(fontsize=7)
        ax.grid(alpha=0.4, linestyle="--")

        ax = axs[0, 1]
        for name, hist in histories.items():
            ax.semilogy(hist["time"], np.maximum(1e-8, hist["slosh_energy"]), color=colors.get(name), label=labels.get(name, name))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Slosh Energy (J) [log]")
        ax.set_title("(b) Slosh Energy Decay")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.4, linestyle="--")

        ax = axs[1, 0]
        for name, hist in histories.items():
            if name == "passive":
                continue
            ax.plot(hist["time"], hist["torque"], color=colors.get(name), label=labels.get(name, name))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Commanded Torque (N*m)")
        ax.set_title("(c) Control Actuation")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.4, linestyle="--")

        ax = axs[1, 1]
        for name, m in metrics.items():
            ax.scatter(m["motor_energy_j"], m["final_slosh_energy_j"], color=colors.get(name), s=100, label=labels.get(name, name))
        ax.set_xlabel("Motor Energy (J)")
        ax.set_ylabel("Final Slosh Energy (J)")
        ax.set_title("(d) Pareto: Effort vs. Suppression")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.4, linestyle="--")

        plt.suptitle("Controller Benchmark -- real SloshingSuppressionEnv", fontweight="bold")
        plt.tight_layout()
        png_path = os.path.join(output_dir, "benchmark_comparison.png")
        plt.savefig(png_path)
        plt.close()
        logger.info("Saved %s", png_path)
    except ImportError:
        logger.warning("matplotlib not installed -- CSV written, plot skipped (see the project notes).")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ppo-model",
        default=None,
        help="Path to a trained PPO model (e.g. models/ppo_sloshing) to include as a real comparison row.",
    )
    parser.add_argument("--ppo-vecnormalize", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark_campaign(
        seed=args.seed, ppo_model_path=args.ppo_model, ppo_vecnormalize_path=args.ppo_vecnormalize
    )
