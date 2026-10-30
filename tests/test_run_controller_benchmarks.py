"""Tests for benchmarks/run_controller_benchmarks.py -- pure-Python
(this project's PyChrono-free surrogate env), no PyChrono needed."""

import numpy as np

from benchmarks.run_controller_benchmarks import compute_metrics, run_episode
from config.simulation_config import EnvironmentConfig
from environment.sloshing_env import SloshingSuppressionEnv


def test_run_episode_arrays_are_consistent_length():
    env_config = EnvironmentConfig(max_episode_seconds=1.0)
    env = SloshingSuppressionEnv(env_config=env_config)
    history = run_episode(env, lambda obs: np.array([0.0], dtype=np.float32), seed=0, max_motor_torque_nm=2.0)
    env.close()

    n = len(history["time"])
    assert n == 51  # 1.0s / 0.02s control_dt + 1 (post-reset state)
    for key in ("theta", "slosh_energy", "torque", "angular_velocity"):
        assert len(history[key]) == n


def test_run_episode_first_entry_is_the_post_reset_state():
    env_config = EnvironmentConfig(max_episode_seconds=1.0, reset_disturbance_impulse_range=(0.0, 0.0))
    env = SloshingSuppressionEnv(env_config=env_config)
    history = run_episode(env, lambda obs: np.array([0.0], dtype=np.float32), seed=0, max_motor_torque_nm=2.0)
    env.close()

    assert history["time"][0] == 0.0
    assert history["theta"][0] == 0.0
    assert history["torque"][0] == 0.0


def test_passive_controller_applies_zero_torque_throughout():
    env_config = EnvironmentConfig(max_episode_seconds=1.0)
    env = SloshingSuppressionEnv(env_config=env_config)
    history = run_episode(env, lambda obs: np.array([0.0], dtype=np.float32), seed=0, max_motor_torque_nm=2.0)
    env.close()
    assert np.all(history["torque"] == 0.0)


def test_full_torque_command_is_scaled_by_max_motor_torque():
    env_config = EnvironmentConfig(max_episode_seconds=0.1)
    env = SloshingSuppressionEnv(env_config=env_config)
    history = run_episode(env, lambda obs: np.array([1.0], dtype=np.float32), seed=0, max_motor_torque_nm=3.5)
    env.close()
    # Every commanded step (excluding the index-0 pre-action placeholder)
    # should be exactly the full torque limit.
    assert np.all(history["torque"][1:] == 3.5)


def test_compute_metrics_settling_time_detects_decay_below_threshold():
    time = np.linspace(0, 10, 501)
    # Energy starts at 1.0 and decays to well below 2% within the window.
    energy = 1.0 * np.exp(-time)
    torque = np.zeros_like(time)
    angular_velocity = np.zeros_like(time)
    metrics = compute_metrics(
        {"time": time, "slosh_energy": energy, "torque": torque, "angular_velocity": angular_velocity},
        control_dt=0.02,
    )
    assert 0.0 < metrics["settling_time_s"] < 10.0
    assert metrics["final_slosh_energy_j"] == energy[-1]


def test_compute_metrics_never_settling_reports_full_duration():
    time = np.linspace(0, 10, 501)
    energy = np.full_like(time, 1.0)  # never decays
    torque = np.zeros_like(time)
    angular_velocity = np.zeros_like(time)
    metrics = compute_metrics(
        {"time": time, "slosh_energy": energy, "torque": torque, "angular_velocity": angular_velocity},
        control_dt=0.02,
    )
    assert metrics["settling_time_s"] == time[-1]


def test_compute_metrics_motor_energy_is_nonnegative():
    time = np.linspace(0, 1, 51)
    energy = np.linspace(1.0, 0.0, 51)
    torque = np.sin(time)
    angular_velocity = np.cos(time)
    metrics = compute_metrics(
        {"time": time, "slosh_energy": energy, "torque": torque, "angular_velocity": angular_velocity},
        control_dt=0.02,
    )
    assert metrics["motor_energy_j"] >= 0.0
    assert metrics["torque_total_variation"] >= 0.0
