"""
Unit tests for environment/sloshing_env.py (Phase 8).

No pychrono import anywhere in this environment's dependency chain --
FluidBuilder/PendulumBackend and environment/tank_dynamics.py are both
pure Python (see their own module docstrings), so these tests run in any
sandbox.

Run:
    pytest tests/test_sloshing_env.py -v
"""

import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from config.simulation_config import EnvironmentConfig
from environment.sloshing_env import SloshingSuppressionEnv


def _make_env(**env_config_overrides) -> SloshingSuppressionEnv:
    env_config = EnvironmentConfig(**env_config_overrides) if env_config_overrides else None
    return SloshingSuppressionEnv(env_config=env_config)


def test_gymnasium_registration_resolves():
    env = gym.make("LiquidSloshingRL/SloshingSuppression-v0")
    assert isinstance(env.unwrapped, SloshingSuppressionEnv)
    env.close()


class _StubTankDynamics:
    """Minimal reset/step/angle/angular_velocity double -- see
    environment/chrono_tank_dynamics.py::ChronoTankDynamics for the real
    Phase 10 implementation of this same duck-typed surface; this stub
    exists purely so the injection point itself can be tested without
    pychrono."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self._angle = 0.0
        self._angular_velocity = 0.0

    def reset(self, seed=None) -> None:
        self.reset_calls += 1
        self._angle = 0.0
        self._angular_velocity = 0.0

    def step(self, dt: float, commanded_torque: float) -> None:
        self.step_calls += 1
        self._angle += 0.001  # arbitrary deterministic marker, not physical

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def angular_velocity(self) -> float:
        return self._angular_velocity


def test_injected_tank_dynamics_is_used_instead_of_the_default_surrogate():
    """Phase 10: confirms the `tank_dynamics=` constructor argument
    actually reaches step()/reset(), the same injection contract
    `fluid_backend=` already had -- this is what lets
    environment/chrono_tank_dynamics.py::ChronoTankDynamics be dropped in
    without this file changing."""
    stub = _StubTankDynamics()
    env = SloshingSuppressionEnv(tank_dynamics=stub)

    env.reset(seed=0)
    assert stub.reset_calls == 1

    env.step(np.array([0.0], dtype=np.float32))
    assert stub.step_calls == env._env_config.substeps_per_control_step


def test_passes_gymnasium_env_checker():
    # skip_render_check=True: render() is a documented no-op (see
    # SloshingSuppressionEnv.render's own docstring), not a real render
    # pipeline the checker's image-shape assertions apply to.
    check_env(_make_env().unwrapped, skip_render_check=True)


def test_reset_returns_observation_matching_space():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "slosh_energy" in info


def test_reset_is_reproducible_with_same_seed():
    env_a = _make_env()
    env_b = _make_env()
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_allclose(obs_a, obs_b)


def test_step_returns_well_formed_tuple():
    env = _make_env()
    env.reset(seed=0)
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert math.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "tank_angle" in info


def test_episode_truncates_at_max_episode_steps():
    env = _make_env(max_episode_seconds=0.1, control_dt=0.02)  # 5 steps
    env.reset(seed=0)
    zero_action = np.array([0.0], dtype=np.float32)

    truncated = False
    terminated = False
    steps_taken = 0
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(zero_action)
        steps_taken += 1
        if terminated or truncated:
            break

    assert truncated and not terminated
    assert steps_taken == env._env_config.max_episode_steps


def test_episode_terminates_when_tank_angle_exceeds_limit():
    # A tiny angle limit combined with max-magnitude (normalized) torque
    # should tip the surrogate tank past the safety bound almost
    # immediately.
    env = _make_env(max_tank_angle_rad=0.01, max_motor_torque_nm=50.0, pivot_damping_coefficient=0.0)
    env.reset(seed=0)
    max_torque_action = np.array([1.0], dtype=np.float32)

    terminated = False
    for _ in range(env._env_config.max_episode_steps):
        _, _, terminated, truncated, _ = env.step(max_torque_action)
        if terminated:
            break

    assert terminated


def test_action_is_clipped_to_torque_limit():
    # Actions outside [-1, 1] should clip to the action space bound before
    # being scaled to physical torque, so an out-of-range action and the
    # boundary action itself must produce identical dynamics.
    env = _make_env(max_motor_torque_nm=1.0, pivot_damping_coefficient=0.0)
    env.reset(seed=0)
    huge_action = np.array([1000.0], dtype=np.float32)
    clipped_action = np.array([1.0], dtype=np.float32)

    obs_after_huge, *_ = env.step(huge_action)

    env2 = _make_env(max_motor_torque_nm=1.0, pivot_damping_coefficient=0.0)
    env2.reset(seed=0)
    obs_after_clipped, *_ = env2.step(clipped_action)

    np.testing.assert_allclose(obs_after_huge, obs_after_clipped, rtol=1e-5)


def test_zero_action_leaves_undisturbed_fluid_at_rest():
    env = _make_env(reset_disturbance_impulse_range=(0.0, 0.0))
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(0.0)  # slosh_energy

    zero_action = np.array([0.0], dtype=np.float32)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(zero_action)

    assert obs[-1] == pytest.approx(0.0, abs=1e-9)
    assert reward == pytest.approx(0.0, abs=1e-9)
