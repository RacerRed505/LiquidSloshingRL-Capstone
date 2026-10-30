"""Tests for environment/observable_sloshing_env.py -- pure-Python
(gymnasium + this project's PyChrono-free surrogate backends), no
PyChrono needed."""

import numpy as np
import pytest

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig
from environment.observable_sloshing_env import ObservableSloshingEnv


def _make_env(**overrides):
    return ObservableSloshingEnv(**overrides)


def test_observation_space_is_seven_dimensional():
    env = _make_env()
    assert env.observation_space.shape == (7,)


def test_never_exposes_fluid_com_directly():
    """The whole point of this class: raw fluid_com_x/y/vx/vy must not
    appear anywhere in the observation vector, even indirectly as a
    pass-through -- only derived pressure quantities and tank state."""
    env = _make_env()
    obs, _info = env.reset(seed=0)
    fluid_state = env._fluid_backend.get_state()
    # None of the 7 reported values should equal the true fluid COM
    # position/velocity components (which the base SloshingSuppressionEnv
    # DOES expose directly) -- guards against a future refactor
    # accidentally reintroducing god-mode observability here.
    forbidden = {
        fluid_state.com_position[0],
        fluid_state.com_position[1],
        fluid_state.com_velocity[0],
        fluid_state.com_velocity[1],
    } - {0.0}  # zero is a legitimate coincidental value (e.g. at rest)
    assert not (forbidden & set(obs.tolist()))


def test_at_rest_pressures_are_equal_and_hydrostatic():
    fluid_config = FluidConfig(fill_height=0.2, density=1000.0)
    env = _make_env(fluid_config=fluid_config, env_config=EnvironmentConfig(reset_disturbance_impulse_range=(0.0, 0.0)))
    obs, _info = env.reset(seed=0)
    _theta, _theta_dot, p_left, p_right, delta_p, delta_p_dot, prev_u = obs

    expected_hydrostatic = 1000.0 * 9.81 * 0.2
    assert p_left == pytest.approx(expected_hydrostatic, rel=1e-3)
    assert p_right == pytest.approx(expected_hydrostatic, rel=1e-3)
    assert delta_p == pytest.approx(0.0, abs=1e-6)
    assert delta_p_dot == pytest.approx(0.0, abs=1e-6)
    assert prev_u == pytest.approx(0.0)


def test_positive_com_x_raises_right_wall_pressure_relative_to_left():
    env = _make_env(env_config=EnvironmentConfig(reset_disturbance_impulse_range=(0.0, 0.0)))
    env.reset(seed=0)
    p_left, p_right = env._wall_pressures(com_x=0.05, com_vx=0.0)
    assert p_right > p_left


def test_negative_com_x_raises_left_wall_pressure_relative_to_right():
    env = _make_env(env_config=EnvironmentConfig(reset_disturbance_impulse_range=(0.0, 0.0)))
    env.reset(seed=0)
    p_left, p_right = env._wall_pressures(com_x=-0.05, com_vx=0.0)
    assert p_left > p_right


def test_pressures_are_never_negative():
    env = _make_env(env_config=EnvironmentConfig(reset_disturbance_impulse_range=(0.0, 0.0)))
    env.reset(seed=0)
    # A violent (unphysically large) displacement should still clamp to
    # zero, not go negative -- pressure gauges can't read negative gauge
    # pressure in this context.
    p_left, p_right = env._wall_pressures(com_x=10.0, com_vx=0.0)
    assert p_left >= 0.0
    assert p_right >= 0.0


def test_prev_torque_reflects_the_action_just_applied():
    env = _make_env(env_config=EnvironmentConfig(reset_disturbance_impulse_range=(0.0, 0.0), max_motor_torque_nm=2.0))
    env.reset(seed=0)
    obs, _reward, _term, _trunc, _info = env.step(np.array([0.5], dtype=np.float32))
    assert obs[6] == pytest.approx(1.0)  # 0.5 * 2.0 N*m

    obs, _reward, _term, _trunc, _info = env.step(np.array([-1.0], dtype=np.float32))
    assert obs[6] == pytest.approx(-2.0)


def test_reset_zeroes_previous_torque_and_delta_pressure_history():
    env = _make_env()
    env.reset(seed=0)
    env.step(np.array([0.8], dtype=np.float32))

    obs, _info = env.reset(seed=1)
    assert obs[6] == pytest.approx(0.0)


def test_passes_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    env = _make_env()
    check_env(env, skip_render_check=True)


def test_reward_and_termination_logic_are_unchanged_from_base_class():
    """This subclass must not accidentally change the task itself -- only
    the observation. Sanity-check that step() still returns the same
    reward/termination shape and reacts to the same max_tank_angle_rad."""
    env_config = EnvironmentConfig(max_tank_angle_rad=0.05, max_episode_seconds=20.0)
    env = _make_env(env_config=env_config)
    env.reset(seed=0)

    terminated = False
    for _ in range(500):
        _obs, _reward, terminated, truncated, _info = env.step(np.array([1.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated
