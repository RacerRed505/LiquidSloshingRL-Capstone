"""Tests for control/lqr_controller.py -- pure-Python (scipy/numpy), no
PyChrono needed. Verifies the linearization matches this project's real
config-derived parameters, not just that the LQR math is internally
self-consistent."""

import math

import numpy as np
import pytest

from config.simulation_config import EnvironmentConfig, FluidConfig, PivotConfig, TankConfig, WorldConfig
from control.lqr_controller import SloshLQRController, build_state_space
from environment.observable_sloshing_env import wall_pressures
from environment.tank_dynamics import tank_moment_of_inertia_about_pivot
from utilities.housner_utils import compute_housner_parameters


def test_state_space_uses_the_projects_real_tank_inertia():
    tank_config = TankConfig(fixed=False)
    A, _B = build_state_space(tank_config=tank_config)
    expected_inertia = tank_moment_of_inertia_about_pivot(tank_config, PivotConfig())
    # -b/I is the (theta_dot, theta_dot) entry; recover I from it and the
    # known EnvironmentConfig default damping, and confirm it matches the
    # inertia this project's own builder computes independently.
    env_config = EnvironmentConfig()
    implied_inertia = -env_config.pivot_damping_coefficient / A[1, 1]
    assert implied_inertia == pytest.approx(expected_inertia, rel=1e-9)


def test_state_space_natural_frequency_matches_housner_parameters():
    fluid_config = FluidConfig()
    tank_config = TankConfig(fixed=False)
    A, _B = build_state_space(tank_config=tank_config, fluid_config=fluid_config)
    housner = compute_housner_parameters(
        tank_length=tank_config.length,
        tank_breadth=tank_config.width,
        fill_height=fluid_config.fill_height,
        fluid_density=fluid_config.density,
        gravity=9.81,
    )
    # A[3, 2] = -g / L_p; compare against the natural frequency identity
    # g / L_p = omega_n^2 (exactly how utilities/housner_utils.py derives
    # pendulum_length in the first place).
    implied_omega_sq = -A[3, 2]
    assert implied_omega_sq == pytest.approx(housner.natural_frequency_rad_s ** 2, rel=1e-9)


def test_system_is_controllable_at_project_defaults():
    controller = SloshLQRController()
    assert controller.is_controllable()


def test_closed_loop_poles_are_all_stable():
    controller = SloshLQRController()
    poles = controller.poles()
    assert all(pole.real < 0.0 for pole in poles)


def test_action_is_clipped_to_torque_limit():
    env_config = EnvironmentConfig(max_motor_torque_nm=2.0)
    controller = SloshLQRController(env_config=env_config)
    # A large disturbance should saturate the actuator, not exceed it.
    torque = controller.get_action([1.0, 5.0, 1.0, 5.0])
    assert abs(torque) <= 2.0 + 1e-9


def test_zero_state_produces_zero_action():
    controller = SloshLQRController()
    assert controller.get_action([0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_get_action_from_observation_matches_manual_phi_recovery():
    controller = SloshLQRController()
    length = controller._housner.pendulum_length
    phi = 0.05
    theta, theta_dot = 0.1, -0.2
    com_x = length * math.sin(phi)
    com_vx = length * math.cos(phi) * 1.3  # arbitrary phi_dot = 1.3
    observation = [theta, theta_dot, com_x, 0.0, com_vx, 0.0, 0.0]

    torque_from_obs = controller.get_action_from_observation(observation)
    torque_from_state = controller.get_action([theta, theta_dot, phi, 1.3])
    assert torque_from_obs == pytest.approx(torque_from_state, rel=1e-6)


def test_get_action_from_pressure_observation_matches_privileged_recovery():
    """The pressure-based recovery must be an EXACT inversion of
    wall_pressures(), not an approximation on top of the already-exact
    privileged recovery -- so, for the same underlying fluid state, both
    of SloshLQRController's observation wrappers should command the same
    torque, even though one only ever sees pressures.
    """
    tank_config = TankConfig(fixed=False)
    fluid_config = FluidConfig()
    world_config = WorldConfig()
    controller = SloshLQRController(tank_config=tank_config, fluid_config=fluid_config, world_config=world_config)

    length = controller._housner.pendulum_length
    phi = -0.04
    theta, theta_dot = -0.07, 0.15
    com_x = length * math.sin(phi)
    com_vx = length * math.cos(phi) * (-0.9)  # arbitrary phi_dot = -0.9

    gravity = abs(world_config.gravity[1])
    pressure_left, pressure_right = wall_pressures(com_x, com_vx, tank_config, fluid_config, gravity)
    delta_pressure = pressure_left - pressure_right
    # No prior sample to difference against -- pass delta_pressure_dot
    # directly as the exact analytic derivative instead of a one-step
    # finite difference, so this test isolates the inversion itself.
    scale = fluid_config.density * gravity * fluid_config.fill_height / (tank_config.length / 2.0)
    delta_pressure_dot = -scale * com_vx
    pressure_observation = [theta, theta_dot, pressure_left, pressure_right, delta_pressure, delta_pressure_dot, 0.0]

    privileged_observation = [theta, theta_dot, com_x, 0.0, com_vx, 0.0, 0.0]

    torque_from_pressure = controller.get_action_from_pressure_observation(pressure_observation)
    torque_from_privileged = controller.get_action_from_observation(privileged_observation)
    assert torque_from_pressure == pytest.approx(torque_from_privileged, rel=1e-6)


def test_closed_loop_simulation_damps_an_initial_disturbance():
    controller = SloshLQRController()
    A, B = controller.A, controller.B
    state = np.array([0.12, 0.0, 0.06, 0.0])
    dt = 0.01
    peak_early = 0.0
    for step in range(1000):
        u = controller.get_action(state)
        state_dot = (A @ state.reshape(4, 1) + B * u).flatten()
        state = state + state_dot * dt
        if step < 50:
            peak_early = max(peak_early, abs(state[0]))

    # After 10s of simulated time under LQR feedback, the tank angle
    # should have settled close to zero (allowing for forward-Euler
    # integration error over 1000 steps, not exact convergence), not
    # merely stayed bounded.
    assert abs(state[0]) < 0.02
    assert peak_early <= 0.15 + 1e-6
