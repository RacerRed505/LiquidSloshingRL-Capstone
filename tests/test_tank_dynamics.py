"""
Unit tests for environment/tank_dynamics.py.

Plain NumPy/Python arithmetic, no pychrono import anywhere -- runs in any
sandbox, same rationale as tests/test_housner_utils.py and
tests/test_inertia_utils.py.

Run:
    pytest tests/test_tank_dynamics.py -v
"""

import math

import pytest

from config.simulation_config import EnvironmentConfig, PivotConfig, TankConfig
from environment.tank_dynamics import (
    TankDynamicsModel,
    build_tank_dynamics_model,
    tank_moment_of_inertia_about_pivot,
)


def test_moment_of_inertia_is_positive_and_finite():
    inertia = tank_moment_of_inertia_about_pivot(TankConfig(), PivotConfig())
    assert inertia > 0.0
    assert math.isfinite(inertia)


def test_moment_of_inertia_scales_with_tank_size():
    small = tank_moment_of_inertia_about_pivot(TankConfig(length=0.20, width=0.15), PivotConfig())
    big = tank_moment_of_inertia_about_pivot(TankConfig(length=0.80, width=0.60), PivotConfig())
    assert big > small


def test_rejects_nonpositive_inertia():
    with pytest.raises(ValueError):
        TankDynamicsModel(rotational_inertia=0.0, damping_coefficient=0.0, torque_limit=1.0)


def test_rejects_negative_damping():
    with pytest.raises(ValueError):
        TankDynamicsModel(rotational_inertia=1.0, damping_coefficient=-0.1, torque_limit=1.0)


def test_rejects_nonpositive_torque_limit():
    with pytest.raises(ValueError):
        TankDynamicsModel(rotational_inertia=1.0, damping_coefficient=0.0, torque_limit=0.0)


def test_reset_zeroes_state():
    model = TankDynamicsModel(rotational_inertia=1.0, damping_coefficient=0.0, torque_limit=5.0)
    for _ in range(100):
        model.step(dt=0.01, commanded_torque=1.0)
    assert model.angle != 0.0

    model.reset()
    assert model.angle == 0.0
    assert model.angular_velocity == 0.0


def test_positive_torque_accelerates_in_positive_direction():
    model = TankDynamicsModel(rotational_inertia=1.0, damping_coefficient=0.0, torque_limit=5.0)
    model.step(dt=0.01, commanded_torque=2.0)
    assert model.angular_velocity > 0.0
    assert model.angle > 0.0


def test_torque_is_clipped_to_limit():
    model = TankDynamicsModel(rotational_inertia=1.0, damping_coefficient=0.0, torque_limit=1.0)
    model.step(dt=0.01, commanded_torque=1000.0)
    clipped = TankDynamicsModel(rotational_inertia=1.0, damping_coefficient=0.0, torque_limit=1.0)
    clipped.step(dt=0.01, commanded_torque=1.0)
    assert model.angular_velocity == pytest.approx(clipped.angular_velocity)


def test_damping_bleeds_off_energy_over_time_with_zero_torque():
    model = TankDynamicsModel(rotational_inertia=1.0, damping_coefficient=0.5, torque_limit=5.0)
    # Kick it once, then coast with zero commanded torque.
    model.step(dt=0.01, commanded_torque=3.0)
    speed_after_kick = abs(model.angular_velocity)

    for _ in range(500):
        model.step(dt=0.01, commanded_torque=0.0)

    assert abs(model.angular_velocity) < speed_after_kick


def test_build_tank_dynamics_model_uses_env_config_limits():
    env_config = EnvironmentConfig(max_motor_torque_nm=3.5, pivot_damping_coefficient=0.1)
    model = build_tank_dynamics_model(TankConfig(), PivotConfig(), env_config)
    assert model.torque_limit == pytest.approx(3.5)
