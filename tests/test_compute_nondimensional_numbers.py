"""Tests for scripts/compute_nondimensional_numbers.py -- pure math, no
PyChrono needed."""

import math

import pytest

from scripts.compute_nondimensional_numbers import froude_number, reynolds_number


def test_froude_number_matches_hand_computed_value():
    fr = froude_number(theta_dot_max=0.377, radius=0.20, gravity=9.81, length_scale=0.20)
    assert fr == pytest.approx(0.0538, rel=1e-2)


def test_froude_number_scales_linearly_with_angular_velocity():
    fr1 = froude_number(theta_dot_max=0.1, radius=0.2, gravity=9.81, length_scale=0.2)
    fr2 = froude_number(theta_dot_max=0.2, radius=0.2, gravity=9.81, length_scale=0.2)
    assert fr2 == pytest.approx(2.0 * fr1)


def test_froude_number_is_zero_at_zero_angular_velocity():
    assert froude_number(0.0, 0.2, 9.81, 0.2) == 0.0


def test_reynolds_number_matches_hand_computed_value():
    eta_max = 0.1378 * math.sin(math.radians(9.5))
    re = reynolds_number(
        density=1000.0, omega1=8.4375, eta_max=eta_max, length_scale=0.4, dynamic_viscosity=1.0e-3
    )
    assert re == pytest.approx(7.676e4, rel=1e-2)


def test_reynolds_number_scales_linearly_with_amplitude():
    re1 = reynolds_number(1000.0, 8.4, 0.01, 0.4, 1.0e-3)
    re2 = reynolds_number(1000.0, 8.4, 0.02, 0.4, 1.0e-3)
    assert re2 == pytest.approx(2.0 * re1)


def test_reynolds_number_is_in_the_expected_turbulent_lab_scale_range():
    """Regression guard: this project's own real config should keep
    landing in the 1e4-1e5 range cited in the report, not silently drift
    out of it if TankConfig/FluidConfig defaults ever change."""
    from config.simulation_config import FluidConfig, TankConfig
    from scripts.compute_nondimensional_numbers import WATER_DYNAMIC_VISCOSITY_PA_S
    from utilities.housner_utils import compute_housner_parameters

    tank_config = TankConfig(fixed=False)
    fluid_config = FluidConfig()
    housner = compute_housner_parameters(
        tank_length=tank_config.length,
        tank_breadth=tank_config.width,
        fill_height=fluid_config.fill_height,
        fluid_density=fluid_config.density,
        gravity=9.81,
    )
    eta_max = housner.pendulum_length * math.sin(math.radians(9.5))
    re = reynolds_number(
        fluid_config.density, housner.natural_frequency_rad_s, eta_max,
        tank_config.length, WATER_DYNAMIC_VISCOSITY_PA_S,
    )
    assert 1e4 <= re <= 1e5
