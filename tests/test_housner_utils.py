"""
Unit tests for utilities/housner_utils.py.

Pure Python + math, no pychrono import at all -- runs anywhere, same
rationale as test_inertia_utils.py. FluidBuilder-level integration
(reading TankConfig/FluidConfig, constructing a real PendulumBackend) is
exercised separately once builders/fluid_builder.py has a matching
integration test alongside test_world_and_tank.py.

Run:
    pytest tests/test_housner_utils.py -v
"""

import math

import pytest

from utilities.housner_utils import compute_housner_parameters


DEFAULT_KWARGS = dict(tank_length=0.40, tank_breadth=0.20, fill_height=0.25)


def test_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError):
        compute_housner_parameters(tank_length=0.0, tank_breadth=0.2, fill_height=0.2)
    with pytest.raises(ValueError):
        compute_housner_parameters(tank_length=0.4, tank_breadth=0.2, fill_height=-0.1)


def test_total_mass_matches_volume_times_density():
    params = compute_housner_parameters(**DEFAULT_KWARGS, fluid_density=1000.0)
    expected = 1000.0 * 0.40 * 0.20 * 0.25
    assert params.total_liquid_mass == pytest.approx(expected)


def test_masses_and_heights_are_physically_bounded():
    """Structural sanity bounds every published source agrees on --
    NOT including m_i + m_c <= total, which the classical Housner
    approximation does not guarantee (see modeled_mass_fraction test
    below)."""
    params = compute_housner_parameters(**DEFAULT_KWARGS)
    assert 0.0 < params.impulsive_mass < params.total_liquid_mass
    assert 0.0 < params.convective_mass < params.total_liquid_mass
    assert 0.0 < params.impulsive_height < DEFAULT_KWARGS["fill_height"]
    assert 0.0 < params.convective_pivot_height < DEFAULT_KWARGS["fill_height"]
    assert params.pendulum_length > 0.0
    assert params.natural_frequency_rad_s > 0.0


def test_modeled_mass_fraction_is_not_assumed_to_be_exactly_one():
    """Documents a known property of the classical Housner model rather
    than silently assuming it away: m_i/m + m_c/m is an approximation,
    not an exact partition, and can exceed 1.0 for some L/H ratios. This
    project's default tank geometry (L=0.40, H=0.25 -> L/H=1.6) is one
    such case -- pinned here so a future "helpful" fix doesn't quietly
    clamp it to 1.0 and hide the fact that this is expected."""
    params = compute_housner_parameters(**DEFAULT_KWARGS)
    assert params.modeled_mass_fraction == pytest.approx(1.0431, abs=5e-4)


def test_pendulum_length_reproduces_target_frequency():
    """Cross-checks pendulum_length against the real (nonlinear) simple
    pendulum equation theta'' = -(g/L) sin(theta) by direct numerical
    integration, independent of the closed-form omega^2 = g/L algebra
    used to derive it. This is the one number PendulumBackend actually
    consumes, so it's the one most worth verifying end-to-end rather
    than trusting the formula chain alone."""
    g = 9.81
    params = compute_housner_parameters(**DEFAULT_KWARGS, gravity=g)

    theta, theta_dot = 0.05, 0.0  # small angle, radians
    dt, t = 1e-5, 0.0
    crossings = []
    prev_theta = theta
    while t < 4.0:
        theta_dot += -(g / params.pendulum_length) * math.sin(theta) * dt
        theta += theta_dot * dt
        if prev_theta < 0 <= theta:
            crossings.append(t)
        prev_theta = theta
        t += dt

    assert len(crossings) >= 2, "not enough oscillation cycles captured"
    measured_period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    measured_freq = 1.0 / measured_period

    assert measured_freq == pytest.approx(params.natural_frequency_hz, rel=1e-2)


def test_frequency_increases_with_deeper_fill():
    """Physical monotonicity check, matching the standard gravity-wave
    dispersion relation omega^2 = g*k*tanh(k*H): as H grows relative to L,
    tanh saturates toward 1 and omega approaches the deep-water limit;
    as H shrinks, tanh(kH) ~ kH and omega -> 0 (shallow-water limit).
    So deeper fill => higher sloshing frequency, not lower -- verified
    here so the direction doesn't get flipped by accident later.

    Both fill heights stay within a moderate L/H range on purpose: the
    ACI 350.3-06 impulsive-height formula (the hi_frac branch in
    compute_housner_parameters) is an empirical fit intended for
    reasonably proportioned tanks, and produces a negative
    impulsive_height for very shallow/wide tanks (large L/H) -- a known
    range limitation of the curve fit, not something to paper over with
    an artificially extreme test case here.
    """
    shallower = compute_housner_parameters(tank_length=0.40, tank_breadth=0.20, fill_height=0.15)
    deeper = compute_housner_parameters(tank_length=0.40, tank_breadth=0.20, fill_height=0.28)
    assert deeper.natural_frequency_hz > shallower.natural_frequency_hz
