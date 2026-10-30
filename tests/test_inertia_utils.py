"""
Unit tests for utilities/inertia_utils.py.

These tests use plain NumPy arithmetic and do not import pychrono at all,
so they run anywhere Python + NumPy is installed -- including this
sandbox, where the real (conda-only) PyChrono bindings are not available.
Tests that need an actual ChSystem live in test_world_and_tank.py instead,
and are meant to be run in the project's real PyChrono environment.

Run:
    pytest tests/test_inertia_utils.py -v
"""

import numpy as np
import pytest

from utilities.inertia_utils import (
    BoxComponent,
    box_inertia_about_centroid,
    compute_composite_mass_properties,
    parallel_axis_shift,
)


def test_box_inertia_matches_closed_form():
    mass, dims = 2.0, (0.4, 0.3, 0.2)
    a, b, c = dims
    expected = np.diag([
        mass / 12 * (b**2 + c**2),
        mass / 12 * (a**2 + c**2),
        mass / 12 * (a**2 + b**2),
    ])
    result = box_inertia_about_centroid(mass, dims)
    np.testing.assert_allclose(result, expected)


def test_parallel_axis_shift_never_decreases_diagonal():
    """Shifting away from the centroid is a positive-semidefinite addition,
    so no diagonal (principal-axis-aligned) inertia term can decrease."""
    i_centroid = box_inertia_about_centroid(1.0, (0.1, 0.1, 0.1))
    shifted = parallel_axis_shift(i_centroid, mass=1.0, offset=(0.5, 0.0, 0.0))
    assert np.all(np.diag(shifted) >= np.diag(i_centroid) - 1e-12)


def test_composite_mass_is_sum_of_components():
    components = [
        BoxComponent(mass=1.0, center=(0, 0, 0), dims=(0.1, 0.1, 0.1)),
        BoxComponent(mass=2.5, center=(0.2, 0, 0), dims=(0.1, 0.1, 0.1)),
    ]
    result = compute_composite_mass_properties(components)
    assert result.mass == pytest.approx(3.5)


def test_composite_com_is_mass_weighted_average():
    components = [
        BoxComponent(mass=1.0, center=(0.0, 0.0, 0.0), dims=(0.1, 0.1, 0.1)),
        BoxComponent(mass=1.0, center=(1.0, 0.0, 0.0), dims=(0.1, 0.1, 0.1)),
    ]
    result = compute_composite_mass_properties(components)
    np.testing.assert_allclose(result.center_of_mass, (0.5, 0.0, 0.0), atol=1e-9)


def test_open_top_tank_com_and_symmetry():
    """Builds the same 5-box tank geometry as TankBuilder._component_boxes()
    and checks two physical properties an open-top box must have:
      1. COM sits on the vertical centerline (x=0, z=0) by left/right and
         front/back symmetry, even though it's pulled off y=h/2 by the base.
      2. All products of inertia (off-diagonal terms) vanish -- which is
         exactly what lets TankBuilder set inertia with independent
         SetInertiaXX/SetInertiaXY calls and trust the diagonal terms are
         the true principal moments.
    """
    L, W, h, t, density = 0.40, 0.25, 0.30, 0.006, 1180.0

    def box_mass(a, b, c):
        return density * a * b * c

    components = [
        BoxComponent(mass=box_mass(L, t, W), center=(0, -t / 2, 0), dims=(L, t, W)),
        BoxComponent(mass=box_mass(L, h, t), center=(0, h / 2, W / 2 - t / 2), dims=(L, h, t)),
        BoxComponent(mass=box_mass(L, h, t), center=(0, h / 2, -(W / 2 - t / 2)), dims=(L, h, t)),
        BoxComponent(mass=box_mass(t, h, W), center=(L / 2 - t / 2, h / 2, 0), dims=(t, h, W)),
        BoxComponent(mass=box_mass(t, h, W), center=(-(L / 2 - t / 2), h / 2, 0), dims=(t, h, W)),
    ]
    result = compute_composite_mass_properties(components)

    x_com, y_com, z_com = result.center_of_mass
    assert x_com == pytest.approx(0.0, abs=1e-9)
    assert z_com == pytest.approx(0.0, abs=1e-9)
    # Walls outweigh the thin base, so COM sits above the floor (y=0) but
    # below the walls' own midpoint (h/2), pulled down slightly by the base.
    assert 0.0 < y_com < h / 2

    off_diag = result.inertia_tensor - np.diag(np.diag(result.inertia_tensor))
    np.testing.assert_allclose(off_diag, np.zeros((3, 3)), atol=1e-12)


def test_rejects_empty_component_list():
    with pytest.raises(ValueError):
        compute_composite_mass_properties([])


def test_rejects_zero_or_negative_total_mass():
    components = [BoxComponent(mass=0.0, center=(0, 0, 0), dims=(0.1, 0.1, 0.1))]
    with pytest.raises(ValueError):
        compute_composite_mass_properties(components)
