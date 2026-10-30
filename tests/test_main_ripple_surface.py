"""
Tests for main.py's --fluid-viz demo math: _RippleSurface (the CPU-only
deforming height-field surface) and _surface_tilt_rad() (the base-tilt
proxy). Both are cosmetic/decorative -- see main.py's own module
docstring and _RippleSurface's own docstring for exactly what's real vs.
not; these tests exist to catch regressions in the demo math itself
(stability, clamping, excitation direction), not to validate any physics
claim.

Requires the REAL PyChrono bindings (conda-forge) since main.py imports
pychrono.core/pychrono.irrlicht at module level -- guarded with
pytest.importorskip, same pattern as test_chrono_tank_dynamics.py /
test_pivot_and_motor.py.

Run:
    pytest tests/test_main_ripple_surface.py -v
"""

import math

import numpy as np
import pytest

chrono = pytest.importorskip(
    "pychrono.core",
    reason="Real PyChrono (conda-forge) is required for these integration tests.",
)

import main
from config.simulation_config import FluidConfig, TankConfig

_PENDULUM_LENGTH = 0.1378  # m, a representative value from this project's own default Housner params


def _build_ripple_surface() -> "main._RippleSurface":
    system = chrono.ChSystemNSC()
    return main._RippleSurface(system, TankConfig(fixed=False), FluidConfig())


# -- _surface_tilt_rad() ----------------------------------------------------


def test_tilt_is_zero_at_zero_displacement():
    assert main._surface_tilt_rad(0.0, _PENDULUM_LENGTH) == 0.0


def test_tilt_reaches_max_at_full_displacement():
    tilt = main._surface_tilt_rad(_PENDULUM_LENGTH, _PENDULUM_LENGTH)
    assert tilt == pytest.approx(main._SURFACE_MAX_TILT_RAD)


def test_tilt_reaches_negative_max_at_full_negative_displacement():
    tilt = main._surface_tilt_rad(-_PENDULUM_LENGTH, _PENDULUM_LENGTH)
    assert tilt == pytest.approx(-main._SURFACE_MAX_TILT_RAD)


def test_tilt_is_proportional_below_the_cap():
    tilt = main._surface_tilt_rad(0.5 * _PENDULUM_LENGTH, _PENDULUM_LENGTH)
    assert tilt == pytest.approx(0.5 * main._SURFACE_MAX_TILT_RAD)


def test_tilt_clamps_beyond_full_displacement():
    # fluid_com_x / pendulum_length is sin(theta) and should stay in
    # [-1, 1] in practice, but the clamp guards against float/edge-case
    # overshoot -- this is what that clamp is for.
    tilt = main._surface_tilt_rad(2.0 * _PENDULUM_LENGTH, _PENDULUM_LENGTH)
    assert tilt == pytest.approx(main._SURFACE_MAX_TILT_RAD)
    tilt_negative = main._surface_tilt_rad(-2.0 * _PENDULUM_LENGTH, _PENDULUM_LENGTH)
    assert tilt_negative == pytest.approx(-main._SURFACE_MAX_TILT_RAD)


# -- _RippleSurface construction --------------------------------------------


def test_grid_has_expected_vertex_and_triangle_counts():
    surface = _build_ripple_surface()
    n = main._RIPPLE_GRID_N
    assert surface.mesh.GetNumVertices() == n * n
    assert surface.mesh.GetNumTriangles() == 2 * (n - 1) * (n - 1)


def test_grid_vertices_start_flat_at_zero_height():
    surface = _build_ripple_surface()
    for v in surface.mesh.GetCoordsVertices():
        assert v.y == 0.0


# -- _RippleSurface.step_wave() ----------------------------------------------


def test_step_wave_stays_zero_with_no_excitation_from_a_flat_start():
    surface = _build_ripple_surface()
    surface.step_wave(tank_angular_velocity=0.0)
    assert np.all(surface._h_curr == 0.0)


def test_step_wave_excites_edges_oppositely_from_a_flat_start():
    surface = _build_ripple_surface()
    angular_velocity = 0.4  # rad/s, representative of values seen in the demo's own logs
    surface.step_wave(angular_velocity)
    expected = main._RIPPLE_EXCITATION_GAIN * angular_velocity
    assert np.allclose(surface._h_curr[0, :], expected)
    assert np.allclose(surface._h_curr[-1, :], -expected)
    # Interior points haven't been reached by the wave after only one step.
    assert np.allclose(surface._h_curr[1:-1, 1:-1], 0.0)


def test_step_wave_direction_flips_with_angular_velocity_sign():
    surface = _build_ripple_surface()
    surface.step_wave(tank_angular_velocity=-0.4)
    assert np.all(surface._h_curr[0, :] < 0.0)
    assert np.all(surface._h_curr[-1, :] > 0.0)


def test_repeated_settling_without_further_input_decays_not_diverges():
    """Regression guard for _RIPPLE_DAMPING and the wave-step formula: a
    disturbance excited once, then left alone (zero angular velocity every
    subsequent step), must settle down over time, not blow up. If this
    ever fails after changing _RIPPLE_DAMPING or the neighbor-averaging
    formula, that's the numerical instability this test exists to catch.
    """
    surface = _build_ripple_surface()
    surface.step_wave(tank_angular_velocity=1.0)  # seed a real disturbance
    peak_after_seed = np.max(np.abs(surface._h_curr))
    assert peak_after_seed > 0.0

    for _ in range(3000):  # 3s of simulated time at the real 1ms physics step
        surface.step_wave(tank_angular_velocity=0.0)

    assert np.isfinite(surface._h_curr).all()
    assert np.max(np.abs(surface._h_curr)) < peak_after_seed


# -- _RippleSurface.sync_mesh() ----------------------------------------------


def test_sync_mesh_writes_tilt_shear_into_vertex_heights():
    surface = _build_ripple_surface()
    tilt_rad = math.radians(5.0)
    surface.sync_mesh(tilt_rad)
    verts = surface.mesh.GetCoordsVertices()
    n = main._RIPPLE_GRID_N
    for i, x in enumerate(surface._x_coords):
        for j in range(n):
            # Ripple is still all-zero here (no step_wave() call yet), so
            # the only height contribution is the tilt shear itself.
            assert verts[i * n + j].y == pytest.approx(tilt_rad * x)


def test_sync_mesh_includes_ripple_height_on_top_of_tilt():
    surface = _build_ripple_surface()
    surface.step_wave(tank_angular_velocity=0.4)  # seeds a known, nonzero ripple
    surface.sync_mesh(tilt_rad=0.0)
    verts = surface.mesh.GetCoordsVertices()
    n = main._RIPPLE_GRID_N
    # Row 0 is the excited edge; tilt is zero here, so this is exactly the
    # excitation height computed in step_wave() above, now on the mesh.
    expected = main._RIPPLE_EXCITATION_GAIN * 0.4
    for j in range(n):
        assert verts[j].y == pytest.approx(expected)
