"""
Unit tests for utilities/quaternion_utils.py.

Pure Python + math, no pychrono import at all -- same rationale as
test_housner_utils.py and test_inertia_utils.py. This is deliberately
verified standalone, against a plain Python vector-rotation
implementation (rotate_vector_by_quaternion), BEFORE the resulting
quaternion is ever handed to a real ChQuaterniond in
builders/pivot_builder.py -- see that module's own docstring.

Run:
    pytest tests/test_quaternion_utils.py -v
"""

import math

import pytest

from utilities.quaternion_utils import quaternion_aligning_z_to, rotate_vector_by_quaternion


def _assert_unit_quaternion(q, tol=1e-9):
    norm_sq = sum(c * c for c in q)
    assert norm_sq == pytest.approx(1.0, abs=tol)


def _assert_rotates_z_onto(axis, tol=1e-9):
    """The real property every case below actually needs: applying the
    returned quaternion to +Z must reproduce the (normalized) target axis."""
    q = quaternion_aligning_z_to(axis)
    _assert_unit_quaternion(q)

    norm = math.sqrt(sum(c * c for c in axis))
    expected = tuple(c / norm for c in axis)

    rotated = rotate_vector_by_quaternion(q, (0.0, 0.0, 1.0))
    for got, want in zip(rotated, expected):
        assert got == pytest.approx(want, abs=tol)
    return q


def test_rejects_zero_vector():
    with pytest.raises(ValueError):
        quaternion_aligning_z_to((0.0, 0.0, 0.0))


def test_identity_when_axis_is_already_plus_z():
    q = quaternion_aligning_z_to((0.0, 0.0, 1.0))
    assert q == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_identity_case_also_holds_for_unnormalized_plus_z():
    """axis need not be pre-normalized -- (0, 0, 5) is still "+Z"."""
    q = quaternion_aligning_z_to((0.0, 0.0, 5.0))
    assert q == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_180_degree_edge_case_minus_z():
    """The degenerate case: cross(z, -z) is the zero vector, so this must
    be handled as an explicit special case rather than falling through to
    a division by a zero-length rotation axis."""
    q = _assert_rotates_z_onto((0.0, 0.0, -1.0))
    # A genuine 180 degree rotation has zero scalar (w) component.
    assert q[0] == pytest.approx(0.0, abs=1e-9)


def test_project_rotation_axis_plus_z_pivot_config_default():
    """This project's own PivotConfig.axis default -- confirms the common
    "no rotation needed" path used by the actual pivot/motor joint."""
    _assert_rotates_z_onto((0.0, 0.0, 1.0))


def test_quarter_turn_onto_plus_x():
    _assert_rotates_z_onto((1.0, 0.0, 0.0))


def test_quarter_turn_onto_plus_y():
    _assert_rotates_z_onto((0.0, 1.0, 0.0))


def test_oblique_axis():
    _assert_rotates_z_onto((1.0, 2.0, 3.0))


def test_result_is_always_a_unit_quaternion_across_several_axes():
    for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 1), (-2, 3, -5)]:
        _assert_unit_quaternion(quaternion_aligning_z_to(axis))
