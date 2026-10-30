"""
Pure-math quaternion utilities for aligning a joint's local Z axis to an
arbitrary world-frame rotation axis.

Kept free of any PyChrono dependency, same rationale as inertia_utils.py
and housner_utils.py: PyChrono's ChLinkMotorRotation family always
actuates rotation about the joint FRAME's own local Z axis (Chrono's own
convention, not a choice made in this project -- "Motors of this type
assume that the spindle is directed along Z direction of the master
frame," per the ChLinkMotorRotation reference docs). There is no
"pick your axis" argument on the motor itself, so builders/pivot_builder.py
and builders/motor_builder.py need the joint FRAME oriented so that its
local Z happens to already point along whichever axis this project wants
the tank to rock about (PivotConfig.axis). This module computes that
orientation as a plain (w, x, y, z) quaternion, verified numerically here
(tests/test_quaternion_utils.py) before it ever touches a ChFramed.
"""

import math
from typing import Tuple

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (w, x, y, z), matches ChQuaterniond's own arg order


def _normalize(v: Vector3) -> Vector3:
    norm = math.sqrt(sum(c * c for c in v))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def rotate_vector_by_quaternion(q: Quaternion, v: Vector3) -> Vector3:
    """Rotate vector v by quaternion q=(w,x,y,z). Exposed for test use only --
    builders/ never needs this directly, since ChQuaterniond does its own
    rotation once constructed from this module's output."""
    w, x, y, z = q
    qv = (x, y, z)
    t = tuple(2.0 * c for c in _cross(qv, v))
    cross_qv_t = _cross(qv, t)
    return (
        v[0] + w * t[0] + cross_qv_t[0],
        v[1] + w * t[1] + cross_qv_t[1],
        v[2] + w * t[2] + cross_qv_t[2],
    )


def quaternion_aligning_z_to(axis: Vector3) -> Quaternion:
    """Shortest-arc rotation quaternion mapping the canonical +Z axis onto `axis`.

    Args:
        axis: Target direction, in the parent frame, that a joint frame's
            own local +Z axis should point along once this quaternion is
            applied as that frame's orientation. Need not be pre-normalized
            -- this function normalizes it. Must not be the zero vector.

    Returns:
        (w, x, y, z), ready to pass straight into ChQuaterniond(w, x, y, z).

    Two edge cases are handled explicitly, because the general
    cross-product formula below is undefined at both of them (cross(z,
    axis) is the zero vector in both cases, so there is no well-defined
    rotation axis from the formula alone):
        - axis parallel to +Z: identity rotation, returned directly.
        - axis parallel to -Z: a 180 degree flip is required, but ANY axis
          perpendicular to Z is a valid rotation axis for it (they all
          produce the same end orientation for a vector that starts
          exactly on the rotation axis... here the "vector" IS the axis,
          so this is really just "any 180 degree flip maps +Z to -Z");
          +X is picked arbitrarily and is guaranteed perpendicular to Z.
    """
    z: Vector3 = (0.0, 0.0, 1.0)
    target = _normalize(axis)

    cos_angle = max(-1.0, min(1.0, _dot(z, target)))  # clamp guards float round-off past +-1

    if cos_angle > 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)  # already +Z -- identity

    if cos_angle < -1.0 + 1e-12:
        return (0.0, 1.0, 0.0, 0.0)  # exactly -Z -- 180 deg about arbitrary perpendicular (+X)

    rot_axis = _normalize(_cross(z, target))
    angle = math.acos(cos_angle)
    half = angle / 2.0
    s = math.sin(half)
    return (math.cos(half), rot_axis[0] * s, rot_axis[1] * s, rot_axis[2] * s)
