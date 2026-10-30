"""
Mass-property utilities for procedurally generated composite rigid bodies.

These functions are intentionally kept free of any PyChrono dependency so
they can be unit tested in isolation -- fast, no simulation engine, no
compiled bindings required -- and reused anywhere a composite rigid body
needs mass/COM/inertia computed from primitive box components (the tank
today; the frame or other assemblies later).

Design note: builders/tank_builder.py calls into this module rather than
duplicating the math, and its own unit tests (tests/test_inertia_utils.py)
exercise this module directly. That split is what lets the "mass and
inertia are correct" claim be independently verified without needing a
live ChSystem -- consistent with the Layer 0-1 testing philosophy from
Module 1 (unit tests: mass/inertia sanity checks, joint constraint
satisfaction).
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BoxComponent:
    """A single rectangular-prism component of a composite rigid body.

    Attributes:
        mass: Mass of the component, in kg.
        center: (x, y, z) position of the component's own centroid,
            expressed in the composite body's local reference frame, in m.
        dims: (a, b, c) full side lengths along x, y, z, in m.
    """

    mass: float
    center: Tuple[float, float, float]
    dims: Tuple[float, float, float]


@dataclass(frozen=True)
class MassProperties:
    """Aggregate mass properties of a composite rigid body."""

    mass: float
    center_of_mass: Tuple[float, float, float]
    inertia_tensor: np.ndarray  # 3x3, about the COM, in body-local axes


def box_inertia_about_centroid(mass: float, dims: Tuple[float, float, float]) -> np.ndarray:
    """Diagonal inertia tensor of a solid rectangular prism about its own centroid.

    Standard closed-form result for a uniform-density box with side lengths
    (a, b, c) aligned with the x, y, z axes:
        Ixx = m/12 * (b^2 + c^2)
        Iyy = m/12 * (a^2 + c^2)
        Izz = m/12 * (a^2 + b^2)
    """
    a, b, c = dims
    ixx = (mass / 12.0) * (b**2 + c**2)
    iyy = (mass / 12.0) * (a**2 + c**2)
    izz = (mass / 12.0) * (a**2 + b**2)
    return np.diag([ixx, iyy, izz])


def parallel_axis_shift(
    inertia_about_centroid: np.ndarray,
    mass: float,
    offset: Tuple[float, float, float],
) -> np.ndarray:
    """Shift an inertia tensor from a component's centroid to another point.

    Implements the generalized (tensor) parallel axis theorem:
        I_new = I_centroid + m * (|d|^2 * Identity - d (x) d)
    where d is the offset vector between the centroid and the new
    reference point (direction does not matter -- both terms on the right
    depend only on d via |d|^2 and the outer product d(x)d, which are
    identical for d and -d).
    """
    d = np.array(offset, dtype=float)
    shift = mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return inertia_about_centroid + shift


def compute_composite_mass_properties(components: Sequence[BoxComponent]) -> MassProperties:
    """Combine several box components into one rigid body's mass properties.

    Steps:
        1. Total mass = sum of component masses.
        2. Center of mass = mass-weighted average of component centroids.
        3. Inertia tensor about the composite COM = sum, over components, of
           (component's own centroidal inertia) shifted via the parallel
           axis theorem from the component's centroid to the composite COM.

    This function makes no PyChrono calls and has no side effects, so it is
    plain arithmetic all the way down -- safe to unit test with exact
    numbers rather than approximate simulation output.
    """
    if not components:
        raise ValueError("At least one BoxComponent is required.")

    total_mass = sum(c.mass for c in components)
    if total_mass <= 0.0:
        raise ValueError("Total mass must be positive.")

    com = np.zeros(3)
    for c in components:
        com += c.mass * np.array(c.center, dtype=float)
    com /= total_mass

    inertia = np.zeros((3, 3))
    for c in components:
        i_centroid = box_inertia_about_centroid(c.mass, c.dims)
        offset = np.array(c.center, dtype=float) - com
        inertia += parallel_axis_shift(i_centroid, c.mass, tuple(offset))

    return MassProperties(
        mass=total_mass,
        center_of_mass=tuple(com.tolist()),
        inertia_tensor=inertia,
    )
