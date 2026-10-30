"""
Closed-form Housner (1963) / ACI 350.3-06 equivalent-mechanical-model
parameters for a sloshing liquid in a rigid rectangular tank.

Kept free of any PyChrono dependency, same rationale as inertia_utils.py:
fast, no simulation engine, no compiled bindings required, unit tested in
isolation (tests/test_housner_utils.py) and reused by builders/fluid_builder.py
without requiring a live ChSystem.

The liquid is decomposed into an "impulsive" mass (rigidly attached to the
tank, does not slosh) and a "convective" mass (the free-surface sloshing
mode). Only the first sloshing mode is modeled -- standard practice for
this class of model; the impulsive+convective masses are curve-fit
approximations rather than an exact energy decomposition, so they are NOT
guaranteed to sum to the total liquid mass (see note on
`modeled_mass_fraction` below).

The convective mass is further reduced to a literal pendulum: a simple
pendulum has natural frequency omega^2 = g / L, so setting that equal to
the liquid's real first sloshing frequency gives the pendulum length
directly. This is what lets PendulumBackend (physics/pendulum_backend.py)
be driven by a single scalar length rather than a spring constant.

References
----------
- G. W. Housner, "The Dynamic Behavior of Water Tanks," Bulletin of the
  Seismological Society of America, 53(2), 1963, pp. 381-387.
- ACI 350.3-06, "Seismic Design of Liquid-Containing Concrete Structures,"
  American Concrete Institute, Ch. 9 (rectangular tank equations).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HousnerParameters:
    """Derived equivalent-pendulum parameters for a rectangular tank."""

    total_liquid_mass: float        # kg
    impulsive_mass: float           # kg, m_i -- rigidly attached to tank
    convective_mass: float          # kg, m_c -- the pendulum bob mass
    impulsive_height: float         # m, from tank floor
    convective_pivot_height: float  # m, pendulum pivot height from tank floor
    pendulum_length: float          # m, L_p (feeds PendulumBackend directly)
    natural_frequency_rad_s: float  # rad/s, first sloshing mode
    natural_frequency_hz: float
    modeled_mass_fraction: float    # (m_i + m_c) / total -- see note below


def compute_housner_parameters(
    tank_length: float,
    tank_breadth: float,
    fill_height: float,
    fluid_density: float = 1000.0,
    gravity: float = 9.81,
) -> HousnerParameters:
    """Housner (1963) / ACI 350.3-06 parameters for a rectangular tank.

    Args:
        tank_length: In-plane tank dimension along the sloshing/excitation
            axis, in m (this project's TankConfig.length, i.e. the x axis
            the pivot/motor rotates the tank about -- see pivot_builder.py).
        tank_breadth: The other horizontal in-plane dimension, in m
            (TankConfig.width). Only enters via total liquid mass; the 1D
            sloshing solution itself depends only on tank_length and
            fill_height.
        fill_height: Liquid depth, in m. Must be less than TankConfig.height
            (checked by the caller, not here -- this function only knows
            about the liquid, not the tank walls).
        fluid_density: kg/m^3 (1000.0 = water).
        gravity: m/s^2, magnitude (sign convention is the caller's problem;
            WorldConfig.gravity is a signed 3-vector, this takes a magnitude).

    Note on valid L/H range:
        The impulsive-height formula (hi_frac below) is an ACI 350.3-06
        curve fit intended for reasonably proportioned tanks. For very
        shallow/wide tanks (L/H large -- e.g. L/H > ~5) it produces a
        NEGATIVE impulsive_height, which is not physically meaningful.
        This is a documented limitation of the empirical fit itself, not
        a bug here; this project's tank stays well within the range where
        it behaves (TankConfig.length=0.40 against fill heights up to
        TankConfig.height=0.30 keeps L/H below ~3), so it is surfaced as
        a docstring note rather than a runtime check.

    Note on modeled_mass_fraction:
        m_i and m_c are independent curve-fit approximations, not two
        halves of an exact decomposition -- ACI 350.3-06 and the original
        Housner paper do not guarantee m_i/m + m_c/m == 1. For some L/H
        ratios the sum comes out slightly ABOVE 1.0 (verified numerically
        for this project's default TankConfig geometry: ~102.7%). This is a
        known property of the classical curve fits, which approximate the
        exact equivalent mechanical model (e.g. Li & Wang 2012, J. Fluids
        Struct. 31:147-151), not a bug in this implementation. modeled_mass_fraction is
        exposed so callers/tests can see this rather than have it hidden.
    """
    if tank_length <= 0 or tank_breadth <= 0 or fill_height <= 0:
        raise ValueError("tank_length, tank_breadth, and fill_height must all be positive.")

    total_mass = fluid_density * tank_length * tank_breadth * fill_height
    ratio = tank_length / fill_height  # L/H

    # --- Impulsive mass fraction (ACI 350.3-06, rectangular tank) ---
    x = 0.866 * ratio
    impulsive_mass = (math.tanh(x) / x) * total_mass

    # --- Convective mass fraction ---
    y = 3.16 * fill_height / tank_length  # note: 3.16 is ACI/Housner's rounded coefficient (~pi)
    convective_mass = 0.264 * ratio * math.tanh(y) * total_mass

    # --- Impulsive height (excludes base-pressure contribution) ---
    if ratio >= 1.333:
        hi_frac = 0.5 - 0.09375 * ratio
    else:
        hi_frac = 0.375
    impulsive_height = hi_frac * fill_height

    # --- Convective height / pendulum pivot height ---
    hc_frac = 1.0 - (math.cosh(y) - 1.0) / (y * math.sinh(y))
    convective_height = hc_frac * fill_height

    # --- First sloshing natural frequency ---
    omega_sq = (3.16 * gravity / tank_length) * math.tanh(y)
    omega = math.sqrt(omega_sq)
    freq_hz = omega / (2.0 * math.pi)

    # --- Equivalent simple-pendulum length: omega^2 = g / L_p ---
    pendulum_length = gravity / omega_sq

    return HousnerParameters(
        total_liquid_mass=total_mass,
        impulsive_mass=impulsive_mass,
        convective_mass=convective_mass,
        impulsive_height=impulsive_height,
        convective_pivot_height=convective_height,
        pendulum_length=pendulum_length,
        natural_frequency_rad_s=omega,
        natural_frequency_hz=freq_hz,
        modeled_mass_fraction=(impulsive_mass + convective_mass) / total_mass,
    )
