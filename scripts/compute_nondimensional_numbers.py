"""
Non-dimensional sloshing parameters for this project's real tank/fluid
config -- for the report's theory section (Froude number, Reynolds
number; Keulegan-Carpenter number is explicitly flagged as not
applicable, see below).

Written as a runnable script rather than hand-computed prose specifically
so the numbers are independently checkable (`python -m
scripts.compute_nondimensional_numbers` reproduces them exactly), per
this project's own "verify by running, don't trust a claimed number"
convention -- an external reference package (see the project notes)
presented this same theory section as prose with no accompanying
computation to check it against.

Every input number here is either read directly from
config/simulation_config.py or cited from an already-documented,
already-run result elsewhere in this project (the project notes' Architecture
section, the development log) -- none are invented for this script.

Run:
    python -m scripts.compute_nondimensional_numbers
"""

from __future__ import annotations

import math

from config.simulation_config import FluidConfig, MotorConfig, TankConfig
from utilities.housner_utils import compute_housner_parameters

WATER_DYNAMIC_VISCOSITY_PA_S = 1.0e-3  # mu, water at ~20 deg C -- standard physical constant


def froude_number(theta_dot_max: float, radius: float, gravity: float, length_scale: float) -> float:
    """Fr = theta_dot_max * radius / sqrt(g * length_scale).

    Ratio of inertial overturning forces to gravitational restoring
    forces. Fr << 0.2 => linear potential-flow sloshing (small, sinusoidal
    waves); Fr > ~0.4 => hydraulic jumps / wave breaking dominate (see
    e.g. Ibrahim, "Liquid Sloshing Dynamics", 2005, Ch. 1 for the
    qualitative regime boundaries -- this project's own tank has never
    been driven hard enough to test the Fr > 0.4 regime, so that
    threshold is cited as literature context, not independently verified
    here).
    """
    return (theta_dot_max * radius) / math.sqrt(gravity * length_scale)


def reynolds_number(
    density: float, omega1: float, eta_max: float, length_scale: float, dynamic_viscosity: float
) -> float:
    """Re = rho * omega1 * eta_max * length_scale / mu.

    Ratio of convective inertial forces to viscous forces at the
    fluid/wall boundary layer. Re ~ 1e4-1e5 (as computed below) puts this
    project's real tank in the turbulent-boundary-layer regime typical of
    lab-scale water sloshing tanks.
    """
    return (density * omega1 * eta_max * length_scale) / dynamic_viscosity


def main() -> None:
    tank_config = TankConfig(fixed=False)
    fluid_config = FluidConfig()
    motor_config = MotorConfig()
    gravity = 9.81

    housner = compute_housner_parameters(
        tank_length=tank_config.length,
        tank_breadth=tank_config.width,
        fill_height=fluid_config.fill_height,
        fluid_density=fluid_config.density,
        gravity=gravity,
    )

    print("=" * 70)
    print("NON-DIMENSIONAL SLOSHING PARAMETERS -- this project's real config")
    print("=" * 70)
    print(f"\nTank: length={tank_config.length} m, width={tank_config.width} m, "
          f"fill_height={fluid_config.fill_height} m")
    print(f"Housner: pendulum_length={housner.pendulum_length:.4f} m, "
          f"natural_frequency={housner.natural_frequency_hz:.4f} Hz "
          f"({housner.natural_frequency_rad_s:.4f} rad/s)")

    # -- Froude number: MotorConfig's own default scripted drive
    # (Phase 6's checkpoint demo, amplitude_rad=0.15, frequency_hz=0.4)
    # as the characteristic tank motion -- a real, already-documented
    # operating point, not an invented one.
    theta_dot_max = motor_config.amplitude_rad * 2.0 * math.pi * motor_config.frequency_hz
    radius = tank_config.length / 2.0
    fr = froude_number(theta_dot_max, radius, gravity, fluid_config.fill_height)

    print("\n-- Froude number (MotorConfig's default scripted drive) --")
    print(f"theta_dot_max = amplitude_rad * 2*pi*frequency_hz = {theta_dot_max:.4f} rad/s")
    print(f"radius (tank half-length) = {radius:.4f} m")
    print(f"Fr = theta_dot_max * radius / sqrt(g * fill_height) = {fr:.4f}")
    print(f"  ({'Fr << 0.2 -> linear potential-flow sloshing regime' if fr < 0.2 else 'Fr >= 0.2 -- outside the small-wave linear regime'})")

    # -- Reynolds number: the `--fluid-viz-resonance` demo's own
    # documented wave amplitude (~9-10 deg tilt, natural-frequency drive)
    # as the characteristic sloshing amplitude -- see the project notes' Architecture
    # section for that number's own origin (a real, run demo, not invented
    # here).
    theta_max_deg = 9.5  # midpoint of the documented ~9-10 deg range
    eta_max = housner.pendulum_length * math.sin(math.radians(theta_max_deg))
    re = reynolds_number(
        fluid_config.density, housner.natural_frequency_rad_s, eta_max,
        tank_config.length, WATER_DYNAMIC_VISCOSITY_PA_S,
    )

    print("\n-- Reynolds number (--fluid-viz-resonance demo's documented ~9-10 deg tilt) --")
    print(f"eta_max = pendulum_length * sin({theta_max_deg} deg) = {eta_max * 1000:.2f} mm")
    print(f"Re = rho * omega1 * eta_max * tank_length / mu = {re:.3e}")
    print(f"  ({'Re in the 1e4-1e5 turbulent-boundary-layer range typical of lab-scale water sloshing' if 1e4 <= re <= 1e5 else 'Re outside the commonly-cited 1e4-1e5 lab-scale range'})")

    # -- Keulegan-Carpenter number: explicitly NOT computed here.
    print("\n-- Keulegan-Carpenter number: NOT APPLICABLE to this project as built --")
    print("KC = 2*pi*eta_max / D_hole characterizes vortex shedding at a PERFORATED")
    print("BAFFLE's orifices. This project's real TankConfig has no baffle at all")
    print("(an external reference package's baffle designs were never")
    print("integrated -- see the project notes' note on that package). Computing KC here")
    print("would require inventing a hole diameter this project doesn't have;")
    print("left as a formula for if/when a real baffle is actually added and")
    print("tested against physics/closed_loop_sph_tank.py, not fabricated now.")
    print("=" * 70)


if __name__ == "__main__":
    main()
