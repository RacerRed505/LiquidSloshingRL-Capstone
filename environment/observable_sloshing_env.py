"""Hardware-deployable observable variant of SloshingSuppressionEnv.

environment/sloshing_env.py::SloshingSuppressionEnv's own 7-D observation
includes the fluid's raw center-of-mass position/velocity
(fluid_com_x/y, fluid_com_vx/vy) -- not measurable on a physical tank
without an internal fluid sensor no real hardware deployment would have.
This is a real, legitimate critique (see the project notes' notes on the
external reference package for where this idea came from) -- but
the fix belongs INSIDE this project's own architecture, not as a
disconnected parallel reimplementation.

This class subclasses SloshingSuppressionEnv rather than rebuilding it:
every physics/action-space/reward guarantee the base class already gives
(and every FluidBackend/TankDynamicsModel injection point Phase 10-14
already rely on) still holds unchanged. Only the OBSERVATION changes --
reset()/step() are inherited as-is (not duplicated here) and call
self._build_observation() via ordinary Python polymorphism, which
resolves to THIS class's override automatically.

Sensor model
------------
Real hardware cannot see fluid_com directly, but a pair of differential
wall-pressure transducers is physically installable and directly related
to how far the fluid has shifted: a higher fluid column at one wall means
higher static pressure there. This class derives that relationship from
FluidState.com_position ALONE (never anything backend-internal like
PendulumBackend's own pendulum length) -- so it works unchanged whichever
concrete FluidBackend is injected (PendulumBackend today; SPHBackend/a
real Chrono::FSI backend later, exactly the swap physics/backend_interface.py's
own docstring describes), which raw fluid COM cannot claim, since a
particle-based SPH backend's own com_position is exactly as "observable"
in code as a pendulum's.

The tilt/wave-elevation relationship is an explicit small-slope PROXY,
not an exact free-surface solve -- the same honesty convention main.py's
own `_surface_tilt_rad` (a fluid_com_x-derived cosmetic proxy) already
uses for the identical reason: treating "the fluid COM shifted toward one
side" as "the free surface tilted linearly across the tank" is a
reasonable first-order approximation, not a first-principles derivation.
Don't cite it as validated free-surface physics.

Observation (float32, shape (7,)):
    [tank_angle, tank_angular_velocity,
     pressure_left, pressure_right, delta_pressure, delta_pressure_dot,
     previous_torque_nm]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from config.simulation_config import (
    EnvironmentConfig,
    FluidConfig,
    PivotConfig,
    TankConfig,
    WorldConfig,
)
from environment.sloshing_env import SloshingSuppressionEnv
from environment.tank_dynamics import TankDynamicsModel
from physics.backend_interface import FluidBackend

_OBSERVATION_LOW = np.array(
    [-np.inf, -np.inf, 0.0, 0.0, -np.inf, -np.inf, -np.inf], dtype=np.float32
)
_OBSERVATION_HIGH = np.full((7,), np.inf, dtype=np.float32)


def wall_pressures(
    com_x: float,
    com_vx: float,
    tank_config: TankConfig,
    fluid_config: FluidConfig,
    gravity: float,
) -> Tuple[float, float]:
    """Differential hydrostatic + dynamic wall pressure proxy, in Pa.

    Treats the fluid's lateral COM displacement (com_x, tank-local frame,
    from FluidState.com_position -- the backend-agnostic interface, never
    a concrete backend's own internals) as a linear free-surface tilt about
    the tank's own footprint center -- an explicit small-slope PROXY, not a
    first-principles free-surface solve (see this module's docstring).

    A standalone function (not a method) specifically so it can be reused
    unchanged by any environment wrapping a `FluidState`-producing backend
    -- `ObservableSloshingEnv` (the fast surrogate) and
    `environment/observable_closed_loop_sph_env.py::ObservableClosedLoopSPHEnv`
    (the real Chrono::FSI closed loop) both call this exact function rather
    than maintaining two copies of the same physics.
    """
    half_length = tank_config.length / 2.0
    fill_height = fluid_config.fill_height
    rho = fluid_config.density

    tilt_fraction = com_x / half_length if half_length > 0.0 else 0.0
    wave_elevation = tilt_fraction * (fill_height / 2.0)

    height_left = fill_height - wave_elevation
    height_right = fill_height + wave_elevation

    dynamic_head = 0.5 * rho * (com_vx ** 2)
    pressure_left = rho * gravity * max(0.0, height_left) + dynamic_head
    pressure_right = rho * gravity * max(0.0, height_right) + dynamic_head
    return pressure_left, pressure_right


class ObservableSloshingEnv(SloshingSuppressionEnv):
    """SloshingSuppressionEnv with a hardware-deployable, sensor-based observation.

    See this module's docstring for the sensor model and why it's derived
    only from FluidState.com_position (backend-agnostic), never from a
    concrete backend's own internal parameters.
    """

    def __init__(
        self,
        tank_config: Optional[TankConfig] = None,
        pivot_config: Optional[PivotConfig] = None,
        fluid_config: Optional[FluidConfig] = None,
        world_config: Optional[WorldConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
        fluid_backend: Optional[FluidBackend] = None,
        tank_dynamics: Optional[TankDynamicsModel] = None,
    ) -> None:
        super().__init__(
            tank_config=tank_config,
            pivot_config=pivot_config,
            fluid_config=fluid_config,
            world_config=world_config,
            env_config=env_config,
            fluid_backend=fluid_backend,
            tank_dynamics=tank_dynamics,
        )

        # Overrides the base class's observation_space -- the pressures
        # this class reports are non-negative by construction (see
        # _wall_pressures()'s own max(0, ...) clamp), so the space itself
        # documents that rather than reusing the base class's all-real
        # bounds.
        self.observation_space = spaces.Box(
            low=_OBSERVATION_LOW, high=_OBSERVATION_HIGH, dtype=np.float32
        )

        self._prev_applied_torque = 0.0
        self._prev_delta_pressure = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Reset this class's own extra state BEFORE calling super().reset()
        # -- the base class's reset() calls self._build_observation() at
        # its very end, which (via polymorphism) is THIS class's override,
        # so these need to already be at their zeroed values by then.
        self._prev_applied_torque = 0.0
        self._prev_delta_pressure = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # The base class's step() computes the physical torque locally and
        # never stores it -- recomputed here with the IDENTICAL formula
        # (normalize -> scale by max_motor_torque_nm) so this observation
        # can report it, rather than modifying sloshing_env.py just to
        # expose an attribute one subclass wants.
        normalized_action = float(
            np.clip(action, self.action_space.low, self.action_space.high)[0]
        )
        physical_torque = normalized_action * self._env_config.max_motor_torque_nm

        # Set BEFORE calling super().step() -- the base class's step()
        # calls self._build_observation() (this class's override, via
        # polymorphism) at its very end, so "previous_torque" in the
        # observation this call returns means "the torque this exact
        # action just applied," matching the usual RL convention of
        # feeding an agent its own last action back as context.
        self._prev_applied_torque = physical_torque
        return super().step(action)

    def _wall_pressures(self, com_x: float, com_vx: float) -> Tuple[float, float]:
        """Differential hydrostatic + dynamic wall pressure proxy, in Pa.

        Treats the fluid's lateral COM displacement (com_x, tank-local
        frame, from FluidState.com_position) as a linear free-surface tilt
        about the tank's own footprint center -- an explicit small-slope
        PROXY (see this module's docstring), not a first-principles
        free-surface solve.
        """
        gravity = abs(self._world_config.gravity[1])
        return wall_pressures(com_x, com_vx, self._tank_config, self._fluid_config, gravity)

    def _build_observation(self) -> np.ndarray:
        fluid_state = self._fluid_backend.get_state()
        com_x = fluid_state.com_position[0]
        com_vx = fluid_state.com_velocity[0]

        pressure_left, pressure_right = self._wall_pressures(com_x, com_vx)
        delta_pressure = pressure_left - pressure_right
        delta_pressure_dot = (
            delta_pressure - self._prev_delta_pressure
        ) / self._env_config.control_dt
        self._prev_delta_pressure = delta_pressure

        return np.array(
            [
                self._tank_dynamics.angle,
                self._tank_dynamics.angular_velocity,
                pressure_left,
                pressure_right,
                delta_pressure,
                delta_pressure_dot,
                self._prev_applied_torque,
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("OBSERVABLE (POMDP) SLOSHING ENVIRONMENT -- SELF-TEST")
    print("=" * 60)

    env = ObservableSloshingEnv()
    obs, info = env.reset(seed=0)
    print(f"\nInitial observation [theta, theta_dot, P_L, P_R, dP, dP_dot, prev_u]:")
    print(f"  {obs}")

    for step in range(10):
        action = np.array([0.3 * (-1.0 if step % 2 == 0 else 1.0)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 2 == 0:
            print(
                f"  step={step:2d}  theta={obs[0]:+.4f}  P_L={obs[2]:8.3f} Pa  "
                f"P_R={obs[3]:8.3f} Pa  dP={obs[4]:+8.3f} Pa  prev_u={obs[6]:+.3f} N*m"
            )

    print("\nObservation space:", env.observation_space)
    print("=" * 60)
