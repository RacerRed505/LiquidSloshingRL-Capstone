"""Hardware-deployable observable variant of ClosedLoopSPHEnv (Phase 15).

environment/observable_sloshing_env.py::ObservableSloshingEnv already did
this for the fast surrogate (SloshingSuppressionEnv). This module is the
same idea applied to the real, two-way-coupled Chrono::FSI closed loop
(physics/closed_loop_sph_tank.py::ClosedLoopSPHTank via
environment/closed_loop_sph_env.py::ClosedLoopSPHEnv) -- the project notes' own
"What's next" list named this as the one piece of the reference-package
audit's legitimate ideas not yet wired into the real SPH backend.

ClosedLoopSPHEnv does NOT subclass SloshingSuppressionEnv (it can't --
ClosedLoopSPHTank combines tank and fluid into one two-way-coupled object,
incompatible with SloshingSuppressionEnv's two-separately-stepped-objects
design; see that module's own docstring). So this is a second, parallel
subclass -- of ClosedLoopSPHEnv, not of ObservableSloshingEnv -- rather
than reusing inheritance across the two. What IS reused, unchanged, is the
actual sensor-proxy math: environment/observable_sloshing_env.py::wall_pressures()
was extracted to a standalone function specifically so both environments
call the exact same formula instead of maintaining two copies of it.

UNTESTED as of 2026-09-16 -- constructing ClosedLoopSPHTank needs a real
CUDA GPU with the from-source Chrono::FSI build (see the project notes'
"Second machine note"), which was busy running the closed-loop fine-tuning
campaign when this was written. Written and reviewed carefully (mirrors
ObservableSloshingEnv's already-tested pattern exactly, and ClosedLoopSPHEnv's
own _build_observation()/step()/reset() were read directly to match its
real structure, not guessed), but do not treat this as confirmed working
until it has actually been run once the GPU is free. See this project's
own "don't design blind" convention -- if anything here doesn't match
ClosedLoopSPHTank's/ClosedLoopSPHEnv's real behavior once tested, fix this
file, don't paper over the mismatch.

Observation (float32, shape (7,)):
    [tank_angle, tank_angular_velocity,
     pressure_left, pressure_right, delta_pressure, delta_pressure_dot,
     previous_torque_nm]
(identical layout to ObservableSloshingEnv's, so a policy/VecNormalize
trained against one transfers its observation SHAPE to the other --
scale/dynamics still differ, see the project notes' own notes on that.)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig, WorldConfig
from environment.closed_loop_sph_env import ClosedLoopSPHEnv
from environment.observable_sloshing_env import wall_pressures

_OBSERVATION_LOW = np.array(
    [-np.inf, -np.inf, 0.0, 0.0, -np.inf, -np.inf, -np.inf], dtype=np.float32
)
_OBSERVATION_HIGH = np.full((7,), np.inf, dtype=np.float32)


class ObservableClosedLoopSPHEnv(ClosedLoopSPHEnv):
    """ClosedLoopSPHEnv with the same hardware-deployable sensor observation
    ObservableSloshingEnv uses for the fast surrogate.

    Only `_build_observation()` is overridden -- `reset()`/`step()` are
    inherited from ClosedLoopSPHEnv (its exact FSI-scene-rebuild,
    mid-substep safety-trip, and termination logic all still apply
    unchanged) except for the two small additions ObservableSloshingEnv
    also needed: zeroing this class's own extra state in reset(), and
    capturing the just-applied physical torque in step() before delegating
    to the base class (whose own step() calls self._build_observation()
    via ordinary Python polymorphism at the very end).
    """

    def __init__(
        self,
        tank_config: Optional[TankConfig] = None,
        fluid_config: Optional[FluidConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
        world_config: Optional[WorldConfig] = None,
        reset_angular_velocity_rad_s: float = 0.05,
        artificial_viscosity: float = 0.03,
        boundary_method: str = "adami",
    ) -> None:
        # world_config isn't part of ClosedLoopSPHEnv's own constructor at
        # all (ClosedLoopSPHTank hardcodes gravity via WorldConfig()
        # defaults internally, same as this env needs for the pressure
        # proxy) -- accepted here only so gravity can be overridden
        # consistently with ObservableSloshingEnv's own constructor shape,
        # not because the base class uses it.
        self._world_config = world_config or WorldConfig()

        super().__init__(
            tank_config=tank_config,
            fluid_config=fluid_config,
            env_config=env_config,
            reset_angular_velocity_rad_s=reset_angular_velocity_rad_s,
            artificial_viscosity=artificial_viscosity,
            boundary_method=boundary_method,
        )

        # Overrides the base class's observation_space -- same reasoning as
        # ObservableSloshingEnv: the pressures this class reports are
        # non-negative by construction (wall_pressures()'s own max(0, ...)
        # clamp), so the space itself documents that rather than reusing
        # the base class's all-real bounds.
        self.observation_space = spaces.Box(
            low=_OBSERVATION_LOW, high=_OBSERVATION_HIGH, dtype=np.float32
        )

        self._prev_applied_torque = 0.0
        self._prev_delta_pressure = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Reset this class's own extra state BEFORE calling super().reset()
        # -- ClosedLoopSPHEnv.reset() calls self._build_observation() (via
        # polymorphism, this class's override) at its very end, so these
        # need to already be at their zeroed values by then. Identical
        # ordering reasoning to ObservableSloshingEnv.reset().
        self._prev_applied_torque = 0.0
        self._prev_delta_pressure = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # ClosedLoopSPHEnv.step() computes the physical torque locally
        # (normalized action * max_motor_torque_nm) and never stores it --
        # recomputed here with the IDENTICAL formula so this observation
        # can report it, rather than modifying closed_loop_sph_env.py just
        # to expose an attribute one subclass wants. Same pattern as
        # ObservableSloshingEnv.step().
        normalized_action = float(
            np.clip(action, self.action_space.low, self.action_space.high)[0]
        )
        physical_torque = normalized_action * self._env_config.max_motor_torque_nm

        # Set BEFORE calling super().step() -- its own step() calls
        # self._build_observation() (this class's override) at its end (on
        # every return path: the two early-termination safety branches AND
        # the normal path), so "previous_torque" in the observation this
        # call returns means "the torque this exact action just applied."
        self._prev_applied_torque = physical_torque
        return super().step(action)

    def _build_observation(self) -> np.ndarray:
        # self._tank is ClosedLoopSPHEnv's own attribute (the real
        # ClosedLoopSPHTank instance) -- get_fluid_state() is the same
        # backend-agnostic FluidState interface ObservableSloshingEnv reads
        # from a FluidBackend, so wall_pressures() below needs no changes
        # to work with either.
        fluid_state = self._tank.get_fluid_state()
        com_x = fluid_state.com_position[0]
        com_vx = fluid_state.com_velocity[0]

        gravity = abs(self._world_config.gravity[1])
        pressure_left, pressure_right = wall_pressures(
            com_x, com_vx, self._tank_config, self._fluid_config, gravity
        )
        delta_pressure = pressure_left - pressure_right
        delta_pressure_dot = (
            delta_pressure - self._prev_delta_pressure
        ) / self._env_config.control_dt
        self._prev_delta_pressure = delta_pressure

        return np.array(
            [
                self._tank.angle,
                self._tank.angular_velocity,
                pressure_left,
                pressure_right,
                delta_pressure,
                delta_pressure_dot,
                self._prev_applied_torque,
            ],
            dtype=np.float32,
        )
