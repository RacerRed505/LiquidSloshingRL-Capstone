"""
Phase 8 -- Gymnasium environment for liquid-sloshing suppression.

This is Layer 3 of the project's five-layer architecture. It is written
against physics/backend_interface.py::FluidBackend only -- never against
PendulumBackend or SPHBackend directly (see that module's own docstring)
-- and against environment/tank_dynamics.py::TankDynamicsModel's small
reset/step/angle/angular_velocity surface, not against pychrono. That is
what let Phase 10 swap in a real ChSystem-backed tank
(environment/chrono_tank_dynamics.py::ChronoTankDynamics, injectable via
this class's `tank_dynamics` constructor argument, same pattern as
`fluid_backend`) without this file's step()/reset() logic changing at
all -- and is what will let Phase 13-14 swap PendulumBackend for
SPHBackend the same way.

Task
----
An episode starts with the tank at rest and the fluid given a small
random disturbance (domain randomization -- see
EnvironmentConfig.reset_disturbance_impulse_range). The agent commands a
torque about the pivot each control_dt; the objective is to damp out the
resulting slosh (minimize FluidState.slosh_energy) without excessive
control effort or parking the tank at an extreme tilt. See
EnvironmentConfig's own docstring for the reward weights.

Observation (float32, shape (7,)):
    [tank_angle, tank_angular_velocity,
     fluid_com_x, fluid_com_y, fluid_com_vx, fluid_com_vy,
     slosh_energy]

Action (float32, shape (1,)):
    normalized commanded torque about the pivot axis, in [-1, 1] --
    scaled internally to [-max_motor_torque_nm, +max_motor_torque_nm].
    Kept symmetric and normalized rather than exposing raw N*m directly,
    per Gymnasium's own env_checker guidance (SB3's PPO -- the Phase 9
    algorithm this project's roadmap names -- assumes actions live on a
    consistent, bounded scale rather than a physically-unit-dependent
    one).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from builders.fluid_builder import FluidBuilder
from config.simulation_config import (
    EnvironmentConfig,
    FluidConfig,
    PivotConfig,
    TankConfig,
    WorldConfig,
)
from environment.tank_dynamics import TankDynamicsModel, build_tank_dynamics_model
from physics.backend_interface import FluidBackend

logger = logging.getLogger(__name__)

_OBSERVATION_LOW = np.array(
    [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32
)
_OBSERVATION_HIGH = np.full((7,), np.inf, dtype=np.float32)


class SloshingSuppressionEnv(gym.Env):
    """Gymnasium environment: suppress tank sloshing via commanded pivot torque.

    Constructed from the project's own config dataclasses so its physical
    parameters stay identical to whatever `main.py` uses -- passing in a
    non-default TankConfig/FluidConfig/etc. here trains a policy for a
    correspondingly different physical tank, rather than silently
    diverging from it.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

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
        super().__init__()

        self._tank_config = tank_config or TankConfig(fixed=False)
        self._pivot_config = pivot_config or PivotConfig()
        self._fluid_config = fluid_config or FluidConfig()
        self._world_config = world_config or WorldConfig()
        self._env_config = env_config or EnvironmentConfig()

        # `fluid_backend` is injectable so a) tests can pass in a fake/stub
        # FluidBackend without touching Housner math, and b) Phase 13-14
        # can hand this class an SPHBackend instance instead of rebuilding
        # this constructor -- consistent with "write against the
        # interface, not the concrete class."
        self._fluid_backend: FluidBackend = fluid_backend or FluidBuilder(
            self._tank_config, self._fluid_config, self._world_config
        ).build()

        # `tank_dynamics` is injectable for the exact same reason, added
        # Phase 10: environment/chrono_tank_dynamics.py::ChronoTankDynamics
        # exposes the same reset/step/angle/angular_velocity surface as
        # the default TankDynamicsModel surrogate built below, so a caller
        # (e.g. scripts/validate_real_dynamics.py) can hand in the real
        # ChSystem-backed adapter without this class needing to change --
        # type-hinted as TankDynamicsModel for the common case, but only
        # duck-typed against in practice (see step()/reset() below), so
        # ChronoTankDynamics (not a TankDynamicsModel subclass) works too.
        self._tank_dynamics: TankDynamicsModel = tank_dynamics or build_tank_dynamics_model(
            self._tank_config, self._pivot_config, self._env_config
        )

        # Normalized [-1, 1] action space -- see this module's docstring
        # for why. self._env_config.max_motor_torque_nm is the physical
        # N*m scale factor applied in step(), not the action bound itself.
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=_OBSERVATION_LOW, high=_OBSERVATION_HIGH, dtype=np.float32
        )

        self._step_count = 0
        self._np_random: Optional[np.random.Generator] = None

    # -- Gymnasium API -----------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)  # sets self.np_random per Gymnasium convention

        self._fluid_backend.reset(seed=seed)
        self._tank_dynamics.reset(seed=seed)
        self._step_count = 0

        low, high = self._env_config.reset_disturbance_impulse_range
        if high > 0.0:
            # A random planar impulse so episodes don't all start from a
            # trivially-already-solved dead-flat fluid -- see
            # EnvironmentConfig.reset_disturbance_impulse_range's own
            # docstring.
            magnitude = self.np_random.uniform(low, high)
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
            impulse = (magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0)
            self._fluid_backend.apply_disturbance(impulse)

        observation = self._build_observation()
        info = self._build_info()
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        normalized_action = float(np.clip(action, self.action_space.low, self.action_space.high)[0])
        torque = normalized_action * self._env_config.max_motor_torque_nm

        for _ in range(self._env_config.substeps_per_control_step):
            self._tank_dynamics.step(self._env_config.physics_dt, torque)
            tank_motion = (self._tank_dynamics.angle, self._tank_dynamics.angular_velocity)
            self._fluid_backend.step(self._env_config.physics_dt, tank_motion)

        self._step_count += 1

        fluid_state = self._fluid_backend.get_state()
        tank_angle = self._tank_dynamics.angle

        reward = self._compute_reward(fluid_state.slosh_energy, torque, tank_angle)

        terminated = abs(tank_angle) > self._env_config.max_tank_angle_rad
        truncated = self._step_count >= self._env_config.max_episode_steps

        observation = self._build_observation()
        info = self._build_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        # Phase 8's checkpoint is the RL interface itself. Live 3D
        # visualization already exists for the real PyChrono tank
        # (main.py's Irrlicht window, Phase 6/7's checkpoint) -- this
        # surrogate model has no PyChrono scene to render, so rendering is
        # intentionally a documented no-op rather than a half-built stub.
        logger.debug(
            "SloshingSuppressionEnv.render() is a no-op -- see this method's docstring."
        )

    def close(self) -> None:
        pass

    # -- internals -----------------------------------------------------

    def _compute_reward(self, slosh_energy: float, torque: float, tank_angle: float) -> float:
        cfg = self._env_config
        return -(
            cfg.slosh_energy_weight * slosh_energy
            + cfg.control_effort_weight * (torque**2)
            + cfg.angle_penalty_weight * (tank_angle**2)
        )

    def _build_observation(self) -> np.ndarray:
        fluid_state = self._fluid_backend.get_state()
        return np.array(
            [
                self._tank_dynamics.angle,
                self._tank_dynamics.angular_velocity,
                fluid_state.com_position[0],
                fluid_state.com_position[1],
                fluid_state.com_velocity[0],
                fluid_state.com_velocity[1],
                fluid_state.slosh_energy,
            ],
            dtype=np.float32,
        )

    def _build_info(self) -> Dict[str, Any]:
        fluid_state = self._fluid_backend.get_state()
        return {
            "tank_angle": self._tank_dynamics.angle,
            "tank_angular_velocity": self._tank_dynamics.angular_velocity,
            "slosh_energy": fluid_state.slosh_energy,
            "slosh_potential_energy": fluid_state.slosh_potential_energy,
            "slosh_total_energy": fluid_state.slosh_total_energy,
            "fluid_backend_name": self._fluid_backend.name,
            "step_count": self._step_count,
        }
