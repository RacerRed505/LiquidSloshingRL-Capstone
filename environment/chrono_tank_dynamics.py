"""
Phase 10 -- Real ChSystem-backed tank rotational dynamics.

environment/tank_dynamics.py::TankDynamicsModel is an analytic, PyChrono-
free surrogate, deliberately built that way so environment/sloshing_env.py
could train fast (see that module's own docstring). This module is its
real counterpart: a small adapter around the actual Phase 4-6 builders
(WorldBuilder, TankBuilder, PivotBuilder, MotorBuilder with the new
MotorKind.TORQUE) that exposes the EXACT SAME reset()/step()/.angle/
.angular_velocity surface, so it can be dropped into
SloshingSuppressionEnv's optional `tank_dynamics` constructor argument
without that file changing at all.

Unlike environment/tank_dynamics.py, this module DOES import pychrono at
module level -- same as builders/tank_builder.py and main.py already do
-- so it requires the real PyChrono (conda-forge) bindings to import at
all. It is NOT used by default, and is NOT part of the fast PPO training
loop (training/train_ppo.py still uses the analytic surrogate) -- it
exists for validating a surrogate-trained policy against the real
physics, and eventually for training against it directly once that's
been decided to be worth the throughput cost (see this module's own
"Known limitations" section below).

Design choice: reset() rebuilds the entire ChSystem/tank/pivot/motor from
scratch rather than trying to reset the existing bodies' kinematic state
in place. This is slower per-episode than an in-place reset would be, but
it's the same construction sequence main.py already uses (confirmed
working on real hardware), rather than new, unverified Chrono-API usage
(zeroing a ChLinkMotorRotationTorque's internal state cleanly is not
something this project has tested) for a performance win that hasn't
been shown to be necessary yet. Revisit if profiling on real PyChrono
shows reset() cost actually dominates.

Known limitations (read before using this for training, not just
validation):
- Every real ChSystem step costs orders of magnitude more wall-clock time
  than TankDynamicsModel's analytic integration -- main.py's Irrlicht
  demo ran close to real-time for a SINGLE instance. Training used 8
  parallel envs of the fast surrogate; running 8 parallel real ChSystem
  instances will need SB3's SubprocVecEnv (true multiprocessing, PyChrono
  is not free-threading-safe the way pure Python is) instead of the
  DummyVecEnv training/train_ppo.py currently uses, and will still be
  far slower per env even so.
- Because of the above, the recommended first use of this module is
  scripts/validate_real_dynamics.py -- evaluating the ALREADY-TRAINED
  surrogate policy against real physics, not training from scratch here.
  That's a cheap way to find out whether the surrogate is a good enough
  proxy before deciding whether a slow real-physics retrain is even
  needed.
"""

from __future__ import annotations

from typing import Optional

import pychrono.core as chrono

from builders.motor_builder import MotorBuilder
from builders.pivot_builder import PivotBuilder
from builders.tank_builder import TankBuilder
from builders.world_builder import WorldBuilder
from config.simulation_config import (
    EnvironmentConfig,
    MotorConfig,
    MotorKind,
    PivotConfig,
    TankConfig,
    WorldConfig,
)


class ChronoTankDynamics:
    """Real ChSystem-backed counterpart to TankDynamicsModel.

    Exposes reset(seed)/step(dt, commanded_torque)/.angle/
    .angular_velocity/.torque_limit -- the same duck-typed surface
    environment/sloshing_env.py already calls on TankDynamicsModel, so it
    can be passed to SloshingSuppressionEnv's `tank_dynamics=` argument
    as a straight substitute.
    """

    def __init__(
        self,
        tank_config: Optional[TankConfig] = None,
        pivot_config: Optional[PivotConfig] = None,
        world_config: Optional[WorldConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
    ) -> None:
        # fixed=False is required -- see TankConfig.fixed's own docstring;
        # the tank must be free to rock under the motor joint, not welded
        # to the world the way Phase 4-5's own checkpoint used it.
        self._tank_config = tank_config or TankConfig(fixed=False)
        if self._tank_config.fixed:
            raise ValueError(
                "ChronoTankDynamics requires TankConfig(fixed=False) -- a fixed "
                "tank has no pivot motion for the motor to actuate."
            )
        self._pivot_config = pivot_config or PivotConfig()
        self._world_config = world_config or WorldConfig()
        self._env_config = env_config or EnvironmentConfig()
        self._motor_config = MotorConfig(kind=MotorKind.TORQUE)

        self._torque_limit = self._env_config.max_motor_torque_nm

        self._system: Optional[chrono.ChSystem] = None
        self._motor: Optional["chrono.ChLinkMotorRotationTorque"] = None
        self._build_system()

    def _build_system(self) -> None:
        """(Re)build a fresh world/tank/pivot/motor -- see module docstring."""
        system = WorldBuilder(self._world_config).build()
        tank_body = TankBuilder(self._tank_config).build(system)
        pivot_body = PivotBuilder(self._pivot_config).build(system)
        motor = MotorBuilder(self._pivot_config, self._motor_config).build(
            system, pivot_body, tank_body
        )
        self._system = system
        self._motor = motor

    def reset(self, seed: Optional[int] = None) -> None:
        """Rebuild the ChSystem from scratch -- see module docstring for why."""
        # seed is accepted (not used) purely to match TankDynamicsModel's
        # signature -- this adapter has no stochastic element of its own;
        # any reset-time randomization (the fluid disturbance impulse)
        # lives in SloshingSuppressionEnv.reset(), applied to the fluid
        # backend, not here.
        self._build_system()

    def step(self, dt: float, commanded_torque: float) -> None:
        """Advance the real ChSystem by dt seconds under a clipped commanded torque."""
        torque = max(-self._torque_limit, min(self._torque_limit, commanded_torque))
        MotorBuilder.set_commanded_torque(self._motor, torque)
        self._system.DoStepDynamics(dt)

    @property
    def angle(self) -> float:
        return MotorBuilder.read_tank_motion(self._motor)[0]

    @property
    def angular_velocity(self) -> float:
        return MotorBuilder.read_tank_motion(self._motor)[1]

    @property
    def torque_limit(self) -> float:
        return self._torque_limit
