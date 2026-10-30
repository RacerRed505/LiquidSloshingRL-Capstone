"""
Phase 15 -- Gymnasium wrapper around physics/closed_loop_sph_tank.py::ClosedLoopSPHTank,
so the existing PPO policy/training machinery (training/train_ppo_real_physics.py's
own pattern) can fine-tune against the real, two-way-coupled FSI closed loop
directly, instead of only running an already-trained policy against it
open-loop (scripts/run_closed_loop_sph_policy.py).

Written after the development log Entry 55/56's own findings: the closed loop's
real failure mode (a sudden, real control breakdown after an extended period
of good behavior, not a numerics bug -- confirmed by testing higher
artificial_viscosity, which prevented the SPH solver crash but did NOT
prevent the underlying loss of control, Entry 56) means torque-limit/SPH-
parameter tuning alone plateaus. This environment exists to let the policy
itself adapt to the real coupled dynamics via fine-tuning, the same way
Phase 11 closed the gap between the surrogate and the real (non-FSI)
ChSystem tank.

Mirrors environment/sloshing_env.py::SloshingSuppressionEnv's observation
space (7,), action space (1,) in [-1, 1], and reward function EXACTLY (same
EnvironmentConfig weights) so a policy/VecNormalize stats already trained
there transfer directly -- only the underlying physics differs. Real,
necessary differences from SloshingSuppressionEnv, stated plainly rather
than silently glossed over:

- **Reset disturbance mechanism differs.** SloshingSuppressionEnv applies a
  random PLANAR IMPULSE TO THE FLUID (FluidBackend.apply_disturbance()).
  ClosedLoopSPHTank.apply_disturbance() does not exist (SPHBackend/
  ClosedLoopSPHTank's design, Entry 41/48, never implemented fluid-impulse
  injection for the FSI case) -- the substitute is an ANGULAR-VELOCITY KICK
  TO THE TANK ITSELF, built into ClosedLoopSPHTank.reset()'s own
  `reset_angular_velocity_rad_s` parameter. This is a real, already-
  documented deviation (see that class's own docstring), not something this
  wrapper can paper over.
- **No manual substep loop needed.** SloshingSuppressionEnv loops
  `substeps_per_control_step` times, stepping tank_dynamics and
  fluid_backend separately at `physics_dt` each -- because those are two
  independently-stepped objects (one-way coupling). ClosedLoopSPHTank is a
  single, genuinely two-way-coupled object; its own `step(dt)` already
  subdivides `dt` into its own internal `step_size` increments (the Entry
  55 fix). This wrapper therefore calls `tank.step(env_config.control_dt)`
  ONCE per Gym step, not `substeps_per_control_step` times at `physics_dt`.
- **Episode-length note:** rebuilding the whole FSI scene from scratch
  (ClosedLoopSPHTank.reset()) is expensive -- expect real per-episode
  construction cost on top of the already-slow per-step cost (~35x slower
  than real-time, Entries 54-56). This environment is NOT suitable for
  DummyVecEnv/SubprocVecEnv parallelism at any meaningful n_envs on a
  single consumer GPU -- see training/train_ppo_closed_loop_sph.py's own
  docstring for why it defaults to n_envs=1.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pychrono.fsi as fsi
from gymnasium import spaces

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig
from physics.closed_loop_sph_tank import ClosedLoopSPHTank

_BOUNDARY_METHOD_VALUES = {
    "adami": fsi.BoundaryMethod_ADAMI,
    "holmes": fsi.BoundaryMethod_HOLMES,
}

logger = logging.getLogger(__name__)

_OBSERVATION_LOW = np.array(
    [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32
)
_OBSERVATION_HIGH = np.full((7,), np.inf, dtype=np.float32)


class ClosedLoopSPHEnv(gym.Env):
    """Gymnasium environment: suppress tank sloshing via commanded pivot
    torque, against a REAL, two-way-coupled Chrono::FSI fluid -- not the
    fast surrogate (SloshingSuppressionEnv's default) or the real but
    non-FSI rigid-body tank (ChronoTankDynamics). See this module's own
    docstring for exactly how this differs from SloshingSuppressionEnv.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        tank_config: Optional[TankConfig] = None,
        fluid_config: Optional[FluidConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
        reset_angular_velocity_rad_s: float = 0.05,
        artificial_viscosity: float = 0.03,
        boundary_method: str = "adami",
    ) -> None:
        super().__init__()

        self._tank_config = tank_config or TankConfig(fixed=False)
        self._fluid_config = fluid_config or FluidConfig()
        self._env_config = env_config or EnvironmentConfig()

        # Added 2026-09-22 (the development log Entry 119) so a HOLMES-boundary
        # training run is actually possible -- previously only the raw
        # ClosedLoopSPHTank class and the one-off evaluation script
        # (scripts/run_closed_loop_sph_policy.py) exposed this; the real
        # PPO training path never did. Default ("adami") preserves every
        # existing call site's exact prior behavior unchanged.
        self._tank = ClosedLoopSPHTank(
            tank_config=self._tank_config,
            fluid_config=self._fluid_config,
            env_config=self._env_config,
            reset_angular_velocity_rad_s=reset_angular_velocity_rad_s,
            artificial_viscosity=artificial_viscosity,
            boundary_method=_BOUNDARY_METHOD_VALUES[boundary_method],
        )

        # Same normalized [-1, 1] action space as SloshingSuppressionEnv --
        # see that class's own docstring for why.
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=_OBSERVATION_LOW, high=_OBSERVATION_HIGH, dtype=np.float32
        )

        self._step_count = 0
        self._episode_count = 0
        self._np_random: Optional[np.random.Generator] = None

    # -- Gymnasium API -----------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)  # sets self.np_random per Gymnasium convention

        # Rebuilds the ENTIRE FSI scene from scratch -- expensive, but
        # required (ClosedLoopSPHTank.reset()'s own docstring explains why
        # an already-Initialize()'d FSI problem can't be reconfigured in
        # place). The angular-velocity disturbance is applied inside this
        # call, per this module's own docstring on why that differs from
        # SloshingSuppressionEnv's fluid-impulse mechanism.
        # Logged explicitly (not just step progress) because repeated
        # mid-process FSI scene reconstruction -- as real termination now
        # triggers, unlike the single-continuous-episode diagnostic runs
        # in the development log Entry 56 -- is itself an untested scenario,
        # per Entry 57's own investigation.
        self._episode_count += 1
        logger.info("ClosedLoopSPHEnv.reset() -- episode #%d (rebuilding FSI scene from scratch)", self._episode_count)
        self._tank.reset(seed=seed)
        self._step_count = 0

        observation = self._build_observation()
        info = self._build_info()
        return observation, info

    _ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S = 1.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # PROACTIVE safety check, added after a real crash investigation
        # (the development log Entry 57): the angle-based termination below can
        # only catch a bad state AFTER a full control_dt step completes --
        # but the SPH solver's real crashes (Entries 55-57) can go from a
        # moderate excursion to a full NaN blowup WITHIN a single
        # control_dt's ~200 internal substeps, faster than any check after
        # the fact can catch. Every observed crash across many real trials
        # was preceded by angular velocity reaching roughly 1.0-1.2 rad/s
        # -- checking THIS step's starting angular velocity (the previous
        # step's result) and terminating BEFORE attempting another
        # potentially-crash-inducing step is a real, evidence-based
        # mitigation, not a guess. This trades a slightly more conservative
        # (shorter) episode for avoiding the unrecoverable native-process
        # abort entirely.
        #
        # UPDATE (Entry 58): this between-step check alone was found to be
        # too coarse -- a real 16,384-timestep fine-tune crashed inside a
        # single control step's substeps, after this check had already
        # passed. ClosedLoopSPHTank.step() now ALSO checks after every
        # internal substep (~1e-4s granularity) and returns False if it
        # trips -- passed as the same limit here so both the before-step
        # and mid-step guards agree.
        if self._step_count > 0 and abs(self._tank.angular_velocity) > self._ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S:
            logger.warning(
                "ep#%d step %d: angular_velocity=%.4f exceeds safety limit (%.2f) -- "
                "terminating BEFORE attempting another step, not after a crash.",
                self._episode_count, self._step_count, self._tank.angular_velocity,
                self._ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S,
            )
            fluid_state = self._tank.get_fluid_state()
            tank_angle = self._tank.angle
            torque = 0.0  # no step was actually taken this call
            reward = self._compute_reward(fluid_state.slosh_energy, torque, tank_angle)
            observation = self._build_observation()
            info = self._build_info()
            return observation, reward, True, False, info

        normalized_action = float(np.clip(action, self.action_space.low, self.action_space.high)[0])
        torque = normalized_action * self._env_config.max_motor_torque_nm
        self._tank.set_commanded_torque(torque)

        # ClosedLoopSPHTank.step() already subdivides internally (Entry 55
        # fix) -- ONE call here covers a full control_dt, unlike
        # SloshingSuppressionEnv's own substeps_per_control_step loop over
        # two SEPARATELY-stepped objects (see this module's docstring).
        # Entry 58: now passes the safety limit through so a mid-step
        # blowup stops the substep loop immediately instead of running to
        # completion and crashing.
        completed = self._tank.step(
            self._env_config.control_dt,
            angular_velocity_safety_limit=self._ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S,
        )
        self._step_count += 1

        if not completed:
            logger.warning(
                "ep#%d step %d: mid-step safety trip -- angular_velocity=%.4f exceeded "
                "safety limit (%.2f) WITHIN this control step's substeps, stopped early "
                "(the development log Entry 58).",
                self._episode_count, self._step_count, self._tank.angular_velocity,
                self._ANGULAR_VELOCITY_SAFETY_LIMIT_RAD_S,
            )
            fluid_state = self._tank.get_fluid_state()
            tank_angle = self._tank.angle
            reward = self._compute_reward(fluid_state.slosh_energy, torque, tank_angle)
            observation = self._build_observation()
            info = self._build_info()
            return observation, reward, True, False, info

        fluid_state = self._tank.get_fluid_state()
        tank_angle = self._tank.angle

        reward = self._compute_reward(fluid_state.slosh_energy, torque, tank_angle)

        # CRITICAL, not present in scripts/run_closed_loop_sph_policy.py's
        # own open-loop runner: terminate once the tank exceeds the same
        # safety bound SloshingSuppressionEnv itself trains against. This
        # matters specifically for FINE-TUNING -- without it, a policy
        # already known to lose control (Entries 55-56's wild multi-
        # rotation spin) would keep contributing nonsensical, already-
        # failed-state transitions to the training signal instead of
        # ending the episode and starting a fresh one, corrupting learning.
        terminated = abs(tank_angle) > self._env_config.max_tank_angle_rad
        truncated = self._step_count >= self._env_config.max_episode_steps

        if self._step_count % 10 == 0 or terminated or truncated:
            logger.info(
                "ep#%d step %d: angle=%.4f torque=%.4f slosh=%.5f terminated=%s truncated=%s",
                self._episode_count, self._step_count, tank_angle, torque,
                fluid_state.slosh_energy, terminated, truncated,
            )

        observation = self._build_observation()
        info = self._build_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        logger.debug("ClosedLoopSPHEnv.render() is a no-op, matching SloshingSuppressionEnv.")

    def close(self) -> None:
        pass

    # -- internals -----------------------------------------------------

    def _compute_reward(self, slosh_energy: float, torque: float, tank_angle: float) -> float:
        # Identical formula to SloshingSuppressionEnv._compute_reward --
        # same EnvironmentConfig weights, so a fine-tuned policy's reward
        # scale stays comparable to what it was originally trained against.
        cfg = self._env_config
        return -(
            cfg.slosh_energy_weight * slosh_energy
            + cfg.control_effort_weight * (torque**2)
            + cfg.angle_penalty_weight * (tank_angle**2)
        )

    def _build_observation(self) -> np.ndarray:
        fluid_state = self._tank.get_fluid_state()
        return np.array(
            [
                self._tank.angle,
                self._tank.angular_velocity,
                fluid_state.com_position[0],
                fluid_state.com_position[1],
                fluid_state.com_velocity[0],
                fluid_state.com_velocity[1],
                fluid_state.slosh_energy,
            ],
            dtype=np.float32,
        )

    def _build_info(self) -> Dict[str, Any]:
        fluid_state = self._tank.get_fluid_state()
        return {
            "tank_angle": self._tank.angle,
            "tank_angular_velocity": self._tank.angular_velocity,
            "slosh_energy": fluid_state.slosh_energy,
            "fluid_backend_name": "closed_loop_sph",
            "step_count": self._step_count,
        }
