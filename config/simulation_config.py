"""
Configuration dataclasses for Phase 4 (World Builder) and Phase 5
(Procedural Tank Generation).

Per the project rules established in Module 1, no magic numbers live inside
builders/ -- every tunable physical or numerical parameter is declared here,
with a sensible default and a short justification. This file will grow as
later phases (motor, pivot, RL) are added. It deliberately does NOT use
YAML/JSON yet -- see the architecture note from Module 1: config files are
introduced at Phase 16 (Domain Randomization), where sweeping parameters
without code changes actually becomes necessary. Until then, frozen
dataclasses keep the config type-checked, diffable in git, and impossible
to mutate accidentally mid-simulation.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ContactMethod(str, Enum):
    """Which PyChrono contact formulation the world uses.

    SMC (Smooth Contact / penalty-based) is the project default: for a
    small number of persistent, well-defined contacts (tank walls, and
    later the pivot bearing) it is cheaper to tune than NSC's
    complementarity solver and degrades gracefully rather than requiring
    iterative solver convergence at every step -- which matters once this
    system is stepped at RL-training throughput (Phase 10 onward), not
    just real-time visualization speed.
    """

    SMC = "SMC"
    NSC = "NSC"


@dataclass(frozen=True)
class WorldConfig:
    """Parameters for the empty PyChrono physics world (Phase 4)."""

    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    contact_method: ContactMethod = ContactMethod.SMC
    time_step: float = 1.0e-3
    solver_max_iterations: int = 150
    enable_gravity: bool = True


@dataclass(frozen=True)
class TankConfig:
    """Parameters for the procedurally generated open-top tank (Phase 5).

    The tank is built from five rectangular components: one base slab plus
    four walls. All dimensions are OUTER dimensions in meters; the interior
    cavity is (length - 2*wall_thickness) x height x (width - 2*wall_thickness).
    The tank's local origin is the center of the footprint at interior floor
    level (y = 0); `position` places that origin in world coordinates.
    """

    length: float = 0.40  # outer footprint size along x, m
    width: float = 0.25  # outer footprint size along z, m
    height: float = 0.30  # wall height along y, m
    wall_thickness: float = 0.006  # m (~6 mm, typical cast-acrylic sheet)
    density: float = 1180.0  # kg/m^3 (cast acrylic / PMMA)

    friction: float = 0.35
    restitution: float = 0.05
    # Real acrylic Young's modulus (~3.2 GPa) makes SMC penalty-contact
    # stiffness require an impractically small time step for this project's
    # purposes. These are deliberately softened, simulation-stability
    # values, not measurement errors -- flagged here per the "no silent
    # physics shortcuts" rule from Module 1, so it's visible in one place
    # rather than buried in builder code.
    contact_young_modulus: float = 5.0e6
    contact_poisson_ratio: float = 0.35

    # World-frame position of the tank's local origin (footprint center,
    # interior floor level).
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # The tank has no motor/pivot joint until Phase 6, so it is held fixed
    # by default for the Phase 4-5 "working simulation" checkpoint (world
    # opens, tank appears, nothing falls through the floor). Phase 6 will
    # override this to False once a joint replaces the fixed constraint.
    fixed: bool = True

    def __post_init__(self) -> None:
        if self.wall_thickness * 2 >= min(self.length, self.width):
            raise ValueError(
                "wall_thickness is too large relative to length/width; "
                "the interior cavity would have zero or negative size."
            )
        if self.wall_thickness <= 0 or self.height <= 0:
            raise ValueError("wall_thickness and height must be positive.")


@dataclass(frozen=True)
class FluidConfig:
    """Parameters for the Phase 7 equivalent-pendulum fluid surrogate.

    Tank footprint (length, width) is read from TankConfig by
    builders/fluid_builder.py rather than duplicated here -- fill_height is
    the one fluid-specific geometric parameter TankConfig doesn't already
    describe, since the same tank can hold different fill levels.
    """

    fill_height: float = 0.20  # m; must be less than TankConfig.height
    density: float = 1000.0  # kg/m^3 (water)

    # Sloshing damping is not predicted by the inviscid potential-flow
    # theory the Housner frequency/mass formulas come from -- it has to be
    # an empirical input. 0.5% of critical is the standard literature value
    # for the convective (sloshing) mode of a smooth-walled tank holding
    # water (see e.g. Sivy et al. 2017, MATEC Web Conf. 107:00069,
    # doi:10.1051/matecconf/201710700069, which uses 0.5% liquid damping vs.
    # 2-5% for the surrounding steel/concrete *structure* -- the higher
    # values reported elsewhere in that literature are for the structure,
    # not the liquid, and don't apply here).
    damping_ratio: float = 0.005

    def __post_init__(self) -> None:
        if self.fill_height <= 0:
            raise ValueError("fill_height must be positive.")
        if not (0.0 <= self.damping_ratio < 1.0):
            raise ValueError("damping_ratio must be in [0, 1).")


@dataclass(frozen=True)
class PivotConfig:
    """Parameters for the Phase 6 pivot support (builders/pivot_builder.py).

    The pivot is a small, fixed pedestal, anchored to ground, that the
    tank's motor joint (builders/motor_builder.py) attaches to -- giving
    the joint a physical mounting point rather than joining the tank
    straight to an implicit world frame. It is a plain box primitive,
    sized only to be a plausible mechanical stand-in (same "no CAD"
    approach as tank_builder.py); it carries no reward-relevant physics of
    its own and is not load-bearing-accurate.
    """

    # World-frame location of the pivot axis. Defaults to TankConfig's own
    # local origin (footprint center, floor level, y=0) -- the tank rocks
    # about a point at its own base, matching the "tips over its
    # front/back edge" mental model of a rocking tray. Because the support
    # pedestal is collision-free (see support_size/EnableCollision below),
    # this coincidence with the tank's floor origin is a harmless visual
    # overlap at this checkpoint, not a physics bug -- flagged here rather
    # than papered over with an arbitrary offset that isn't dimensioned to
    # anything.
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Rotation axis for the rocking motion, in world/parent-frame
    # coordinates -- +Z (out of the x-y plane) so the tank tips in the x-y
    # plane, i.e. along TankConfig.length, matching the axis
    # utilities/housner_utils.py's tank_length assumes the sloshing
    # excitation acts along (see FluidBuilder's own docstring).
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Dimensions of the pedestal block itself -- a plain box, no CAD.
    support_size: Tuple[float, float, float] = (0.05, 0.05, 0.05)  # m
    support_density: float = 2700.0  # kg/m^3 (aluminum stand-in)

    def __post_init__(self) -> None:
        if any(dim <= 0 for dim in self.support_size):
            raise ValueError("support_size dimensions must all be positive.")
        if all(component == 0.0 for component in self.axis):
            raise ValueError("axis must not be the zero vector.")


class MotorKind(str, Enum):
    """Which PyChrono ChLinkMotorRotation subclass drives the pivot joint.

    ANGLE (Phase 6's scripted open-loop rock) and TORQUE (Phase 10's
    RL-driven control, added this session -- see builders/motor_builder.py)
    are implemented. SPEED remains a placeholder, declared here so a
    future addition doesn't require a new config class or a breaking
    rename of this one.
    """

    ANGLE = "ANGLE"  # ChLinkMotorRotationAngle -- prescribes theta(t) directly
    SPEED = "SPEED"  # ChLinkMotorRotationSpeed -- prescribes theta_dot(t); not yet wired up
    TORQUE = "TORQUE"  # ChLinkMotorRotationTorque -- prescribes torque(t); Phase 10 RL control


@dataclass(frozen=True)
class MotorConfig:
    """Parameters for the drive motor (builders/motor_builder.py).

    Phase 6's own checkpoint is a scripted, open-loop sinusoidal rock, not
    RL-driven control -- so ANGLE is the default: theta(t) =
    amplitude_rad * sin(2*pi*frequency_hz*t), enforced directly on the
    joint via a rheonomic ChLinkMotorRotationAngle. Phase 10 adds
    MotorKind.TORQUE for direct RL torque control of the same physical
    joint (see environment/chrono_tank_dynamics.py) -- amplitude_rad and
    frequency_hz are simply unused when kind=TORQUE, not repurposed for
    anything, so validation below still requires them to be positive
    (their ANGLE-mode meaning) rather than adding a second set of
    kind-conditional fields.
    """

    kind: MotorKind = MotorKind.ANGLE
    amplitude_rad: float = 0.15  # ~8.6 deg -- a visually clear but gentle rock
    frequency_hz: float = 0.4

    def __post_init__(self) -> None:
        if self.amplitude_rad <= 0:
            raise ValueError("amplitude_rad must be positive.")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive.")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Parameters for the Phase 8 Gymnasium environment (environment/sloshing_env.py).

    Phase 8's own checkpoint is the RL interface itself, not RL-quality
    physics: MotorBuilder (builders/motor_builder.py) only implements
    MotorKind.ANGLE (a scripted, open-loop sine) today, and per its own
    docstring, RL torque/speed control of the REAL PyChrono joint is a
    Phase 10+ item. Meanwhile physics/backend_interface.py's own
    docstring commits the environment layer to depending on FluidBackend
    only -- never importing a concrete backend, and (by the same logic)
    never importing builders/tank_builder.py or builders/motor_builder.py
    either, since both pull in `pychrono.core` at module level the same
    way physics/pendulum_backend.py deliberately does not.

    So this config governs environment/tank_dynamics.py, a torque-driven
    single-DOF rotational surrogate for "how the tank itself responds to
    the RL agent" -- standing in for the real Chrono-stepped tank exactly
    the way PendulumBackend stands in for the eventual SPHBackend, until
    Phase 10+ swaps a real ChSystem-backed adapter in behind the same
    small interface.
    """

    # RL decision interval. Kept distinct from WorldConfig.time_step (the
    # physics integration step) so the environment can sub-step the
    # dynamics several times per agent action -- the same
    # decimation relationship Phase 10+'s real ChSystem stepping will
    # need anyway (an RL policy is not meant to run at 1 kHz).
    control_dt: float = 0.02  # s, 50 Hz agent decision rate

    # Physics integration step used to sub-step both TankDynamicsModel and
    # the FluidBackend within a single control_dt. Mirrors
    # WorldConfig.time_step's default exactly (1.0e-3 s) rather than
    # importing WorldConfig here, so this config stays self-contained and
    # constructible without also instantiating WorldConfig.
    physics_dt: float = 1.0e-3  # s

    # Episode horizon. 10 s at 50 Hz control = 500 steps per episode --
    # long enough to contain several sloshing periods (PendulumBackend's
    # natural frequency is on the order of 1-2 Hz for this project's
    # default TankConfig/FluidConfig) without making early PPO rollouts
    # (Phase 9) expensively long.
    max_episode_seconds: float = 10.0

    # Safety/termination bound on tank tilt. 45 degrees is well beyond
    # MotorConfig's own scripted Phase 6 demo amplitude (~8.6 deg) --
    # chosen as a generous envelope an RL policy can explore inside of
    # before the episode is judged to have failed, not a value tuned to
    # any specific reward outcome.
    max_tank_angle_rad: float = math.radians(45.0)

    # Action-space bound: peak torque (N*m) the agent may command about
    # the pivot axis. Sized so a policy CAN meaningfully accelerate the
    # tank against TankConfig's own default mass (~a few kg) within a few
    # control_dt steps, without being so large that a single action can
    # snap straight to max_tank_angle_rad -- reconsider if TankConfig's
    # mass changes substantially.
    max_motor_torque_nm: float = 2.0

    # Linear viscous damping at the pivot bearing (N*m*s/rad) -- an
    # explicit modeling simplification, not a measured value, flagged
    # here per the project's "no silent physics shortcuts" rule (see
    # TankConfig.contact_young_modulus's own docstring for the same
    # convention). Keeps the surrogate tank from spinning up without
    # bound under sustained one-directional torque.
    pivot_damping_coefficient: float = 0.02

    # Reward shaping weights. slosh_energy is the primary thing being
    # suppressed; the other two are small regularizers -- control_effort
    # discourages bang-bang/high-frequency torque as a degenerate way to
    # keep slosh_energy low, and angle_penalty discourages "solving" the
    # task by parking the tank at a large, otherwise-untested tilt.
    slosh_energy_weight: float = 1.0
    control_effort_weight: float = 1.0e-3
    angle_penalty_weight: float = 1.0e-2

    # Reset-time domain randomization: a random planar impulse (N*s) is
    # handed to FluidBackend.apply_disturbance so each episode starts
    # from a nonzero slosh state instead of dead-flat water every time --
    # otherwise a policy could trivially "succeed" by doing nothing.
    reset_disturbance_impulse_range: Tuple[float, float] = (0.0, 0.02)

    def __post_init__(self) -> None:
        if self.control_dt <= 0 or self.physics_dt <= 0:
            raise ValueError("control_dt and physics_dt must be positive.")
        if self.control_dt < self.physics_dt:
            raise ValueError("control_dt must be >= physics_dt.")
        if self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive.")
        if self.max_tank_angle_rad <= 0:
            raise ValueError("max_tank_angle_rad must be positive.")
        if self.max_motor_torque_nm <= 0:
            raise ValueError("max_motor_torque_nm must be positive.")
        if self.pivot_damping_coefficient < 0:
            raise ValueError("pivot_damping_coefficient must be non-negative.")
        low, high = self.reset_disturbance_impulse_range
        if low < 0 or high < low:
            raise ValueError(
                "reset_disturbance_impulse_range must satisfy 0 <= low <= high."
            )

    @property
    def substeps_per_control_step(self) -> int:
        """How many physics_dt sub-steps make up one control_dt agent step."""
        return max(1, round(self.control_dt / self.physics_dt))

    @property
    def max_episode_steps(self) -> int:
        """Episode horizon expressed in control_dt (agent) steps."""
        return max(1, round(self.max_episode_seconds / self.control_dt))


@dataclass(frozen=True)
class TrainingConfig:
    """Phase 9: PPO hyperparameters (training/train_ppo.py).

    Trains against `LiquidSloshingRL/SloshingSuppression-v0`
    (environment/sloshing_env.py, registered in environment/__init__.py).
    Defaults are standard PPO starting points for continuous-control
    tasks of this size (see Schulman et al. 2017 and the values
    stable-baselines3's own PPO docs recommend as reasonable defaults),
    not numbers tuned specifically to this task yet -- that tuning is
    Phase 9's own iterative work, not something to bake into a frozen
    default before a single training run has happened.
    """

    # Total agent steps across ALL parallel envs combined (SB3 convention:
    # this is what model.learn(total_timesteps=...) counts, not per-env
    # steps). 500k is a reasonable first-run budget for a 7-dim
    # observation, 1-dim action task -- expect to raise this once a
    # baseline run shows whether the policy has converged.
    total_timesteps: int = 500_000

    # Parallel environments (SB3 VecEnv). Each env instance builds its own
    # PendulumBackend + TankDynamicsModel (both cheap, pure-Python, no
    # pychrono), so running several in parallel processes is inexpensive
    # relative to how slow a single real-time PyChrono/Irrlicht instance
    # would be -- exactly the throughput reason
    # WorldConfig.contact_young_modulus's own docstring flags for
    # "RL-training throughput (Phase 10 onward)".
    n_envs: int = 8

    # PPO rollout length collected per env before each policy update.
    # n_steps * n_envs is the total rollout batch size PPO trains on.
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0

    seed: int = 0

    # SB3's VecNormalize running-mean/variance normalization for
    # observations and rewards -- addresses the -inf/+inf observation
    # bound and physically-unit-dependent reward scale flagged as a known
    # limitation when environment/sloshing_env.py was built (see
    # CONTINUE_HERE.md), without having to hand-pick finite observation
    # bounds that aren't grounded in anything.
    normalize_observations: bool = True
    normalize_rewards: bool = True

    # How often (in total timesteps) to write a model checkpoint via
    # SB3's CheckpointCallback -- frequent enough to survive an
    # interrupted run without losing much progress, infrequent enough not
    # to dominate training time with disk I/O.
    checkpoint_every_steps: int = 50_000

    # Where artifacts land. Both are relative to the project root (same
    # convention as the pre-existing empty logs/ and models/ folders);
    # created automatically if missing. tensorboard_log_dir="" disables
    # TensorBoard logging entirely -- useful in environments (like a fresh
    # sandbox) where the optional `tensorboard` package isn't installed,
    # since stable-baselines3 raises ImportError at learn()-time (not
    # construction-time) if a tensorboard_log path is set without it.
    tensorboard_log_dir: str = "logs/tensorboard"
    model_save_path: str = "models/ppo_sloshing"

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError(
                "batch_size must not exceed n_steps * n_envs (the rollout "
                "buffer size PPO trains on each update)."
            )
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1].")
        if self.clip_range <= 0:
            raise ValueError("clip_range must be positive.")
        if self.ent_coef < 0:
            raise ValueError("ent_coef must be non-negative.")
        if self.checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be positive.")
