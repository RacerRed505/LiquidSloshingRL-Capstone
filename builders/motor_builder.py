"""
Phase 6b -- Drive Motor Builder. Phase 10 -- RL torque-driven motor.

Attaches a ChLinkMotorRotation between the pivot support
(builders/pivot_builder.py) and the tank body (builders/tank_builder.py).
Two modes are implemented:

- MotorKind.ANGLE (Phase 6): a rheonomic ChLinkMotorRotationAngle
  prescribing an open-loop sinusoidal rock per MotorConfig -- Phase 6's
  own checkpoint ("the tank visibly tilts back and forth"), not RL
  control.
- MotorKind.TORQUE (Phase 10): a ChLinkMotorRotationTorque prescribing a
  commanded torque that changes every RL control step -- the real
  ChSystem-backed counterpart to environment/tank_dynamics.py's analytic
  surrogate, used by environment/chrono_tank_dynamics.py. The tank's
  resulting angle/velocity emerge from the actual rigid-body solve
  (mass, inertia, damping-free pivot -- same "no artificial restoring
  term" stance environment/tank_dynamics.py's docstring already takes)
  rather than being prescribed directly.

MotorKind.SPEED remains a NotImplementedError placeholder.

ChLinkMotorRotationAngle/Torque always actuate about the joint frame's
own local Z axis (Chrono's convention, not a choice made here -- see
ChLinkMotorRotation's own reference docs: "Motors of this type assume
that the spindle is directed along Z direction of the master frame").
That's exactly why builders/pivot_builder.py returns a joint_frame() whose
Z axis is already pre-rotated onto PivotConfig.axis via
utilities/quaternion_utils.py, rather than this file doing any axis math
of its own.

Motion profile note: both motion functions below are written as small
Python subclasses of chrono.ChFunction (GetVal/Clone) -- Chrono's own
documented supported pattern ("create an ad-hoc motion function by
python-side inheritance from ChFunction") -- rather than via
chrono.ChFunctionSine/ChFunctionConst, partly to sidestep the
constructor-argument-drift issue already hit once (see git history), and
for TORQUE mode because the value needs to be externally mutable every
RL control step, which a built-in ChFunction can't do without this same
technique anyway.
"""

import logging
import math
from typing import Tuple

import pychrono.core as chrono

from builders.pivot_builder import PivotBuilder
from config.simulation_config import MotorConfig, MotorKind, PivotConfig

logger = logging.getLogger(__name__)


class _SineAngleFunction(chrono.ChFunction):
    """theta(t) = amplitude_rad * sin(2*pi*frequency_hz*t), as a ChFunction.

    See this module's own docstring for why this is written by hand
    rather than via chrono.ChFunctionSine.
    """

    def __init__(self, amplitude_rad: float, frequency_hz: float) -> None:
        super().__init__()
        self._amplitude = amplitude_rad
        self._frequency_hz = frequency_hz
        self._omega = 2.0 * math.pi * frequency_hz

    def GetVal(self, x: float) -> float:
        return self._amplitude * math.sin(self._omega * x)

    def Clone(self) -> "chrono.ChFunction":
        return _SineAngleFunction(self._amplitude, self._frequency_hz)


class _LiveSineAngleFunction(chrono.ChFunction):
    """Like _SineAngleFunction, but amplitude_rad/frequency_hz are live-
    mutable Python attributes instead of fixed at construction.

    For main.py's --fluid-viz-interactive demo only (not used by
    training/evaluation/chrono_tank_dynamics.py, which use MotorKind.TORQUE
    instead): a person can retune the motor's max inclination and swing
    speed while the simulation runs. Reuses the exact same hand-written-
    ChFunction-subclass technique _SineAngleFunction and
    _ConstantTorqueFunction already use, rather than a new mechanism --
    same reasoning as _ConstantTorqueFunction's own docstring: a plain
    ChFunctionSine can't be retuned after construction, and _SineAngleFunction
    itself is deliberately left set-once (used verbatim, unmodified,
    everywhere else in this project).

    Note: GetVal uses the absolute time argument `x` directly (same as
    _SineAngleFunction), not an integrated phase -- so changing
    amplitude_rad or frequency_hz mid-simulation will produce a small
    instantaneous jump in the commanded angle at that instant, since the
    new formula doesn't generally equal the old one at the moment of the
    change. This is an accepted, visible quirk of the simplest correct
    live-tuning approach, not a bug.
    """

    def __init__(self, amplitude_rad: float, frequency_hz: float) -> None:
        super().__init__()
        self.amplitude_rad = amplitude_rad
        self.frequency_hz = frequency_hz

    def GetVal(self, x: float) -> float:
        return self.amplitude_rad * math.sin(2.0 * math.pi * self.frequency_hz * x)

    def Clone(self) -> "chrono.ChFunction":
        return _LiveSineAngleFunction(self.amplitude_rad, self.frequency_hz)


class _ConstantTorqueFunction(chrono.ChFunction):
    """A ChFunction returning a constant value that Python can mutate live.

    Used for MotorKind.TORQUE: ChLinkMotorRotationTorque.SetTorqueFunction
    still wants a function of time, not a settable scalar, but an RL
    agent's commanded torque changes every control step, not on a fixed
    schedule. GetVal ignores its `x` (time) argument entirely and just
    returns whatever `self.value` currently holds -- MotorBuilder.
    set_commanded_torque() is the intended way to update it each step,
    rather than callers reaching into `.value` directly.
    """

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = value

    def GetVal(self, x: float) -> float:
        return self.value

    def Clone(self) -> "chrono.ChFunction":
        clone = _ConstantTorqueFunction(self.value)
        return clone


class MotorBuilder:
    """Builds the pivot/motor joint connecting the tank to its support."""

    def __init__(self, pivot_config: PivotConfig, motor_config: MotorConfig) -> None:
        if motor_config.kind not in (MotorKind.ANGLE, MotorKind.TORQUE):
            raise NotImplementedError(
                f"MotorKind.{motor_config.kind.value} is not wired up yet -- "
                "only MotorKind.ANGLE and MotorKind.TORQUE are implemented."
            )
        self._pivot_config = pivot_config
        self._motor_config = motor_config

    def build(
        self,
        system: chrono.ChSystem,
        pivot_body: chrono.ChBody,
        tank_body: chrono.ChBody,
    ) -> "chrono.ChLinkMotorRotation":
        """Create the motor joint, wire in its motion function, add it to `system`.

        Body order matches Chrono's own ChLinkMotorRotation*.Initialize
        convention (rotor body first, stator body second, both examples in
        the official PyChrono docs put the MOVING part first): tank_body
        is the rotor (it moves), pivot_body is the stator (fixed support).
        """
        joint_frame = PivotBuilder(self._pivot_config).joint_frame()

        if self._motor_config.kind is MotorKind.ANGLE:
            motor = chrono.ChLinkMotorRotationAngle()
            motor.Initialize(tank_body, pivot_body, joint_frame)
            motion_function = _SineAngleFunction(
                self._motor_config.amplitude_rad, self._motor_config.frequency_hz
            )
            motor.SetAngleFunction(motion_function)
        else:  # MotorKind.TORQUE
            motor = chrono.ChLinkMotorRotationTorque()
            motor.Initialize(tank_body, pivot_body, joint_frame)
            motion_function = _ConstantTorqueFunction(0.0)
            motor.SetTorqueFunction(motion_function)

        # Keep the Python-side function object alive: pybind11 directors
        # can end up holding only a weak/borrowed reference on the C++
        # side, and Python's own GC would otherwise be free to collect
        # `motion_function` as soon as this method returns, silently
        # breaking the motor profile. Stashed on the returned motor object
        # itself so no caller has to remember to do this separately.
        motor._motion_function_keepalive = motion_function

        # ChSystem.Add() accepts any ChPhysicsItem (bodies, links, ...) --
        # used here rather than the more link-specific AddLink() so this
        # matches the same call tank_builder.py/pivot_builder.py already
        # use for bodies, and doesn't depend on AddLink existing under
        # that exact name in every PyChrono release.
        system.Add(motor)
        logger.info(
            "Motor built: kind=%s amplitude_rad=%.4f frequency_hz=%.4f",
            self._motor_config.kind.value,
            self._motor_config.amplitude_rad,
            self._motor_config.frequency_hz,
        )
        return motor

    @staticmethod
    def set_commanded_torque(motor: "chrono.ChLinkMotorRotationTorque", torque: float) -> None:
        """Update a MotorKind.TORQUE motor's commanded torque in place.

        Only valid for a motor built with MotorKind.TORQUE -- ANGLE-mode
        motors don't carry a `_motion_function_keepalive` with a settable
        `.value` in the sense this expects. Intended to be called once per
        physics sub-step, right before `system.DoStepDynamics(...)`, by
        environment/chrono_tank_dynamics.py.
        """
        motor._motion_function_keepalive.value = torque

    @staticmethod
    def make_live_angle_function(
        motor: "chrono.ChLinkMotorRotationAngle", amplitude_rad: float, frequency_hz: float
    ) -> _LiveSineAngleFunction:
        """Swap a MotorKind.ANGLE motor's fixed _SineAngleFunction for a
        live-mutable one, for main.py's --fluid-viz-interactive demo only.

        Only valid for a motor built with MotorKind.ANGLE, and intended to
        be called exactly ONCE, right after build() -- same "set once, then
        only mutate the function object's own fields" pattern
        set_commanded_torque() already establishes for MotorKind.TORQUE,
        not a pattern of repeatedly re-calling SetAngleFunction mid-run
        (which this project has never exercised and does not assume is
        safe on an already-stepping joint). Returns the live function so
        the caller can adjust its `.amplitude_rad`/`.frequency_hz`
        directly thereafter.
        """
        live_function = _LiveSineAngleFunction(amplitude_rad, frequency_hz)
        motor.SetAngleFunction(live_function)
        motor._motion_function_keepalive = live_function
        return live_function

    @staticmethod
    def read_tank_motion(motor: "chrono.ChLinkMotorRotation") -> Tuple[float, float]:
        """Read (tank_angle, tank_angular_velocity) straight off a built motor.

        Works for both ANGLE and TORQUE motors -- GetMotorAngle/
        GetMotorAngleDt report the joint's actual kinematic state either
        way, whether that state was prescribed (ANGLE) or emerged from the
        rigid-body solve under a commanded torque (TORQUE). This is the
        entire bridge between the real PyChrono motor and
        physics/pendulum_backend.py's `tank_motion` argument in main.py --
        PendulumBackend itself has no PyChrono import at all (see its own
        docstring), so this one function is what couples the two systems
        every simulation step.
        """
        return motor.GetMotorAngle(), motor.GetMotorAngleDt()
