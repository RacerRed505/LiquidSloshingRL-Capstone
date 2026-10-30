"""PID baseline controller for tank-angle regulation.

A classical control baseline to compare the trained PPO policy against
on the exact same environment/sloshing_env.py::SloshingSuppressionEnv --
see benchmarks/run_controller_benchmarks.py for that comparison. Tracks
tank_angle -> 0 (drives the tank level), which is a reasonable classical
proxy for "suppress sloshing" since PendulumBackend's own coupling is
driven by tank_angular_velocity (see physics/pendulum_backend.py's
docstring) -- damping the tank's own motion is the one lever a
tank-angle-only PID actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class SloshPIDController:
    """Standard PID on tank_angle -> commanded torque, with anti-windup.

    Anti-windup via conditional integration: the integral term only
    accumulates while the unclipped output is within the actuator's
    torque_limit -- the standard fix for the "integrator keeps winding up
    while the actuator is already saturated" failure mode, simpler than
    back-calculation and sufficient for this single-input-single-output
    loop.
    """

    def __init__(
        self,
        kp: float = 14.0,
        ki: float = 1.5,
        kd: float = 4.5,
        torque_limit: float = 2.0,
        derivative_filter_tau: float = 0.05,
    ) -> None:
        if torque_limit <= 0:
            raise ValueError("torque_limit must be positive.")
        if derivative_filter_tau < 0:
            raise ValueError("derivative_filter_tau must be non-negative.")

        self.gains = PIDGains(kp=kp, ki=ki, kd=kd)
        self._torque_limit = torque_limit
        self._tau = derivative_filter_tau

        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._filtered_derivative = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0

    def get_action(self, current_angle: float, target_angle: float, dt: float) -> float:
        """Returns a clipped torque command (N*m) driving current_angle -> target_angle."""
        error = target_angle - current_angle

        if self._prev_error is None:
            raw_derivative = 0.0
        else:
            raw_derivative = (error - self._prev_error) / dt
        self._prev_error = error

        # First-order low-pass filter on the derivative term -- a real PID
        # never differentiates raw (noisy) error directly; this project's
        # analytic observation is noise-free, but a controller meant to
        # generalize to real sensor data (this project's own "God-Mode
        # Observability" critique, see the project notes) should not assume that.
        if self._tau > 0.0:
            alpha = dt / (self._tau + dt)
            self._filtered_derivative += alpha * (raw_derivative - self._filtered_derivative)
        else:
            self._filtered_derivative = raw_derivative

        unclipped = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * self._filtered_derivative
        )

        clipped = max(-self._torque_limit, min(self._torque_limit, unclipped))

        # Conditional-integration anti-windup: only integrate while the
        # unclipped command agrees in sign with (or is smaller than) the
        # clipped one -- i.e. the actuator isn't already saturated against
        # this error's own direction.
        if clipped == unclipped or (error * clipped) <= 0.0 or abs(unclipped) <= self._torque_limit:
            self._integral += error * dt

        return clipped


if __name__ == "__main__":
    import math

    print("=" * 60)
    print("PID CONTROLLER SELF-TEST")
    print("=" * 60)

    controller = SloshPIDController(kp=14.0, ki=1.5, kd=4.5, torque_limit=2.0)
    dt = 0.02
    angle = 0.12
    angular_velocity = 0.0
    inertia = 0.15  # representative tank inertia, kg*m^2 -- see control/lqr_controller.py

    print(f"\nRegulating an initial {math.degrees(angle):.1f} deg tilt to zero:")
    for step in range(50):
        torque = controller.get_action(current_angle=angle, target_angle=0.0, dt=dt)
        angular_accel = torque / inertia
        angular_velocity += angular_accel * dt
        angle += angular_velocity * dt
        if step % 10 == 0:
            print(
                f"  t={step * dt:4.2f}s  angle={math.degrees(angle):+6.2f} deg  "
                f"torque={torque:+5.2f} N*m"
            )

    print(f"\nFinal angle: {math.degrees(angle):+.3f} deg (target: 0.0 deg)")
    print("=" * 60)
