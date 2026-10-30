"""Tests for control/pid_controller.py -- pure-Python, no PyChrono needed."""

from control.pid_controller import SloshPIDController


def test_zero_error_produces_zero_torque_from_rest():
    controller = SloshPIDController()
    torque = controller.get_action(current_angle=0.0, target_angle=0.0, dt=0.02)
    assert torque == 0.0


def test_positive_angle_error_produces_restoring_negative_torque():
    controller = SloshPIDController()
    torque = controller.get_action(current_angle=0.1, target_angle=0.0, dt=0.02)
    assert torque < 0.0


def test_negative_angle_error_produces_restoring_positive_torque():
    controller = SloshPIDController()
    torque = controller.get_action(current_angle=-0.1, target_angle=0.0, dt=0.02)
    assert torque > 0.0


def test_output_never_exceeds_torque_limit():
    controller = SloshPIDController(kp=1000.0, ki=1000.0, kd=1000.0, torque_limit=2.0)
    torque = controller.get_action(current_angle=1.0, target_angle=0.0, dt=0.02)
    assert abs(torque) <= 2.0


def test_reset_clears_integral_and_derivative_history():
    controller = SloshPIDController()
    for _ in range(20):
        controller.get_action(current_angle=0.1, target_angle=0.0, dt=0.02)
    assert controller._integral != 0.0

    controller.reset()
    assert controller._integral == 0.0
    assert controller._prev_error is None


def test_regulates_a_simple_inertia_toward_zero_angle():
    """Closed-loop sanity check: PID + a plain 1-DOF inertia converges."""
    controller = SloshPIDController(kp=14.0, ki=1.5, kd=4.5, torque_limit=2.0)
    dt = 0.02
    angle = 0.15
    angular_velocity = 0.0
    inertia = 0.15

    for _ in range(500):
        torque = controller.get_action(current_angle=angle, target_angle=0.0, dt=dt)
        angular_velocity += (torque / inertia) * dt
        angle += angular_velocity * dt

    assert abs(angle) < 0.01


def test_conditional_antiwindup_stops_integrating_while_saturated():
    """A huge, sustained error that keeps the output pinned at the torque
    limit should not let the integral term grow without bound -- confirms
    the anti-windup gate actually gates, not just that clipping happens."""
    controller = SloshPIDController(kp=1.0, ki=50.0, kd=0.0, torque_limit=2.0)
    for _ in range(200):
        controller.get_action(current_angle=100.0, target_angle=0.0, dt=0.02)
    # Without anti-windup, ki * integral alone would be enormous (ki=50,
    # 200 steps * dt=0.02 * error=-100 => integral ~= -400, output ~=
    # -20000). With conditional integration gating while saturated, the
    # accumulated integral must stay small.
    assert abs(controller._integral) < 5.0
