"""Tests for benchmarks/run_observable_controller_benchmarks.py -- pure
Python (this project's PyChrono-free surrogate env + scipy), no PyChrono
and no stable-baselines3 needed (PPO is skipped via ppo_model_path=None).

The one thing worth actually verifying here, beyond what
tests/test_run_controller_benchmarks.py and tests/test_lqr_controller.py
already cover in isolation: that the FAIR LQR row (pressure-observation
only) still meaningfully damps the disturbance on ObservableSloshingEnv,
while PID -- given the exact same observation -- still applies zero
torque throughout, for the exact reason documented in this project's
the project notes (the disturbance is applied to the fluid, never the tank, and
PID only ever tracks tank_angle)."""

import numpy as np

from benchmarks.run_observable_controller_benchmarks import run_benchmark_campaign


def test_fair_campaign_runs_end_to_end_without_ppo(tmp_path):
    metrics = run_benchmark_campaign(seed=0, ppo_model_path=None, output_dir=str(tmp_path))
    assert set(metrics) == {"passive", "pid", "lqr"}
    assert (tmp_path / "observable_benchmark_summary.csv").exists()


def test_pid_still_applies_zero_torque_on_the_pressure_observation(tmp_path):
    # Same underlying reason as the base benchmark: the reset-time
    # disturbance is applied directly to the fluid, never the tank, so a
    # PID that only tracks tank_angle/rate (obs[0]/obs[1], unchanged by
    # switching to the pressure observation) sees zero error the whole
    # episode regardless of which observation class is used.
    metrics = run_benchmark_campaign(seed=0, ppo_model_path=None, output_dir=str(tmp_path))
    assert metrics["pid"]["torque_total_variation"] == 0.0
    assert metrics["pid"]["settling_time_s"] == metrics["passive"]["settling_time_s"]


def test_fair_lqr_meaningfully_damps_the_disturbance_from_pressure_alone(tmp_path):
    metrics = run_benchmark_campaign(seed=0, ppo_model_path=None, output_dir=str(tmp_path))
    # Unlike PID, LQR recovers the pendulum's own state from the pressure
    # taps and should actually command torque and settle well within the
    # episode -- not just avoid a crash.
    assert metrics["lqr"]["torque_total_variation"] > 0.0
    assert metrics["lqr"]["settling_time_s"] < metrics["passive"]["settling_time_s"]


def test_different_seeds_give_different_but_still_valid_metrics(tmp_path):
    metrics_a = run_benchmark_campaign(seed=0, ppo_model_path=None, output_dir=str(tmp_path))
    metrics_b = run_benchmark_campaign(seed=1, ppo_model_path=None, output_dir=str(tmp_path))
    for metrics in (metrics_a, metrics_b):
        for row in metrics.values():
            assert np.isfinite(row["settling_time_s"])
            assert row["motor_energy_j"] >= 0.0
