"""
Sanity check for scripts/export_policy_weights_to_json.py: reproduce the real
model's deterministic action, for several random observations, using ONLY
the exported JSON (plain numpy forward pass, the same math the JS port will
use) -- and diff against the real model.predict() output directly. If these
don't match to float precision, the JS port would be silently wrong, so this
must pass before trusting the export.

Run:
    python -m scripts.verify_exported_policy_weights
"""

from __future__ import annotations

import json

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def forward_from_json(weights: dict, raw_obs: np.ndarray) -> float:
    mean = np.array(weights["obs_mean"])
    var = np.array(weights["obs_var"])
    eps = weights["obs_epsilon"]
    clip_obs = weights["clip_obs"]
    x = np.clip((raw_obs - mean) / np.sqrt(var + eps), -clip_obs, clip_obs)

    for layer in weights["hidden_layers"]:
        W = np.array(layer["weight"])  # [out, in]
        b = np.array(layer["bias"])
        x = np.tanh(W @ x + b)

    out_layer = weights["output_layer"]
    W = np.array(out_layer["weight"])
    b = np.array(out_layer["bias"])
    action = W @ x + b
    # SB3's BasePolicy.predict() clips the raw mean to the action space
    # bounds ([-1, 1] here) as its final step for any Box action space --
    # missing this was the actual cause of the first verification failure
    # (max diff 0.039 across 200 random obs, entirely from cases where the
    # raw network output exceeded +-1 before clipping).
    action = np.clip(action, -1.0, 1.0)
    return float(action[0])


def main() -> None:
    model = PPO.load("models/ppo_sloshing.zip", device="cpu")

    import environment  # noqa: F401
    import gymnasium as gym

    throwaway = DummyVecEnv([lambda: gym.make("LiquidSloshingRL/SloshingSuppression-v0")])
    normalizer = VecNormalize.load("models/ppo_sloshing_vecnormalize.pkl", throwaway)
    normalizer.training = False

    with open("for_advisor/ppo_sloshing_policy_weights.json", encoding="utf-8") as f:
        weights = json.load(f)

    rng = np.random.default_rng(0)
    max_abs_diff = 0.0
    n_trials = 200
    for _ in range(n_trials):
        # Sample plausible raw observations across the real ranges seen in
        # this project's own env (tank_angle/omega small, fluid_com small,
        # slosh_energy non-negative) -- not just zeros, to actually exercise
        # the tanh nonlinearity away from the origin.
        raw_obs = np.array([
            rng.uniform(-0.5, 0.5),   # tank_angle
            rng.uniform(-2.0, 2.0),   # tank_angular_velocity
            rng.uniform(-0.1, 0.1),   # fluid_com_x
            rng.uniform(-0.1, 0.1),   # fluid_com_y
            rng.uniform(-1.0, 1.0),   # fluid_com_vx
            rng.uniform(-1.0, 1.0),   # fluid_com_vy
            rng.uniform(0.0, 0.5),    # slosh_energy
        ], dtype=np.float32)

        normalized = normalizer.normalize_obs(raw_obs)
        real_action, _ = model.predict(normalized, deterministic=True)
        real_action = float(np.asarray(real_action).reshape(-1)[0])

        json_action = forward_from_json(weights, raw_obs)
        diff = abs(real_action - json_action)
        max_abs_diff = max(max_abs_diff, diff)

    print(f"Ran {n_trials} random observations. Max |real - json| = {max_abs_diff:.8f}")
    if max_abs_diff > 1e-4:
        raise SystemExit("MISMATCH -- exported weights do not reproduce the real policy. Do not use.")
    print("PASS -- exported JSON weights reproduce the real trained policy's actions.")


if __name__ == "__main__":
    main()
