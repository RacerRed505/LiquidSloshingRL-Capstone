"""
Export a trained PPO policy's actual actor-network weights + VecNormalize
observation stats to JSON, so the SAME real trained policy (not a hand-tuned
stand-in) can run a genuine forward pass client-side in the interactive
"3D Sloshing Lab" artifact's Live tab.

Why this exists: the Live tab previously let the user manually set torque via
a slider -- there was no controller at all, so it never demonstrated this
project's actual point (a trained policy recovering from a disturbance and
returning the tank to level). Faking that with a hand-tuned PD controller
would contradict this project's own consistent "real data, not fudged"
convention used everywhere else in its demos/reports. This script instead
exports the REAL policy so the demo can run genuine inference.

Only exports what deterministic inference actually needs: the actor
(policy_net + action_net) MLP weights, and VecNormalize's obs_rms
mean/var/epsilon/clip_obs. Does NOT export the value network (vf) or
anything reward-related -- irrelevant to control.

Run:
    python -m scripts.export_policy_weights_to_json \
        --model-path models/ppo_sloshing.zip \
        --vecnormalize-path models/ppo_sloshing_vecnormalize.pkl \
        --output for_advisor/ppo_sloshing_policy_weights.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _linear_layer_to_json(layer) -> dict:
    # PyTorch Linear stores weight as [out_features, in_features] -- keep
    # that shape explicit in the JSON so the JS side's matmul convention
    # (out[i] = sum_j W[i][j] * in[j] + b[i]) is unambiguous, not guessed.
    return {
        "weight": layer.weight.detach().cpu().numpy().tolist(),
        "bias": layer.bias.detach().cpu().numpy().tolist(),
    }


def export(model_path: str, vecnormalize_path: str, output_path: str) -> None:
    model = PPO.load(model_path, device="cpu")
    policy = model.policy

    # SB3's default ActorCriticPolicy with no policy_kwargs override (confirmed
    # by grepping training/train_ppo.py -- no net_arch/policy_kwargs present)
    # builds mlp_extractor.policy_net as Linear->Tanh->Linear->Tanh (2x64),
    # then a separate linear action_net for the mean. Verified directly here,
    # not assumed, by asserting the real module types below.
    policy_net = policy.mlp_extractor.policy_net
    action_net = policy.action_net

    layers = [m for m in policy_net if m.__class__.__name__ == "Linear"]
    activations = {m.__class__.__name__ for m in policy_net if m.__class__.__name__ != "Linear"}
    if activations != {"Tanh"}:
        raise RuntimeError(
            f"Expected only Tanh activations in policy_net, found {activations} -- "
            "this script's JS-side assumptions would silently be wrong. Stop and check "
            "the real architecture before exporting."
        )
    if action_net.__class__.__name__ != "Linear":
        raise RuntimeError(f"Expected a plain Linear action_net, found {action_net.__class__.__name__}.")

    hidden_layers = [_linear_layer_to_json(layer) for layer in layers]
    output_layer = _linear_layer_to_json(action_net)

    # VecNormalize's real running obs stats -- required to reproduce the
    # exact normalization the policy was trained/evaluated against
    # (obs_norm = clip((obs - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)).
    import environment  # noqa: F401 -- gym.register() side effect
    import gymnasium as gym

    throwaway = DummyVecEnv([lambda: gym.make("LiquidSloshingRL/SloshingSuppression-v0")])
    normalizer = VecNormalize.load(vecnormalize_path, throwaway)

    obs_rms = normalizer.obs_rms
    export_data = {
        "source_model": model_path,
        "source_vecnormalize": vecnormalize_path,
        "obs_dim": int(obs_rms.mean.shape[0]),
        "action_dim": int(action_net.out_features),
        "obs_mean": obs_rms.mean.tolist(),
        "obs_var": obs_rms.var.tolist(),
        "obs_epsilon": float(normalizer.epsilon),
        "clip_obs": float(normalizer.clip_obs),
        "hidden_layers": hidden_layers,
        "hidden_activation": "tanh",
        "output_layer": output_layer,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f)

    print(f"Exported {len(hidden_layers)} hidden layer(s) + 1 output layer to {output_path}")
    print(f"obs_dim={export_data['obs_dim']} action_dim={export_data['action_dim']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/ppo_sloshing.zip")
    parser.add_argument("--vecnormalize-path", default="models/ppo_sloshing_vecnormalize.pkl")
    parser.add_argument("--output", default="for_advisor/ppo_sloshing_policy_weights.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(args.model_path, args.vecnormalize_path, args.output)
