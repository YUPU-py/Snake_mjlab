#!/usr/bin/env python3
"""Standalone script to export an RSL-RL checkpoint to TorchScript for sim2sim evaluation.

Usage:
    python export_policy_jit.py \
        --checkpoint logs/rsl_rl/snake_velocity/2026-05-23_00-04-20/model_1100.pt \
        --output logs/rsl_rl/snake_velocity/2026-05-23_00-04-20/exported/policy.pt

The exported policy.pt can then be used by sim2sim_mujoco.py and sim2sim_eval.py.
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

# RSL-RL imports
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization


def build_actor_from_checkpoint(
    ckpt_path: str,
    obs_dim: int = 30,
    action_dim: int = 7,
    hidden_dims: list[int] = [512, 256, 128],
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    device: str = "cuda:0",
):
    """Rebuild the actor MLPModel from a checkpoint and load its weights."""
    if distribution_cfg is None:
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }

    # Create a fake TensorDict with the right observation groups
    from tensordict import TensorDict
    dummy_obs = TensorDict(
        {"actor": torch.zeros(1, obs_dim)},
        batch_size=[1],
    )

    model = MLPModel(
        obs=dummy_obs,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=action_dim,
        hidden_dims=tuple(hidden_dims),
        activation="elu",
        obs_normalization=obs_normalization,
        distribution_cfg=distribution_cfg,
    ).to(device)

    # Load checkpoint weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = ckpt["actor_state_dict"]
    model.load_state_dict(actor_sd, strict=True)
    model.eval()

    return model


class ExportablePolicy(nn.Module):
    """TorchScript-exportable policy wrapping the RSL-RL MLPModel."""

    def __init__(self, mlp_model: MLPModel):
        super().__init__()
        # Deep-copy components needed for export (avoids traced module holding training state)
        self.obs_normalizer = copy.deepcopy(mlp_model.obs_normalizer)
        self.mlp = copy.deepcopy(mlp_model.mlp)
        # For GaussianDistribution with stochastic_output=False, deterministic_output is identity
        self.deterministic_output = copy.deepcopy(mlp_model.distribution.as_deterministic_output_module())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """No-op reset (MLP has no recurrent state)."""
        pass


def export_policy(
    checkpoint_path: str,
    output_path: str,
    obs_dim: int = 30,
    action_dim: int = 7,
    hidden_dims: list[int] = [512, 256, 128],
    obs_normalization: bool = False,
):
    """Load a checkpoint and export the actor as TorchScript."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Export] Loading checkpoint from: {checkpoint_path}")
    print(f"[Export] Using device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    iter_num = ckpt.get("iter", "unknown")
    print(f"[Export] Checkpoint from iteration: {iter_num}")

    # Rebuild and load actor
    mlp_model = build_actor_from_checkpoint(
        ckpt_path=checkpoint_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        obs_normalization=obs_normalization,
        device=device,
    )

    # Wrap for TorchScript export
    exportable = ExportablePolicy(mlp_model).to(device)
    exportable.eval()

    # Trace with dummy input
    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=device)
    print(f"[Export] Tracing policy with obs shape: {dummy_obs.shape}")
    traced = torch.jit.trace(exportable, dummy_obs)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced.save(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[Export] Saved TorchScript policy to: {output_path}")
    print(f"[Export] File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RSL-RL policy to TorchScript")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the TorchScript policy.pt file. "
        "Defaults to <checkpoint_dir>/exported/policy.pt",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=30,
        help="Observation dimension (default: 30)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimension (default: 7)",
    )
    args = parser.parse_args()

    if args.output is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        args.output = os.path.join(ckpt_dir, "exported", "policy.pt")

    export_policy(
        args.checkpoint,
        args.output,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
    )
