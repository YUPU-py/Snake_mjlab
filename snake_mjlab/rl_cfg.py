"""RL configuration for Snake velocity task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def snake_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Snake velocity task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="snake_velocity",
        max_iterations=20_000,
    )
