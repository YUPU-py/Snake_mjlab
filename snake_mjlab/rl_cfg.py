"""RL configuration for Snake velocity task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def snake_ppo_runner_cfg():
    """Create RL runner configuration for Snake velocity task."""
    cfg = RslRlOnPolicyRunnerCfg(
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
            entropy_coef=0.05,
            value_loss_coef=0.5,
            learning_rate=3.0e-4,    # 从 1e-3 降到 3e-4
            desired_kl=0.02,          # 从 0.01 提升到 0.02
            gamma=0.99,
            lam=0.95,
            num_learning_epochs=5,
            num_mini_batches=4,
            max_grad_norm=1.0,
            schedule="adaptive",
            use_clipped_value_loss=True,
            clip_param=0.2,
        ),
        experiment_name="snake_velocity",
        max_iterations=10_000,
    )
    return cfg
