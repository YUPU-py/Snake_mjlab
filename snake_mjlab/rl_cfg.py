"""RL configuration for Snake velocity task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def snake_ppo_runner_cfg(experiment_name: str = "snake_velocity"):
    """Create RL runner configuration for Snake velocity task."""
    # NOTE: Using 39-dim symmetric observations
    # Original 80-dim asymmetric config is commented out below
    cfg = RslRlOnPolicyRunnerCfg(
        obs_groups={"actor": ["policy"], "critic": ["policy"]},  # Symmetric 39-dim
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
            learning_rate=3.0e-4,
            desired_kl=0.02,
            gamma=0.99,
            lam=0.95,
            num_learning_epochs=5,
            num_mini_batches=4,
            max_grad_norm=1.0,
            schedule="adaptive",
            use_clipped_value_loss=True,
            clip_param=0.2,
        ),
        experiment_name=experiment_name,
        max_iterations=10_000,
    )
    return cfg

    # Original asymmetric 80-dim config:
    # cfg = RslRlOnPolicyRunnerCfg(
    #     obs_groups={"actor": ["actor"], "critic": ["actor", "critic"]},  # Asymmetric 80-dim
    #     ...
    # )


def snake_residual_ppo_runner_cfg():
    """Create RL runner configuration for Snake residual velocity task."""
    # NOTE: Using 39-dim symmetric observations (same as Isaac Lab)
    # Original 80-dim asymmetric config is commented out below
    cfg = RslRlOnPolicyRunnerCfg(
        obs_groups={"actor": ["policy"], "critic": ["policy"]},  # Symmetric 39-dim
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
            entropy_coef=0.01,  # Lower entropy for residual (more deterministic)
            value_loss_coef=0.5,
            learning_rate=1.0e-4,
            desired_kl=0.01,  # Lower KL for finer residual corrections
            gamma=0.99,
            lam=0.95,
            num_learning_epochs=5,
            num_mini_batches=4,
            max_grad_norm=1.0,
            schedule="fixed",
            use_clipped_value_loss=True,
            clip_param=0.2,
        ),
        experiment_name="snake_residual_velocity",
        max_iterations=10_000,
    )
    return cfg

    # Original asymmetric 80-dim config:
    # cfg = RslRlOnPolicyRunnerCfg(
    #     obs_groups={"actor": ["actor"], "critic": ["actor", "critic"]},  # Asymmetric 80-dim
    #     ...
    # )


class SnakeVelocityFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for SnakeVelocityFlatEnvCfg task.

    Matches the Isaac Lab SnakeVelocityFlatPPORunnerCfg parameters:
    - actor/critic hidden dims: [512, 256, 128]
    - Algorithm: PPO with adaptive KL schedule
    - 24 steps per env, 5 epochs, 4 mini-batches
    - lr=1e-3, gamma=0.99, lam=0.95
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "snake_velocity_flat_tracking"

    actor = RslRlModelCfg(
        hidden_dims=[512, 256, 128],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }
    )
    critic = RslRlModelCfg(
        hidden_dims=[512, 256, 128],
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.01,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
