"""Snake velocity task for mjlab — integrates with mjlab's task registry."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import snake_flat_env_cfg
from .rl_cfg import snake_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Snake-14DOF",
    env_cfg=snake_flat_env_cfg(play=False),
    play_env_cfg=snake_flat_env_cfg(play=True),
    rl_cfg=snake_ppo_runner_cfg(),
    runner_cls=None,
)
