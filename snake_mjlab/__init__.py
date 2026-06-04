"""Snake velocity task for mjlab — integrates with mjlab's task registry."""

# Compatibility shim: mjlab >= 1.3 removed configclass; replace with pass-through
# so snake_mjlab code (written for isaaclab-style configclass) stays compatible.
# Old configclass was essentially a marker; dataclass without kw_only works for this code.
def configclass(cls):
    return cls


# Compatibility shim: add resolve_matching_names_values if missing from mjlab.utils.string

try:
    from mjlab.utils.string import resolve_matching_names_values
except ImportError:
    import re
    from mjlab.utils.string import filter_exp
    
    def resolve_matching_names_values(pattern_map: dict, names: tuple) -> tuple:
        """Compatibility shim for mjlab versions that removed this function.

        Takes a dict like {"yaw.*": (min, max)} and resolves it against joint names.
        Returns (index_list, None, value_list) matching the old API.
        """
        index_list = []
        value_list = []
        for pattern, value in pattern_map.items():
            matched_names = filter_exp([pattern], names)
            for name in matched_names:
                idx = names.index(name)
                index_list.append(idx)
                value_list.append(value)
        return (index_list, None, value_list)


try:
    from mjlab.tasks.registry import register_mjlab_task

    from .env_cfgs import (
        SnakeVelocityFlatEnvCfg,
        snake_flat_env_cfg,
        snake_residual_flat_env_cfg,
    )
    from .rl_cfg import SnakeVelocityFlatPPORunnerCfg, snake_ppo_runner_cfg, snake_residual_ppo_runner_cfg

    def _make_play_env_cfg(play: bool = False) -> SnakeVelocityFlatEnvCfg:
        """Create a play-mode environment config from the new class-based config."""
        cfg = SnakeVelocityFlatEnvCfg()
        cfg.scene.entities = {"robot": snake_flat_env_cfg(play=True).scene.entities["robot"]}
        if cfg.scene.terrain is not None:
            cfg.scene.terrain.terrain_type = "plane"
            cfg.scene.terrain.terrain_generator = None
        if cfg.scene.sensors is not None:
            sensor_names_to_remove = {
                "terrain_scan",
                "foot_height_scan",
                "feet_ground_contact",
                "nonfoot_ground_touch",
            }
            cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in sensor_names_to_remove)
        if play:
            cfg.episode_length_s = int(1e9)
            cfg.observations.policy.enable_corruption = False
            cfg.observations.critic.enable_corruption = False
        return cfg

    register_mjlab_task(
        task_id="Mjlab-Velocity-Flat-Snake-14DOF",
        env_cfg=SnakeVelocityFlatEnvCfg(),
        play_env_cfg=_make_play_env_cfg(play=True),
        rl_cfg=SnakeVelocityFlatPPORunnerCfg(),
        runner_cls=None,
    )

    register_mjlab_task(
        task_id="Mjlab-Velocity-Residual-Flat-Snake-14DOF",
        env_cfg=snake_residual_flat_env_cfg(play=False),
        play_env_cfg=snake_residual_flat_env_cfg(play=True),
        rl_cfg=snake_residual_ppo_runner_cfg(),
        runner_cls=None,
    )
except Exception as e:
    import sys
    print(f"[snake_mjlab] Task registration skipped (mjlab version mismatch): {e}", file=sys.stderr)
