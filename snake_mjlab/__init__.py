"""Snake velocity task for mjlab — integrates with mjlab's task registry."""

# Compatibility shim: mjlab >= 1.3 removed configclass.
# The no-op version works for dataclass-decorated term configs (ObservationTermCfg etc.)
# and plain class body-level instances.
def configclass(cls):
    return cls


# Compatibility shim: add resolve_matching_names_values if missing from mjlab.utils.string
try:
    from mjlab.utils.string import resolve_matching_names_values
except ImportError:
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
        make_snake_flat_env_cfg,
        snake_residual_flat_env_cfg,
    )
    from .rl_cfg import SnakeVelocityFlatPPORunnerCfg, snake_ppo_runner_cfg, snake_residual_ppo_runner_cfg

    # Re-export so 'from snake_mjlab import make_snake_flat_env_cfg' works
    import snake_mjlab as _pkg
    _pkg.make_snake_flat_env_cfg = make_snake_flat_env_cfg
    _pkg.SnakeVelocityFlatEnvCfg = make_snake_flat_env_cfg

    register_mjlab_task(
        task_id="Mjlab-Velocity-Flat-Snake-14DOF",
        env_cfg=make_snake_flat_env_cfg(play=False),
        play_env_cfg=make_snake_flat_env_cfg(play=True),
        rl_cfg=snake_ppo_runner_cfg(),
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
