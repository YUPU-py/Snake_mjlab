"""Snake velocity environment configurations for mjlab."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from snake_mjlab.mdp.commands import SnakeVirtualChassisCommandCfg
from snake_mjlab.mdp.curriculums import command_velocity_curriculum
from snake_mjlab.mdp.events import reset_snake_state
from snake_mjlab.mdp.observations import (
    base_ang_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_raw_actions,
    projected_gravity,
)
from snake_mjlab.mdp.rewards import (
    VirtualChassisTrackAngVelZExp,
    VirtualChassisTrackLinVelXYExp,
    contact_penalty,
    joint_amplitude,
    motion_coordination,
    phase_propagation,
)
from snake_mjlab.mdp.terminations import invalid_state, is_terminated, time_out
from snake_mjlab.snake_14dof.snake_14dof_constants import get_snake_robot_cfg

# Yaw joint names for the 14-DOF snake robot.
YAW_JOINT_NAMES = [f"yaw{index}" for index in range(1, 8)]

# Virtual chassis body names (base_link + link1 through link14).
VIRTUAL_CHASSIS_BODY_NAMES = ("base_link",) + tuple(f"link{index}" for index in range(1, 15))


def yaw_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=YAW_JOINT_NAMES, preserve_order=True)


def virtual_chassis_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=list(VIRTUAL_CHASSIS_BODY_NAMES), preserve_order=True)


def snake_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Snake flat terrain velocity configuration for mjlab."""

    cfg = make_velocity_env_cfg()

    # --- Scene: robot + plane terrain ---
    cfg.scene.entities = {"robot": get_snake_robot_cfg()}

    # Switch to flat plane terrain (no generator).
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None

    # Remove all foot/contact sensors (snake has no feet).
    if cfg.scene.sensors is not None:
        sensor_names_to_remove = {
            "terrain_scan",
            "foot_height_scan",
            "feet_ground_contact",
            "nonfoot_ground_touch",
        }
        cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in sensor_names_to_remove)

    # --- Actions: 7 yaw joint position targets ---
    cfg.actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("yaw.*",),
            scale=0.25,
            use_default_offset=True,
            preserve_order=True,
            clip={".*": (-1.57, 1.57)},
        )
    }

    # --- Commands: Virtual chassis velocity tracking ---
    cfg.commands = {
        "base_velocity": SnakeVirtualChassisCommandCfg(
            entity_name="robot",
            body_names=VIRTUAL_CHASSIS_BODY_NAMES,
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            heading_control_stiffness=0.5,
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            planar_zero_threshold=0.0,
            ranges=SnakeVirtualChassisCommandCfg.Ranges(
                lin_vel_x=(-0.4, 0.4),  # 固定向前 0.1 m/s
                lin_vel_y=(-0.2, 0.2),  
                ang_vel_z=(0.0, 0.0),  
                heading=(0.0, 0.0),
            ),
        )
    }

    # --- Observations ---
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.0125, n_max=0.0125),
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        ),
        "velocity_commands": ObservationTermCfg(
            func=generated_commands,
            params={"command_name": "base_velocity"},
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "last_actions": ObservationTermCfg(
            func=last_raw_actions,
            params={"action_name": "joint_pos"},
        ),
    }

    critic_terms = dict(actor_terms)

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            nan_policy="sanitize",
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="sanitize",
        ),
    }

    # --- Add contact sensor for body-ground contact detection ---
    contact_sensor = ContactSensorCfg(
        name="contact_sensor",
        primary=ContactMatch(mode="body", pattern="(base_link|link[0-9]+)", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        history_length=1,
    )
    if cfg.scene.sensors is not None:
        cfg.scene.sensors = (*cfg.scene.sensors, contact_sensor)
    else:
        cfg.scene.sensors = (contact_sensor,)

    # --- Rewards: Snake-specific (matching Isaac Lab weights) ---
    cfg.rewards = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=VirtualChassisTrackLinVelXYExp,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "std": 0.25,
                "asset_cfg": virtual_chassis_body_cfg(),
            },
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=VirtualChassisTrackAngVelZExp,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "std": 0.25,
                "asset_cfg": virtual_chassis_body_cfg(),
            },
        ),
        "ang_vel_xy_l2": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
        ),
        "joint_torques_l2": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-1.0e-4,
            params={"asset_cfg": yaw_joint_cfg()},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=envs_mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": yaw_joint_cfg()},
        ),
        "raw_action_rate": RewardTermCfg(
            func=envs_mdp.action_rate_l2,
            weight=-0.01,
        ),
        "joint_amplitude": RewardTermCfg(
            func=joint_amplitude,
            weight=0.5,
            params={"asset_cfg": yaw_joint_cfg()},
        ),
        "phase_propagation": RewardTermCfg(
            func=phase_propagation,
            weight=0.4,
            params={"asset_cfg": yaw_joint_cfg()},
        ),
        "motion_coordination": RewardTermCfg(
            func=motion_coordination,
            weight=-0.5,
            params={"asset_cfg": yaw_joint_cfg()},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=envs_mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "is_terminated": RewardTermCfg(
            func=is_terminated,
            weight=-10.0,
        ),
        "contact_penalty": RewardTermCfg(
            func=contact_penalty,
            weight=-5.0,
            params={
                "sensor_name": "contact_sensor",
                "threshold": 0.0,
            },
        ),
    }

    # Remove all foot/contact-related rewards.
    foot_reward_names = {
        "upright",
        "pose",
        "body_ang_vel",
        "angular_momentum",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    }
    for name in foot_reward_names:
        cfg.rewards.pop(name, None)

    # --- Terminations ---
    cfg.terminations = {
        "time_out": TerminationTermCfg(
            func=time_out,
        ),
        "invalid_state": TerminationTermCfg(
            func=invalid_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_root_lin_vel": 5.0,
                "max_root_ang_vel": 5.0,
                "min_root_height": -0.2,
                "max_root_height": 0.5,
            },
        ),
    }

    # Remove terrain-bound terminations.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("fell_over", None)

    # --- Events ---
    cfg.events = {
        "reset_snake": EventTermCfg(
            func=reset_snake_state,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_position_range": (-0.05, 0.05),
                "pose_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "yaw": (-0.0, 0.0),
                },
                "velocity_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
            },
        ),
    }

    # --- Curriculum: 清空所有 curriculum（来自 make_velocity_env_cfg 的 terrain_levels 也需要清除）---
    cfg.curriculum = {}

    # --- Remove legged-robot metrics ---
    cfg.metrics = {}

    # --- Sim parameters ---
    cfg.sim = SimulationCfg(
        nconmax=200,
        njmax=2000,
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
        ),
    )

    # --- Viewer ---
    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        body_name="base_link",
        distance=2.0,
        elevation=-30.0,
        azimuth=45.0,
    )

    # --- Play mode overrides ---
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False
        # Disable push events in play mode.
        cfg.events.pop("push_robot", None)

    return cfg
