"""Snake velocity environment configurations for mjlab."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
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
from snake_mjlab import configclass
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from snake_mjlab.mdp.actions import JointPositionActionCfg, ResidualSerpenoidJointPositionActionCfg
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
    residual_gait_phase_sin,
    residual_gait_phase_cos,
    nominal_serpenoid_joint_targets,
)
from snake_mjlab.mdp.rewards import (
    RawActionAccPenalty,
    RawActionRatePenalty,
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


@configclass
class SnakeVelocityFlatSceneCfg:
    """Scene configuration for the snake velocity-tracking task."""

    entities: dict = None  # type: ignore[assignment]
    terrain: object = None  # type: ignore[assignment]
    sensors: tuple = ()


@configclass
class SnakeVelocityFlatCommandsCfg:
    """Command specifications for the velocity-tracking MDP."""

    base_velocity = SnakeVirtualChassisCommandCfg(
        entity_name="robot",
        body_names=VIRTUAL_CHASSIS_BODY_NAMES,
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        planar_zero_threshold=0.0,
        ranges=SnakeVirtualChassisCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class SnakeVelocityFlatActionsCfg:
    """Action specifications for the velocity-tracking MDP."""

    joint_pos = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=("yaw.*",),
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-1.57, 1.57)},
    )


@configclass
class SnakeVelocityFlatObservationsCfg:
    """Observation specifications for the velocity-tracking MDP."""

    @configclass
    class PolicyCfg:
        base_ang_vel = ObservationTermCfg(
            func=base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.0125, n_max=0.0125),
        )
        projected_gravity = ObservationTermCfg(
            func=projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        velocity_commands = ObservationTermCfg(
            func=generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        last_actions = ObservationTermCfg(
            func=last_raw_actions,
            params={"action_name": "joint_pos"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg:
        base_ang_vel = ObservationTermCfg(
            func=base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.0125, n_max=0.0125),
        )
        projected_gravity = ObservationTermCfg(
            func=projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        velocity_commands = ObservationTermCfg(
            func=generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        last_actions = ObservationTermCfg(
            func=last_raw_actions,
            params={"action_name": "joint_pos"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class SnakeVelocityFlatEventCfg:
    """Event configuration for reset and randomization."""

    reset_robot = EventTermCfg(
        func=reset_snake_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_position_range": (0.00, 0.00),
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (0.2, 0.2),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class SnakeVelocityFlatRewardsCfg:
    """Reward terms for the velocity-tracking task."""

    track_lin_vel_xy_exp = RewardTermCfg(
        func=VirtualChassisTrackLinVelXYExp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": 0.4,
            "linear_coef": 0.5,
            "asset_cfg": virtual_chassis_body_cfg(),
        },
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=VirtualChassisTrackAngVelZExp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
            "asset_cfg": virtual_chassis_body_cfg(),
        },
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=velocity_mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    )
    joint_torques_l2 = RewardTermCfg(
        func=envs_mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": yaw_joint_cfg()},
    )
    joint_acc_l2 = RewardTermCfg(
        func=envs_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": yaw_joint_cfg()},
    )
    raw_action_rate = RewardTermCfg(
        func=RawActionRatePenalty,
        weight=-0.01,
        params={"action_term_name": "joint_pos"},
    )
    joint_amplitude = RewardTermCfg(
        func=joint_amplitude,
        weight=0.2,
        params={"asset_cfg": yaw_joint_cfg()},
    )
    phase_propagation = RewardTermCfg(
        func=phase_propagation,
        weight=0.4,
        params={"asset_cfg": yaw_joint_cfg()},
    )
    motion_coordination = RewardTermCfg(
        func=motion_coordination,
        weight=-0.5,
        params={"asset_cfg": yaw_joint_cfg()},
    )


@configclass
class SnakeVelocityFlatTerminationsCfg:
    """Termination conditions for the velocity-tracking task."""

    time_out = TerminationTermCfg(func=time_out)
    invalid_state = TerminationTermCfg(
        func=invalid_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_root_lin_vel": 2.0,
            "max_root_ang_vel": 5.0,
            "min_root_height": -0.2,
            "max_root_height": 0.5,
        },
    )


@configclass
class SnakeVelocityFlatCurriculumCfg:
    """Curriculum configuration for the velocity-tracking task."""
    pass


@configclass
class SnakeVelocityFlatEnvCfg(ManagerBasedRlEnvCfg):
    """Environment configuration for the snake velocity-tracking task."""

    scene: SnakeVelocityFlatSceneCfg = SnakeVelocityFlatSceneCfg()
    commands: SnakeVelocityFlatCommandsCfg = SnakeVelocityFlatCommandsCfg()
    observations: SnakeVelocityFlatObservationsCfg = SnakeVelocityFlatObservationsCfg()
    actions: SnakeVelocityFlatActionsCfg = SnakeVelocityFlatActionsCfg()
    events: SnakeVelocityFlatEventCfg = SnakeVelocityFlatEventCfg()
    rewards: SnakeVelocityFlatRewardsCfg = SnakeVelocityFlatRewardsCfg()
    terminations: SnakeVelocityFlatTerminationsCfg = SnakeVelocityFlatTerminationsCfg()
    curriculum: SnakeVelocityFlatCurriculumCfg = SnakeVelocityFlatCurriculumCfg()

    def __post_init__(self) -> None:
        # Sim parameters
        self.sim = SimulationCfg(
            nconmax=200,
            njmax=2000,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        )

        # Viewer configuration
        self.viewer = ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            body_name="base_link",
            distance=2.0,
            elevation=-30.0,
            azimuth=45.0,
        )

        # Physics material from terrain
        self.sim.physics_material = self.scene.terrain.physics_material

        # Remove terrain-bound terminations if present
        self.terminations.pop("out_of_terrain_bounds", None)
        self.terminations.pop("fell_over", None)

        # Remove legged-robot metrics
        self.metrics = {}


# ---------------------------------------------------------------------------
# Legacy function-based configs (kept for backward compatibility)
# ---------------------------------------------------------------------------


def snake_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Snake flat terrain velocity configuration for mjlab.

    .. deprecated::
        Use :class:`SnakeVelocityFlatEnvCfg` directly instead.
    """
    import warnings
    warnings.warn(
        "snake_flat_env_cfg() is deprecated. Use SnakeVelocityFlatEnvCfg() directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    cfg = SnakeVelocityFlatEnvCfg()
    _apply_scene_setup(cfg)
    _add_contact_sensor(cfg)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations.policy.enable_corruption = False
        cfg.observations.critic.enable_corruption = False

    return cfg


def _apply_scene_setup(cfg: ManagerBasedRlEnvCfg) -> None:
    """Apply scene setup: robot entity, plane terrain, sensor cleanup."""
    # Scene: robot + plane terrain
    cfg.scene.entities = {"robot": get_snake_robot_cfg()}

    # Switch to flat plane terrain (no generator)
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None

    # Remove all foot/contact sensors (snake has no feet)
    if cfg.scene.sensors is not None:
        sensor_names_to_remove = {
            "terrain_scan",
            "foot_height_scan",
            "feet_ground_contact",
            "nonfoot_ground_touch",
        }
        cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in sensor_names_to_remove)


def _add_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
    """Add contact sensor for body-ground contact detection."""
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


def snake_residual_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Snake residual velocity-tracking environment configuration for mjlab.

    This environment uses a residual RL approach where the policy learns to correct
    a fixed serpenoid gait pattern rather than directly controlling the joints.

    Key differences from snake_flat_env_cfg:
    - Uses ResidualSerpenoidJointPositionAction instead of direct joint position control
    - Includes gait phase observations (sin/cos) for temporal awareness
    - Includes nominal serpenoid joint targets as observation
    - Uses residual-specific reward terms (action rate/acc penalties)
    """

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

    # --- Actions: Residual Serpenoid Joint Position Control ---
    cfg.actions = {
        "joint_pos": ResidualSerpenoidJointPositionActionCfg(
            entity_name="robot",
            joint_names=("yaw.*",),
            scale=0.25,
            offset=0.0,
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
                lin_vel_x=(-0.3, 0.3),  # Bidirectional for residual
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.2, 0.2),
                heading=(0.0, 0.0),
            ),
        )
    }

    # --- Observations: Include gait phase and nominal targets ---
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
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
        # Residual-specific observations
        "gait_phase_sin": ObservationTermCfg(
            func=residual_gait_phase_sin,
            params={"action_name": "joint_pos"},
        ),
        "gait_phase_cos": ObservationTermCfg(
            func=residual_gait_phase_cos,
            params={"action_name": "joint_pos"},
        ),
        "nominal_joint_targets": ObservationTermCfg(
            func=nominal_serpenoid_joint_targets,
            params={"action_name": "joint_pos"},
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": yaw_joint_cfg()},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "last_actions": ObservationTermCfg(
            func=last_raw_actions,
            params={"action_name": "joint_pos"},
        ),
    }

    cfg.observations = {
        "policy": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
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

    # --- Rewards: Residual-specific weights ---
    cfg.rewards = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=VirtualChassisTrackLinVelXYExp,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "std": 0.4,
                "linear_coef": 0.5,
                "asset_cfg": virtual_chassis_body_cfg(),
            },
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=VirtualChassisTrackAngVelZExp,
            weight=0.5,
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
            weight=-0.001,
            params={"asset_cfg": yaw_joint_cfg()},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=envs_mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": yaw_joint_cfg()},
        ),
        # Residual-specific reward terms
        "raw_action_rate": RewardTermCfg(
            func=RawActionRatePenalty,
            weight=-0.01,
            params={"action_term_name": "joint_pos"},
        ),
        "raw_action_acc": RewardTermCfg(
            func=RawActionAccPenalty,
            weight=-0.0005,
            params={"action_term_name": "joint_pos"},
        ),
        "is_terminated": RewardTermCfg(
            func=is_terminated,
            weight=-10.0,
        ),
        "contact_penalty": RewardTermCfg(
            func=contact_penalty,
            weight=-10.0,
            params={
                "sensor_name": "contact_sensor",
                "threshold": 0.0,
            },
        ),
    }

    # Remove all foot/contact-related rewards and base velocity rewards.
    remove_reward_names = {
        "upright",
        "pose",
        "body_ang_vel",
        "angular_momentum",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "joint_amplitude",
        "phase_propagation",
        "motion_coordination",
        "dof_pos_limits",
        "lin_vel_z_l2",
    }
    for name in remove_reward_names:
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
                "max_root_ang_vel": 10.0,
                "min_root_height": -0.2,
                "max_root_height": 1.0,
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
                "joint_position_range": (0.0, 0.0),
                "pose_range": {
                    "x": (-0.2, 0.2),
                    "y": (0.2, 0.2),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        ),
    }

    # --- Curriculum: Clear all curriculum ---
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
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg
