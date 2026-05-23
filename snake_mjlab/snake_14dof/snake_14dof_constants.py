"""Snake 14-DOF constants for mjlab."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


##
# MJCF and assets.
##

_HERE = Path(__file__).parent

SNAKE_XML: Path = _HERE / "xmls" / "14DOF-DW.xml"
assert SNAKE_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    # Copy mesh assets from the xmls directory
    mesh_dir = SNAKE_XML.parent / "meshes"
    if mesh_dir.exists():
        for mesh_file in mesh_dir.rglob("*"):
            if mesh_file.is_file():
                rel_path = mesh_file.relative_to(SNAKE_XML.parent)
                assets[str(rel_path)] = mesh_file.read_bytes()
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SNAKE_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

# PD gains matching the Isaac Lab implicit actuator config.
STIFFNESS = 30.0
DAMPING = 0.6
EFFORT_LIMIT = 30.0
ARMATURE = 0.028

# Only control the 7 yaw joints, not pitch (pitch joints are passive).
SNAKE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("yaw.*",),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.1),
    joint_pos={
        "yaw.*": 0.0,
        "pitch.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# The snake uses two geom groups: group=1 for visual, group=3 for collision.
# All geoms with contype/conaffinity=1 collide with terrain.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*",),
    condim=3,
    priority=1,
    friction=(0.8, 0.1, 0.1),
)

##
# Final config.
##

SNAKE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SNAKE_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)


def get_snake_robot_cfg() -> EntityCfg:
    """Get a fresh Snake robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=SNAKE_ARTICULATION,
    )


# Action scale: 0.25 * effort / stiffness
SNAKE_ACTION_SCALE: dict[str, float] = {}
for _a in SNAKE_ARTICULATION.actuators:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    _e = _a.effort_limit
    _s = _a.stiffness
    assert _e is not None
    for _n in _a.target_names_expr:
        SNAKE_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_snake_robot_cfg())
    viewer.launch(robot.spec.compile())
