"""Microduck Ball Fetch RL Task Configuration.

Task description:
    The robot starts standing (HOME pose + noise). A ball is tossed/spawned in
    an arc in front of the robot. The goal is to detect the ball, navigate towards
    it, approach with deceleration, lower the beak to pick up the ball, and
    maintain upright stability.

Observation:
    Unified 61D actor observation (48 proprioceptive + 13 command slots),
    ensuring hot-swappable deployment with walking and standing policies.
    The 3D twist command slot carries the relative ball vector [x_rel, y_rel, dist]
    derived from head vision or target tracker.
    Asymmetric critic has access to privileged ground-truth ball relative
    position and velocity.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_STANDUP_ROBOT_CFG,
    MICRODUCK_BALL_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

EPISODE_LENGTH_S = 6.0
BALL_RADIUS = 0.035
STAND_Z = 0.115


def make_microduck_fetch_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the ManagerBasedRlEnvCfg for the Microduck Fetch task.

    Inherits the full domain randomization, observation noise, actuator latency,
    and 61D unified observation architecture from the velocity tracking base.
    """
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # 1. Ground-contact robot model + ball entity
    cfg.scene.entities["robot"] = MICRODUCK_STANDUP_ROBOT_CFG
    cfg.scene.entities["ball"] = MICRODUCK_BALL_CFG
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 50

    # 2. Clean walking/locomotion-velocity specific rewards
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # 3. Task Rewards & Penalties
    # (a) Potential-based approach reward towards ball
    cfg.rewards["ball_approach_potential"] = RewardTermCfg(
        func=microduck_mdp.ball_approach_potential,
        weight=5.0,
        params={"asset_name": "ball", "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"])},
    )

    # (b) Proximity reward when mouth is near the ball
    cfg.rewards["ball_mouth_proximity"] = RewardTermCfg(
        func=microduck_mdp.ball_mouth_proximity,
        weight=2.5,
        params={"asset_name": "ball", "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]), "std": 0.06},
    )

    # (c) Heading alignment: guide robot to face the ball
    cfg.rewards["ball_heading_alignment"] = RewardTermCfg(
        func=microduck_mdp.ball_heading_alignment,
        weight=1.5,
        params={"asset_name": "ball"},
    )

    # (d) Anti-ramming / anti-crash overspeed penalty near ball
    cfg.rewards["approach_overspeed_penalty"] = RewardTermCfg(
        func=microduck_mdp.approach_overspeed_penalty,
        weight=-2.0,
        params={"asset_name": "ball", "threshold_dist": 0.35, "max_speed": 0.20},
    )

    # 4. Ball reset event (places ball in front of robot in an arc)
    cfg.events["reset_ball"] = EventTermCfg(
        func=microduck_mdp.reset_ball_fetch_target,
        mode="reset",
        params={
            "min_dist": 0.35,
            "max_dist": 1.2,
            "max_angle": 0.6,
            "ball_radius": BALL_RADIUS,
            "asset_name": "ball",
        },
    )

    # 5. Commands: route relative ball vector [rel_x, rel_y, dist] into 3D twist slot
    cfg.commands["twist"] = microduck_mdp.FetchTargetCommandCfg()

    # Drop velocity standing_envs curriculum since commands are dynamic ball targets
    cfg.curriculum.pop("standing_envs", None)

    # 6. Drop terrain scan from critic if present (flat floor)
    if "foot_height" in cfg.observations["critic"].terms:
        del cfg.observations["critic"].terms["foot_height"]

    # 7. Privileged Critic Observations for Ball State
    cfg.observations["critic"].terms["ball_pos_rel"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base,
        scale=1.0,
        params={"asset_name": "ball"},
    )
    cfg.observations["critic"].terms["ball_vel_rel"] = ObservationTermCfg(
        func=microduck_mdp.ball_vel_in_base,
        scale=1.0,
        params={"asset_name": "ball"},
    )

    # 8. NaN Policy: sanitize transient contact solver spikes instead of crashing
    cfg.observations["critic"].nan_policy = "sanitize"
    cfg.observations["actor"].nan_policy = "sanitize"

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckFetchRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_fetch",
    run_name="microduck_fetch",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1_000,
)

