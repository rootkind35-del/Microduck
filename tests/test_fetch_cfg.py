from mjlab_microduck.tasks.microduck_fetch_env_cfg import (
    make_microduck_fetch_env_cfg,
    MicroduckFetchRlCfg,
)


def test_fetch_env_cfg_builds():
    """Verify that the Microduck-Fetch env cfg constructs properly."""
    cfg = make_microduck_fetch_env_cfg()
    assert "robot" in cfg.scene.entities
    assert "ball" in cfg.scene.entities
    assert cfg.sim.nconmax >= 50


def test_fetch_rewards_and_penalties():
    """Verify task-space rewards and self-negating penalty sign conventions."""
    cfg = make_microduck_fetch_env_cfg()
    r = cfg.rewards

    # Positive task-progress rewards
    assert "ball_approach_potential" in r
    assert r["ball_approach_potential"].weight > 0
    assert "ball_mouth_proximity" in r
    assert r["ball_mouth_proximity"].weight > 0
    assert "ball_heading_alignment" in r
    assert r["ball_heading_alignment"].weight > 0

    # Upright stability
    assert "upright" in r
    assert r["upright"].weight > 0

    # Negative regularizers and penalties
    assert "approach_overspeed_penalty" in r
    assert r["approach_overspeed_penalty"].weight < 0
    assert "action_rate_l2" in r
    assert r["action_rate_l2"].weight < 0
    assert "body_ang_vel" in r
    assert r["body_ang_vel"].weight < 0
    assert "self_collisions" in r
    assert r["self_collisions"].weight < 0


def test_fetch_events_and_observations():
    """Verify ball reset events and asymmetric critic observations."""
    cfg = make_microduck_fetch_env_cfg()
    assert "reset_ball" in cfg.events
    assert "ball_pos_rel" in cfg.observations["critic"].terms
    assert "ball_vel_rel" in cfg.observations["critic"].terms
    assert "twist" in cfg.commands

