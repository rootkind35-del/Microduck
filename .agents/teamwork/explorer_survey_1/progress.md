# Progress — Explorer Survey 1

Last visited: 2026-10-11T14:48:45Z

## Current Status
- Complete: Deep-dive survey of existing RL environment configurations (`microduck_velocity_env_cfg.py`, `microduck_velstand_env_cfg.py`, `microduck_standup_env_cfg.py`, `microduck_roller_slope_env_cfg.py`, `slope_terrain.py`, `tasks/__init__.py`, `backlash.py`).
- Complete: Comprehensive report written to `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/report.md`.
- Complete: 5-component handoff report written to `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/handoff.md`.
- Complete: DISPATCH.md updated with all incoming messages.
- Ready to message orchestrator.

## Key Findings Delivered
1. Base template: Inherit from `make_microduck_velocity_env_cfg` + incorporate `velstand` recovery layer.
2. Robot model: `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (70 collision geoms) to handle shin/knee stair collisions and fallen body collisions.
3. Solver settings: `nconmax = 200`, `iterations = 30`, `ls_iterations = 50`, `_soften_terrain_contacts` (`solref=[0.04, 1.0]`).
4. 61D observation invariant preserved strictly.
5. Calibrated stair curriculum: Two-tier architecture combining spatial terrain difficulty grid (`step_height_range=(0.025, 0.035)`) with temporal step schedules (`common_step_counter = iteration * 24`).
6. Self-recovery: Tilt-only gating (`tilt > 40°`), unfarmable potential formulation (`upright_progress`, `height_progress`), smoothness attempt tax relief (`FALLEN_SMOOTHNESS_SCALE = 0.1`), recovery economics (`fallen_tax`, `recovery_success`).
7. Task registration & export: `Mjlab-StairsClimb-MicroDuck` + backlash variant, `MicroduckOnPolicyRunner`, ONNX export compatibility.
