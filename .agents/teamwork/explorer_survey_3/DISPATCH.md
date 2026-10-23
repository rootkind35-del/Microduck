## 2026-10-11T14:37:24Z
You are Explorer Survey 3 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/AGENTS.md`.

Your objective is to investigate the standalone test simulation scene and runner scripts.
Specifically:
1. Examine existing MJCF scenes in `src/mjlab_microduck/robot/microduck/`: `scene_*.xml`, including terrain geometry, visual assets, lighting, floor contacts, friction, and collision pairs.
2. Design `src/mjlab_microduck/robot/microduck/scene_stairs.xml` with consecutive stair flights (2.5cm, 3.0cm, and 3.5cm step risers) and landing platforms. Detail geom dimensions, positions, materials, and how it loads the robot model (`robot_groundcontact.xml` / `robot_walk.xml`).
3. Examine `scripts/infer_policy.py` and other inference scripts in `scripts/`: how CPU MuJoCo rehearsal works, how BAM M6 actuator dynamics are implemented, keyboard controls, command updates, step loop at 50 Hz.
4. Design `scripts/infer_stairs.py` to support interactive rehearsal on stairs, real-time keyboard commands, spawn presets on stairs (bottom, mid-flight 2.5cm, mid-flight 3.0cm, mid-flight 3.5cm, top landing), and balance telemetry.
5. Check existing tests in `tests/`: what invariant tests exist, how configs are tested, and what tests are needed for `microduck_stairs_env_cfg.py`, `scene_stairs.xml`, and `scripts/infer_stairs.py`.

Output a comprehensive, detailed report with exact designs and code references to:
`/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3/report.md`
and create your `progress.md` and `handoff.md` in your working directory.
When finished, send a message to the orchestrator (caller) with a brief summary and path to your report.
