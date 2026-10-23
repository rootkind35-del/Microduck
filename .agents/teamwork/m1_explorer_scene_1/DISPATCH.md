## 2026-10-11T14:51:19Z
You are M1 Scene Explorer in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Also read `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3/report.md`.

Your objective is to validate `scene_stairs.xml` against MuJoCo schema and design `tests/test_scene_stairs.py`.
Specifically:
1. Validate that `scene_stairs.xml` loads cleanly with `mujoco.MjModel.from_xml_path()`.
2. Verify all geometries: Flight 1 (2.5cm risers), Landing 1, Flight 2 (3.0cm risers), Landing 2, Flight 3 (3.5cm risers), Top Summit. Check positions, sizes, gapless edges, `condim="3"`, friction 1.2, and materials.
3. Validate keyframes: `INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`.
4. Provide the exact test file design for `tests/test_scene_stairs.py`.

Write your report to `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1/report.md` and your `handoff.md`.
Notify the orchestrator via `send_message` when done.
