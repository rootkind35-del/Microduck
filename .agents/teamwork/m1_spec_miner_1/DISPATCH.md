## 2026-10-11T14:51:19Z
You are M1 Spec Miner in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m1_spec_miner_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Also read `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2/report.md` and `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3/report.md`.

Your objective is to extract exact line-level specifications, imports, helpers, and type signatures for:
1. Adding the 5 custom stairs MDP functions to `src/mjlab_microduck/tasks/mdp.py`:
   - `stairs_foot_clearance_penalty`
   - `stairs_foot_landing_damping_penalty`
   - `stairs_riser_trip_penalty`
   - `stairs_sole_flat_penalty`
   - `stairs_relative_height_progress`
   Verify exact integration point in `src/mjlab_microduck/tasks/mdp.py` (line number, section heading, imports needed).
2. Creating `src/mjlab_microduck/robot/microduck/scene_stairs.xml`:
   - Verify every tag, asset, material, light, floor, box geoms, site beacons, and keyframes.

Write a concise, precise report to `/home/dhp01/microduck_rl/.agents/teamwork/m1_spec_miner_1/report.md` and your `handoff.md`.
Notify the orchestrator via `send_message` when done.
