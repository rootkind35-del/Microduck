# DISPATCH — Explorer Survey 2

## 2026-10-11T14:37:24Z
From: Orchestrator (f3cd8a4f-6d90-4952-bb58-a5be7069a95c)
Content:
You are Explorer Survey 2 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/AGENTS.md`.

Your objective is to investigate custom MDP functions in `src/mjlab_microduck/tasks/mdp.py` and design the new stairs MDP terms.
Specifically:
1. Review the structure of `src/mjlab_microduck/tasks/mdp.py` — existing reward terms, observations, commands, event functions, and grouping conventions.
2. Design the mathematical definitions and signatures for:
   - Stair-climbing clearance reward/penalty (foot height trajectory over step edge).
   - Foot landing damping (penalize excessive foot impact force / normal velocity upon landing on steps).
   - Contact distribution (encouraging proper sole contacts, penalizing riser-edge tripping).
   - Recovery potential rewards (potential-based upright / standing-up shaping when fallen or tilted, avoiding exploitation / farming).
3. Check reward sign conventions strictly: mjlab cost returns >= 0 (negative weight); microduck penalty (`*_penalty`, `*_l1`) returns <= 0 (positive weight).
4. Verify helper functions in `mdp.py` (`_servo_joint_ids`, `_servo_joint_pos`, foot bodies/geoms) for compatibility with both plain and backlash robot models.
5. Identify any existing terms in `mdp.py` that can be reused or combined with new stairs terms.

Output a comprehensive, detailed report with exact function designs and code references to:
`/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2/report.md`
and create your `progress.md` and `handoff.md` in your working directory.
When finished, send a message to the orchestrator (caller) with a brief summary and path to your report.
