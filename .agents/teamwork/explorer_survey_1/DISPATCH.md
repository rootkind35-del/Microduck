# DISPATCH — Explorer Survey 1

**Task**: Investigate existing RL environment configurations, task registration, curriculum structures, and how to design `microduck_stairs_env_cfg.py` and register `Mjlab-StairsClimb-MicroDuck`.
**Scope**:
- Examine `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`, `microduck_velstand_env_cfg.py`, `microduck_standup_env_cfg.py`, `tasks/__init__.py`.
- Identify how environment configs are structured (DR, obs, noise, delays, BAM friction startup event, NaN-guard).
- Determine how stair height curriculum should be scheduled (2.5cm -> 3.0cm -> 3.5cm) using mjlab curricula / step boundaries (`NUM_STEPS_PER_ENV = 24`, `iteration * 24`).
- Identify how fallen recovery / standing up can be integrated into the stairs climbing policy.
- Output report to `.agents/teamwork/explorer_survey_1/report.md`.

## 2026-10-11T14:37:24Z
[Message] sender=f3cd8a4f-6d90-4952-bb58-a5be7069a95c priority=MESSAGE_PRIORITY_HIGH
Objective: Investigate existing RL environment configs, curricula, and registration patterns to plan `microduck_stairs_env_cfg.py` and task registration in `tasks/__init__.py`.
Specifically:
1. Examine `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`, `microduck_velstand_env_cfg.py`, `microduck_standup_env_cfg.py`, `tasks/__init__.py`, and `backlash.py`.
2. Inspect how environment classes, scenes, terrain/ground models, observations (61D invariant), commands, and DR are set up.
3. Investigate how terrain / stairs can be integrated into the mjlab env cfg. Does mjlab support custom terrain mesh / stairs terrain or scene XML? How do other envs configure scenes?
4. Investigate how the progressive stair height curriculum (2.5cm -> 3.0cm -> 3.5cm) should be structured (iterations, step counting: NUM_STEPS_PER_ENV = 24, params-curriculum vs mdp.reward_weight).
5. Investigate how self-recovery / standing up when tripped or fallen can be integrated alongside stairs climbing.
6. Check `RslRlOnPolicyRunnerCfg` conventions, experiment naming, and export compatibility.

Output a comprehensive, detailed report with exact code references to:
`/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/report.md`
and create your `progress.md` and `handoff.md` in your working directory.
When finished, send a message to the orchestrator (caller) with a brief summary and path to your report.

## 2026-10-11T14:48:03Z
[Message] sender=f3cd8a4f-6d90-4952-bb58-a5be7069a95c priority=MESSAGE_PRIORITY_HIGH
**Context**: Orchestration Survey Phase
**Content**: Explorer 2 and Explorer 3 have completed their reports. Checking in on your status for the RL environment architecture and curriculum report. Are you finalizing `report.md`?
**Action**: Please report current progress and expected completion.
