## 2026-10-11T14:35:39Z

You are the Project Orchestrator for the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/orchestrator_1
The user request has been recorded verbatim in: /home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md

Please read /home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md and AGENTS.md carefully.

Mission:
Build a new Reinforcement Learning training environment and a dedicated standalone test simulation for Microduck to climb high stairs (2.5cm – 3.5cm), maintain robust dynamic balance, and automatically stand up when fallen, conforming strictly to the existing mjlab (MuJoCo Warp) framework without modifying existing codebase.

Key Requirements:
1. R1: Standalone High Stairs RL Environment (`Mjlab-StairsClimb-MicroDuck`):
   - Progressive stair height curriculum staging from 2.5 cm -> 3.0 cm -> 3.5 cm (representing up to ~15% of the ~25cm robot height).
   - MDP terms in `src/mjlab_microduck/tasks/mdp.py` (stair-climbing clearance, foot landing damping, contact distribution, recovery potential). Group properly under task heading.
   - Task config in `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`.
   - Register in `src/mjlab_microduck/tasks/__init__.py`.
2. R2: Dedicated Standalone Simulation Test Scene & Runner:
   - Dedicated MJCF Scene: `src/mjlab_microduck/robot/microduck/scene_stairs.xml` with consecutive stair flights (2.5cm, 3.0cm, 3.5cm step risers) and landing platforms.
   - Dedicated Runner: `scripts/infer_stairs.py` providing interactive MuJoCo CPU rehearsal with BAM M6 actuator dynamics, real-time keyboard commands, spawn presets on stairs, and balance telemetry.
3. R3: Strict Preservation of Existing Code & Repository Invariants:
   - Zero Breaking Changes: Do not modify existing task configs (velocity, velstand, standup, sitstand, groundpick) or existing simulation scenes.
   - 61D Observation Layout: Exactly 48D base proprioception + 13D command block [twist(3), head_pose(4), body_pose(6)].
   - 14 BAM Actuators with friction DR (`expand_bam_friction_fields`).
   - Baked Normalizer: Export with `scripts/export.py`.
   - Reward sign conventions: mjlab cost returns >= 0 (negative weight); microduck penalty returns <= 0 (positive weight).
4. Acceptance Criteria:
   - `uv run list-envs` shows `Mjlab-StairsClimb-MicroDuck`.
   - 5-iteration smoke test at 64 envs passes with 0 NaNs and produces a valid baked ONNX export.
   - `scripts/infer_stairs.py` runs standalone at 50 Hz with `scene_stairs.xml` and BAM dynamics.
   - `pytest tests/` passes 100% without regressions, plus new tests for stairs cfg and runner.
   - Stair height curriculum smoothly schedules 2.5cm, 3.0cm, and 3.5cm stages.

Orchestrate the development cycle with your subagents, maintaining plan.md, progress.md, and BRIEFING.md in your working directory. Report completion with comprehensive verification evidence when finished.

## 2026-10-11T19:17:19Z

Resume execution. Please check current status of M2, M3, and M4, and continue orchestrating the remaining milestones.
