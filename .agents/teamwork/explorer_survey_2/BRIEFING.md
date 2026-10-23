# BRIEFING — 2026-10-11T14:47:00Z

## Mission
Investigate custom MDP functions in `src/mjlab_microduck/tasks/mdp.py` and design the new stairs MDP terms (clearance, landing damping, contact distribution, recovery potential) conforming to microduck invariants and reward conventions.

## 🔒 My Identity
- Archetype: explorer
- Roles: MDP terms investigation, reward design, compatibility analysis
- Working directory: /home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2
- Original parent: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Milestone: high-stairs RL environment MDP term specification

## 🔒 Key Constraints
- Read-only investigation — do NOT implement / modify existing source code
- Adhere strictly to 61D observation layout and BAM actuator conventions
- Follow reward sign conventions: mjlab cost returns >= 0 (weight < 0); microduck penalty returns <= 0 (weight > 0)
- Ensure all joint/geom indexing uses helpers compatible with both plain and backlash models

## Current Parent
- Conversation ID: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Updated: 2026-10-11T14:47:00Z

## Investigation State
- **Explored paths**: `ORIGINAL_REQUEST.md`, `AGENTS.md`, `DISPATCH.md`, `src/mjlab_microduck/tasks/mdp.py`, `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`, `src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py`, `src/mjlab_microduck/robot/microduck/stairs.xml`, test suite `tests/`
- **Key findings**:
  1. Standard mjlab `feet_clearance` punishes lifting above target; high stairs requires one-sided clearance deficit penalty `stairs_foot_clearance_penalty`.
  2. Touchdown damping requires both contact force threshold ($F > 8$ N) and vertical impact velocity damping ($v_z < -0.08$ m/s) in `stairs_foot_landing_damping_penalty`.
  3. Riser-edge collisions are detected via horizontal shear excess ($F_{xy} > 0.5 F_z$) in `stairs_riser_trip_penalty`, paired with sole horizontal alignment `stairs_sole_flat_penalty`.
  4. Standard `height_progress` fails on stairs due to absolute elevation accumulation ($z - z_{origin} > 0.115$ m on higher steps). Resolved by stance-relative trunk height $z_{rel} = z_{trunk} - \min(z_{feet})$.
  5. 9 existing terms identified for direct reuse; 100% test suite pass rate verified.
- **Unexplored areas**: None within Survey 2 scope. Implementation and config wiring handed off to subsequent implementation stages.

## Key Decisions Made
- All new stairs penalty terms designed in self-negating microduck style (`*_penalty`, return $\le 0.0$, weight $> 0.0$) with safe tensor NaN guards.
- All joint access routed through cached `_servo_joint_ids(env, asset)`.

## Artifact Index
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2/report.md` — Comprehensive MDP term investigation & specification
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2/progress.md` — Progress log and liveness heartbeat
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2/handoff.md` — 5-component hard handoff report
