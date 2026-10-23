# BRIEFING — 2026-10-11T19:35:00Z

## Mission
Adversarially challenge Milestone M2 configurations (microduck_stairs_env_cfg.py and ONNX export).

## 🔒 My Identity
- Archetype: EMPIRICAL CHALLENGER
- Roles: critic, specialist
- Working directory: /home/dhp01/microduck_rl/.agents/teamwork/m2_challenger_1
- Original parent: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Milestone: M2
- Instance: 1 of 1

## 🔒 Key Constraints
- Review-only — do NOT modify implementation code
- Run verification code empirically — do not trust unverified claims
- If cannot reproduce a bug empirically, it does not count

## Current Parent
- Conversation ID: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Updated: 2026-10-11T19:35:00Z

## Review Scope
- **Files to review**: `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`, `src/mjlab_microduck/tasks/__init__.py`, `scripts/export.py`
- **Interface contracts**: `PROJECT.md`, `AGENTS.md`
- **Review criteria**:
  1. Reward term weight signs (self-negating penalties > 0, mjlab costs < 0, positive rewards > 0).
  2. `feet_ground_contact` sensor configuration (`track_air_time=True`).
  3. Curriculum parameter stages (target clearance 2.8 -> 3.3 -> 3.8 cm, stair height stages).
  4. ONNX export verification with baked normalizer: input (1, 61), output (1, 14), normalizer nodes present.

## Key Decisions Made
- [Verdict]: APPROVE Milestone M2 environment configurations.
- [Finding]: Uncovered defect in `src/mjlab_microduck/export.py` where `--agent zero` triggers `UnboundLocalError: cannot access local variable 'runner'`. Verified that checkpoint export functions 100% correctly and conforms to all ONNX contracts.

## Artifact Index
- `DISPATCH.md` — Dispatch instructions
- `report.md` — Detailed adversarial challenge report
- `handoff.md` — Handoff report with verdict (APPROVE)
- `progress.md` — Progress heartbeat log

## Attack Surface
- **Hypotheses tested**:
  - H1: Reward weights could double-negate or allow reward hacking $\rightarrow$ Disproven. All penalties evaluate $\le 0.0$ in simulation.
  - H2: `feet_ground_contact` could lack `track_air_time=True` $\rightarrow$ Disproven. Verified `track_air_time=True`.
  - H3: Curriculum clearance target stages could mismatch risers $\rightarrow$ Disproven. Verified 2.8cm -> 3.3cm -> 3.8cm.
  - H4: ONNX export command with `--agent zero` fails $\rightarrow$ CONFIRMED BUG in `src/mjlab_microduck/export.py` (`UnboundLocalError`). Checkpoint-based export verified cleanly.
- **Vulnerabilities found**:
  - `src/mjlab_microduck/export.py:264`: `runner` unbound when `DUMMY_MODE` is enabled (`--agent zero`).
  - `PYTHONPATH` collision: System ROS libraries cause `ModuleNotFoundError: No module named 'lark'` if `PYTHONPATH` is not cleared.
- **Untested angles**:
  - Hardware deployment on physical robot (handled in downstream milestones/repos).
