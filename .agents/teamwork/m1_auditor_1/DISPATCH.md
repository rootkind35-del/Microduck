# DISPATCH — M1 Forensic Auditor

**Task**: Forensic integrity audit of Milestone M1 changes.
**Scope**:
- Examine all files modified or created in Milestone M1:
  - `src/mjlab_microduck/tasks/mdp.py`
  - `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
  - `tests/test_scene_stairs.py`
  - `tests/test_stairs_mdp.py`
- Verify integrity against the Forensic Integrity rules:
  - No cheating, no hardcoded test outputs or return values tailored only to pass tests.
  - No dummy or facade implementations that bypass actual physics.
  - No mock bypassing in production source code.
  - True mathematical implementations for clearance, landing damping, riser shear excess, and relative height progress.
  - Exact riser elevations in XML (2.5cm, 3.0cm, 3.5cm) without fudge factors.
- Run tests independently:
  `PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`
- Report verdict: `CLEAN` or `INTEGRITY VIOLATION` in `/home/dhp01/microduck_rl/.agents/teamwork/m1_auditor_1/handoff.md`.

## 2026-10-11T15:18:35Z
You are M1 Forensic Auditor in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m1_auditor_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m1_auditor_1/DISPATCH.md`.

Your objective is to perform a forensic integrity audit on Milestone M1:
1. Audit source files:
   - `src/mjlab_microduck/tasks/mdp.py`
   - `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
   - `tests/test_scene_stairs.py`
   - `tests/test_stairs_mdp.py`
2. Forensic checks:
   - Static analysis: Are implementations genuine, or are there hardcoded mock returns / test shortcuts?
   - Do penalties implement genuine physics calculations (foot clearance, contact forces, shear ratios, relative trunk height)?
   - Are stair dimensions in `scene_stairs.xml` genuine 2.5cm, 3.0cm, 3.5cm riser boxes anchored to floor?
   - Run verification independently:
     `PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`
3. Report verdict:
   Must conclude with either `CLEAN` or `INTEGRITY VIOLATION`.

Write full report with evidence chains to `/home/dhp01/microduck_rl/.agents/teamwork/m1_auditor_1/report.md` and verdict in `/home/dhp01/microduck_rl/.agents/teamwork/m1_auditor_1/handoff.md`.
Notify the orchestrator via `send_message` when done.
