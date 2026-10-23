# DISPATCH — M1 Challenger 2

**Task**: Empirical adversarial simulation testing of `scene_stairs.xml`.
**Scope**:
- Test `src/mjlab_microduck/robot/microduck/scene_stairs.xml` under aggressive physical conditions:
  - Step contact stability: place robot with high initial downward velocity (-2.0 m/s) on each flight (2.5cm, 3.0cm, 3.5cm) and verify no penetration, no tunneling, and contacts stabilize.
  - Test lateral pushes (2.0 m/s) across step edges to verify solid ground contact and absence of gap catches.
  - Verify solver convergence and absence of NaNs over 500 physics steps across all 8 keyframe presets.
- Run tests against `tests/test_scene_stairs.py`.
- Produce verdict: `APPROVE` or `REJECT` in `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_2/handoff.md`.


## 2026-10-11T15:18:35Z
You are M1 Challenger 2 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_2

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_2/DISPATCH.md`.

Your objective is to adversarially test the physical stability and contact behavior of `src/mjlab_microduck/robot/microduck/scene_stairs.xml`:
1. Stress test in Python using MuJoCo:
   - High downward impact velocities (-2.0 m/s) on each stair flight (2.5, 3.0, 3.5 cm) and verify no penetration or tunneling.
   - Lateral perturbation pushes (2.0 m/s) across step edges to verify no gap trapping.
   - Long-horizon simulation (500 steps) across all 8 keyframes to ensure no numerical divergence or NaNs.
2. Verify all 12 obstacle boxes and landing platforms.

Write your findings to `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_2/report.md` and verdict (`APPROVE` or `REJECT`) in `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_2/handoff.md`.
Notify the orchestrator via `send_message` when done.
