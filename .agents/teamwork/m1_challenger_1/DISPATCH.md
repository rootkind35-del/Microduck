# DISPATCH — M1 Challenger 1

**Task**: Empirical adversarial stress-testing of Milestone M1 changes (`mdp.py` functions and `scene_stairs.xml`).
**Scope**:
- Write adversarial oracles, boundary stress tests, or randomized property tests for the 5 stairs MDP functions:
  - Extreme values: zero velocity, supersonic velocity, inverted gravity, negative height, extreme contact forces (10,000 N).
  - Verify that no input combination produces NaN, Inf, or positive values for penalties.
  - Verify that `stairs_relative_height_progress` never produces runaway values or sign flips.
- Run tests against `src/mjlab_microduck/tasks/mdp.py`.
- Produce verdict: `APPROVE` or `REJECT` in `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_1/handoff.md`.


## 2026-10-11T15:18:18Z
You are M1 Challenger 1 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_1/DISPATCH.md`.

Your objective is to adversarially challenge and stress-test the new stairs MDP functions in `src/mjlab_microduck/tasks/mdp.py`:
1. Write and execute adversarial test cases / oracles feeding extreme and edge inputs into:
   - `stairs_foot_clearance_penalty`
   - `stairs_foot_landing_damping_penalty`
   - `stairs_riser_trip_penalty`
   - `stairs_sole_flat_penalty`
   - `stairs_relative_height_progress`
2. Test extreme scenarios: zero velocities, massive velocities, inverted gravity, extreme contact forces, step reset transitions.
3. Verify that under NO circumstances do penalties return positive values or produce NaNs/Infs.

Write your adversarial findings and test code to `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_1/report.md` and verdict (`APPROVE` or `REJECT`) in `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_1/handoff.md`.
Notify the orchestrator via `send_message` when done.
