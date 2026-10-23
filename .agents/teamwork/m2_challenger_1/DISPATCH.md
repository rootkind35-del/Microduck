# DISPATCH — M2 Challenger 1

**Task**: Empirical adversarial stress-testing of Milestone M2 configurations.
**Scope**:
- Test `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`:
  - Verify reward weight signs: every self-negating penalty has positive weight; every cost has negative weight.
  - Verify that `stairs_foot_clearance_penalty` target scales across scheduled curricula (2.8cm -> 3.3cm -> 3.8cm).
  - Verify that `feet_ground_contact` has `track_air_time=True`.
  - Test ONNX export with baked normalizer:
    `uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/test_export.onnx`
    Check exported model inputs (61D) and outputs (14D).
- Produce verdict: `APPROVE` or `REJECT` in `/home/dhp01/microduck_rl/.agents/teamwork/m2_challenger_1/handoff.md`.

## 2026-10-11T19:27:16Z
You are M2 Challenger 1 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m2_challenger_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m2_challenger_1/DISPATCH.md`.

Your objective is to adversarially challenge Milestone M2 configurations:
1. Inspect `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`:
   - Validate every reward term weight sign.
   - Verify `feet_ground_contact` has `track_air_time=True`.
   - Verify curriculum parameter stages.
2. Test ONNX export with baked normalizer:
   `uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/test_export.onnx`
   Inspect the exported ONNX model: check input shape (1, 61), output shape (1, 14), and verify normalizer nodes.

Write your findings to `/home/dhp01/microduck_rl/.agents/teamwork/m2_challenger_1/report.md` and verdict (`APPROVE` or `REJECT`) in `/home/dhp01/microduck_rl/.agents/teamwork/m2_challenger_1/handoff.md`.
Notify the orchestrator via `send_message` when done.
