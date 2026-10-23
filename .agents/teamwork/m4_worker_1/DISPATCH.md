# Milestone M4 Worker Dispatch: Comprehensive E2E Verification & Export Hardening

## Target Scope
All deliverables of the Microduck High Stairs project:
- `src/mjlab_microduck/tasks/mdp.py`
- `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
- `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
- `src/mjlab_microduck/tasks/__init__.py`
- `scripts/infer_stairs.py`
- `tests/test_scene_stairs.py`, `tests/test_stairs_mdp.py`, `tests/test_stairs_cfg.py`, `tests/test_infer_stairs.py`, `tests/test_stairs_e2e.py`

## Mandatory Verification Protocol
Execute each of the following commands in order, capturing verbatim stdout and return codes:

1. **Registry Verification**:
   ```bash
   PYTHONPATH="" uv run list-envs | grep StairsClimb
   ```
   Assert that both `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck` are present and correctly configured.

2. **Full Repository Regression Test Suite**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/
   ```
   Assert 100% test pass rate with 0 failures and 0 regressions.

3. **GPU Warp 5-Iteration Smoke Test**:
   ```bash
   PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
   ```
   Assert exit code 0, 0 NaNs, and confirm that all penalty terms in `Episode_Reward/` are <= 0.0.

4. **Backlash Variant Smoke Test**:
   ```bash
   PYTHONPATH="" uv run train Mjlab-StairsClimb-Backlash-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 3
   ```
   Assert exit code 0 and 0 NaNs.

5. **ONNX Export with Baked Normalizer**:
   ```bash
   PYTHONPATH="" uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/m4_final_stairs.onnx
   ```
   Inspect exported model in Python:
   - Input tensor shape: `(1, 61)`
   - Output tensor shape: `(1, 14)`
   - Verify that running observation normalizer nodes are baked into the ONNX graph.

6. **Standalone 50 Hz Rehearsal Simulation**:
   ```bash
   PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug
   ```
   Assert 100 steps completed cleanly with 0 NaNs and live telemetry.

## Reporting
Document all executed commands, verbatim outputs, timings, and artifact inspection results in `/home/dhp01/microduck_rl/.agents/teamwork/m4_worker_1/report.md` and `/home/dhp01/microduck_rl/.agents/teamwork/m4_worker_1/handoff.md`.
Notify the orchestrator via `send_message` when complete.

## 2026-10-11T19:58:22Z
You are M4 Worker in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m4_worker_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m4_worker_1/DISPATCH.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your objective is to execute the comprehensive E2E verification protocol:
1. Registry Verification:
   `PYTHONPATH="" uv run list-envs | grep StairsClimb`
2. Full Repository Regression Test Suite:
   `PYTHONPATH="" uv run --with pytest pytest tests/`
3. GPU Warp 5-Iteration Smoke Test:
   `PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`
   Verify 0 NaNs and that all penalty rewards in `Episode_Reward/` are <= 0.0.
4. Backlash Variant Smoke Test:
   `PYTHONPATH="" uv run train Mjlab-StairsClimb-Backlash-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 3`
   Verify 0 NaNs.
5. ONNX Export with Baked Normalizer:
   `PYTHONPATH="" uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/m4_final_stairs.onnx`
   Inspect the exported ONNX model in Python: verify input shape [1, 61], output shape [1, 14], and presence of running observation normalizer nodes.
6. Standalone 50 Hz Rehearsal Simulation:
   `PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug`
   Verify 100 steps completed cleanly with 0 NaNs and live telemetry.

Document all commands, exact verbatim stdout/stderr, return codes, and artifact inspections in:
`/home/dhp01/microduck_rl/.agents/teamwork/m4_worker_1/report.md`
and `/home/dhp01/microduck_rl/.agents/teamwork/m4_worker_1/handoff.md`.
Notify the orchestrator via `send_message` when complete.
