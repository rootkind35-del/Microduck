# Progress — M2 Challenger 1

Last visited: 2026-10-11T19:36:30Z

## Status
Completed adversarial challenge of Milestone M2 configurations. Verdict: APPROVE.

## Steps
- [x] Inspect `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
- [x] Adversarially verify reward weight signs (all 32 terms + curriculum stages verified, smoke test logs confirm Episode_Reward <= 0)
- [x] Adversarially verify `feet_ground_contact` has `track_air_time=True`
- [x] Adversarially verify curriculum parameter stages (2.8cm -> 3.3cm -> 3.8cm clearance targets, recovery economics, topple pushes)
- [x] Test ONNX export with baked normalizer:
  - Discovered defect in `scripts/export.py` when invoked with `--agent zero` (`UnboundLocalError`).
  - Successfully verified ONNX export with checkpoint model (`--checkpoint-file`).
- [x] Inspect ONNX model: input shape [1, 61], output shape [1, 14], normalizer nodes (/obs_normalizer/Sub, /obs_normalizer/Div) confirmed baked in graph.
- [x] Run test suite: 329 passed, 2 skipped in 17.29s.
- [x] Write detailed report in `report.md`
- [x] Write handoff report with verdict (APPROVE) in `handoff.md`
- [x] Notify orchestrator via `send_message`
