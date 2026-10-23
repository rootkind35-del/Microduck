# Progress — M4 Challenger 1

Last visited: 2026-10-11T20:10:00Z

## Status: IN_PROGRESS

### Completed Steps
- [x] Step 1: Read ORIGINAL_REQUEST.md, PROJECT.md, and DISPATCH.md.
- [x] Step 2: Initialize BRIEFING.md and progress.md.

### Current Step
- [ ] Step 3: Investigate `/tmp/m4_final_stairs.onnx` and relevant export logic (`scripts/export.py`, `src/mjlab_microduck/export.py`).

### Upcoming Steps
- [ ] Step 4: Write adversarial verification test suite for ONNX inspection.
  - Subtask 4.1: Tensor naming and shape checks ('obs' -> 'actions', shapes [1, 61] and [1, 14]).
  - Subtask 4.2: Normalizer decomposition: verify graph subtracts mean and divides by standard deviation (or std + eps). Compare outputs of raw input through baked ONNX vs (raw - mean)/std through raw actor model / layers.
  - Subtask 4.3: Batch dimension stress testing: test [1, 61], [64, 61], [4096, 61], arbitrary batch sizes, edge batches (e.g. batch size 0 or 1).
  - Subtask 4.4: Adversarial numerical stress testing: zeros, all-ones, extreme magnitudes (+-1e6), subnormals, NaNs, infinities, dimension mismatches ([60], [62], 1D [61], 3D [1, 1, 61]).
- [ ] Step 5: Execute adversarial tests and gather empirical evidence.
- [ ] Step 6: Document findings in `report.md`.
- [ ] Step 7: Formulate verdict (`APPROVE` or `REJECT`) and produce 5-component `handoff.md`.
- [ ] Step 8: Notify orchestrator via `send_message`.
