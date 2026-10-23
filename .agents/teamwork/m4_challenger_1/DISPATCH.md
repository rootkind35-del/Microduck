# M4 Challenger 1 Dispatch

## Mission
Adversarially challenge the exported ONNX policy and observation normalizer contract:
1. Run programmatic inspection on `/tmp/m4_final_stairs.onnx`:
   - Validate input tensor name `obs`, shape `[1, 61]`.
   - Validate output tensor name `actions`, shape `[1, 14]`.
   - Check numerical outputs when feeding normalized vs raw observations: verify normalizer subtracts mean and divides by standard deviation.
   - Test inference across batch dimensions: verify single-step `[1, 61]` and multi-env batches.
2. Record adversarial findings and test code in `report.md` and explicit verdict (`APPROVE` or `REJECT`) in `handoff.md`.

## 2026-10-11T20:08:28Z
You are M4 Challenger 1 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m4_challenger_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m4_challenger_1/DISPATCH.md`.

Your objective is to adversarially challenge the exported ONNX policy and observation normalizer contract:
1. Run programmatic inspection on `/tmp/m4_final_stairs.onnx`:
   - Validate input tensor name `obs`, shape `[1, 61]`.
   - Validate output tensor name `actions`, shape `[1, 14]`.
   - Check numerical outputs when feeding normalized vs raw observations: verify normalizer subtracts mean and divides by standard deviation.
   - Test inference across batch dimensions: verify single-step `[1, 61]` and multi-env batches.
2. Record adversarial findings and test code in `report.md` and explicit verdict (`APPROVE` or `REJECT`) in `handoff.md`.
Notify the orchestrator via `send_message` when done.
