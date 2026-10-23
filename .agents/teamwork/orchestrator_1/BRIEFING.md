# BRIEFING — 2026-10-11T14:36:10Z

## Mission
Build a new Reinforcement Learning training environment (`Mjlab-StairsClimb-MicroDuck`) and dedicated standalone test simulation scene & runner for Microduck to climb high stairs (2.5cm–3.5cm), maintain dynamic balance, and self-recover when fallen, without breaking existing repository invariants.

## 🔒 My Identity
- Archetype: orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: /home/dhp01/microduck_rl/.agents/teamwork/orchestrator_1
- Original parent: parent
- Original parent conversation ID: 609211e5-9b10-48b2-a10b-41cd3ba6e35a

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: /home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md
1. **Decompose**: Decompose full scope into feature inventory and milestones (M1: Core MDP & Stairs Scene, M2: Env Config & Task Registration, M3: Standalone Runner & Sim Rehearsal, M4: E2E Verification & Adversarial Hardening).
2. **Dispatch & Execute**:
   - **Direct (iteration loop)**: Survey (3 Explorers) -> Milestones -> Iteration Loop (Explorer -> Worker -> Reviewers x2 -> Challengers x2 -> Forensic Auditor) -> Gate.
   - **Dual Track**: Implementation Track + E2E Testing Track.
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (sub-orchestrators only, last resort)
4. **Succession**: At 16 spawns, write handoff.md, spawn successor.
- **Work items**:
  1. Survey and Scope Mapping [in-progress]
  2. M1: MDP Terms & Stairs Scene [pending]
  3. M2: Stairs Env Config & Registry [pending]
  4. M3: Standalone Runner & Telemetry [pending]
  5. M4: Comprehensive E2E & Regression Testing [pending]
- **Current phase**: 0 (Survey)
- **Current focus**: Survey and Scope Mapping

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly (DISPATCH-ONLY).
- NEVER run build/test commands yourself.
- Forensic Auditor INTEGRITY VIOLATION is a hard binary veto.
- 61D observation layout strictly preserved (48 proprioception + 13 command block).
- 14 BAM XL330 servos with expand_bam_friction_fields.
- Zero breaking changes to existing environments or scenes.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.

## Current Parent
- Conversation ID: 609211e5-9b10-48b2-a10b-41cd3ba6e35a
- Updated: 2026-10-11T14:35:39Z

## Key Decisions Made
- Initialized Project Pattern with Survey phase (3 parallel Explorers).

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_survey_1 | teamwork_preview_explorer | Survey RL Environment & Curricula | completed | cb107ba3-2ea8-4def-9910-4be4f221685c |
| explorer_survey_2 | teamwork_preview_explorer | Survey MDP terms & Invariants | completed | c81cff5b-cfbb-4a2e-b5dc-a81128195553 |
| explorer_survey_3 | teamwork_preview_explorer | Survey Scene Stairs & Infer Runner | completed | c5173e23-6815-45b4-9971-8e133fdfacc1 |
| m1_spec_miner_1 | teamwork_preview_spec_miner | M1 Spec Mining (mdp.py & scene_stairs.xml) | in-progress | 580c96a9-245b-4563-b084-b85faf75f4d4 |
| m1_worker_1 | teamwork_preview_worker | Implement M1 (mdp.py & scene_stairs.xml) | completed | cc2cce95-e052-49c0-a52e-25f3fbfd97b5 |
| m1_reviewer_1 | teamwork_preview_reviewer | M1 Correctness Review | in-progress | d50f9ec4-8b00-4a58-a1c2-a052385a2661 |
| m1_reviewer_2 | teamwork_preview_reviewer | M1 Physics & Regression Review | in-progress | e74c6f85-3cd9-4cab-aea3-863c4de4b297 |
| m1_challenger_1 | teamwork_preview_challenger | M1 Adversarial MDP Testing | in-progress | da0ffe39-f220-4287-8a76-7ee303582a8a |
| m1_challenger_2 | teamwork_preview_challenger | M1 Adversarial Scene Simulation | in-progress | c8007de3-18cc-4db9-b0df-e5493309b9bc |
| m2_worker_2 | teamwork_preview_worker | Implement M2 (microduck_stairs_env_cfg.py & registration) | completed | 864a52a4-8cc4-4544-86d6-680b363bddba |
| m2_reviewer_1 | teamwork_preview_reviewer | M2 Correctness Review | in-progress | 14634fec-990c-4c8f-a7f8-6a14fae31dea |
| m2_reviewer_2 | teamwork_preview_reviewer | M2 Physics & Curriculum Review | in-progress | 8bda4340-f08f-42bc-87f5-216e5f6a6d06 |
| m2_challenger_1 | teamwork_preview_challenger | M2 Config & Normalizer Challenger | in-progress | 6bd4a7c0-f336-4402-a256-fdfb9ea2db2b |
| m2_challenger_2 | teamwork_preview_challenger | M2 Sim & Smoke Test Challenger | in-progress | 2e46278b-e96d-4fd2-b21a-0ae58ef3b3fe |
| m3_worker_1 | teamwork_preview_worker | Implement M3 (infer_stairs.py & test_infer_stairs.py) | completed | 0a5e8b5f-ed81-4b3c-9d38-68a0cf413dd9 |
| m3_reviewer_1 | teamwork_preview_reviewer | M3 Correctness Review | completed | a3de9822-84bc-40cd-97c3-cac175ac147a |
| m3_reviewer_2 | teamwork_preview_reviewer | M3 Physics & Telemetry Review | completed | 4f842605-1639-4a70-a441-c75f5302701f |
| m3_challenger_1 | teamwork_preview_challenger | M3 CLI & Preset Challenger | completed | 590b3a2a-528f-4ad6-8acf-284907aeda95 |
| m3_challenger_2 | teamwork_preview_challenger | M3 Dynamics & BAM Challenger | completed | 2c5f9159-f903-4e52-982e-6c07df246f25 |
| m3_auditor_1 | teamwork_preview_auditor | M3 Forensic Integrity Auditor | completed | 6458a652-f3d3-4d80-8282-05274f77f405 |
| m4_worker_1 | teamwork_preview_worker | M4 Comprehensive E2E Verification | completed | 3eafdd09-5752-44dc-a10a-20ee3eb413ee |
| m4_reviewer_1 | teamwork_preview_reviewer | M4 Correctness Review | in-progress | 81e32cf6-ab1a-4bd3-a59d-a3dabed7464a |
| m4_reviewer_2 | teamwork_preview_reviewer | M4 Regression Review | in-progress | 1c3ed2b3-f91b-4733-b278-4341ae7aebd1 |
| m4_challenger_1 | teamwork_preview_challenger | M4 ONNX Normalizer Challenger | in-progress | 889aec17-3b37-4942-ba25-28fd14fea507 |
| m4_challenger_2 | teamwork_preview_challenger | M4 GPU & Backlash Challenger | in-progress | b94c9e9e-1504-4188-8595-d9540a3efa6b |
| m4_auditor_1 | teamwork_preview_auditor | M4 Final Forensic Auditor | in-progress | 8689fc11-daa5-458d-a673-1ad31a156e41 |

## Succession Status
- Succession mode: top-level continuation (subagent types restricted to worker/explorer/reviewer pool)
- Cumulative spawn count: 37
- Pending subagents: 81e32cf6-ab1a-4bd3-a59d-a3dabed7464a, 1c3ed2b3-f91b-4733-b278-4341ae7aebd1, 889aec17-3b37-4942-ba25-28fd14fea507, b94c9e9e-1504-4188-8595-d9540a3efa6b, 8689fc11-daa5-458d-a673-1ad31a156e41

## Active Timers
- Heartbeat cron: f3cd8a4f-6d90-4952-bb58-a5be7069a95c/task-538
- Safety timer: pending
- On succession: kill all timers before spawning successor
- On context truncation: run `manage_task(Action="list")` — re-create if missing

## Artifact Index
- /home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md — User request
- /home/dhp01/microduck_rl/.agents/teamwork/orchestrator_1/DISPATCH.md — Dispatch log
- /home/dhp01/microduck_rl/.agents/teamwork/orchestrator_1/progress.md — Progress log
