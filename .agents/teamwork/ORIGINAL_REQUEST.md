# Original User Request

## 2026-10-11T14:34:00Z

Build a new Reinforcement Learning training environment and a dedicated standalone test simulation for Microduck to climb high stairs (2.5cm – 3.5cm), maintain robust dynamic balance, and automatically stand up when fallen, conforming strictly to the existing `mjlab` (MuJoCo Warp) framework without modifying existing codebase.

Working directory: /home/dhp01/microduck_rl
Integrity mode: demo

## Requirements

### R1. Standalone High Stairs RL Environment (Mjlab-StairsClimb-MicroDuck)
Design and register a brand new task environment for high stairs navigation:
- All-in-one PPO Policy: Learn high stairs traversal, posture stabilization against topple perturbations, and automatic self-recovery / standing up when tripped or fallen.
- Calibrated Stair Curriculum: Progressive stair height curriculum staging from 2.5 cm -> 3.0 cm -> 3.5 cm (representing up to ~15% of the ~25cm robot height, challenging knee clearance and torque limits).
- MDP Terms: Implement stair-climbing clearance, foot landing damping, contact distribution, and recovery potential rewards in src/mjlab_microduck/tasks/mdp.py and task config in src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py.

### R2. Dedicated Standalone Simulation Test Scene & Runner
Create a completely separate test environment without touching existing simulation files:
- Dedicated MJCF Scene: src/mjlab_microduck/robot/microduck/scene_stairs.xml with consecutive stair flights (2.5cm, 3.0cm, 3.5cm step risers) and landing platforms.
- Dedicated Runner: scripts/infer_stairs.py providing interactive MuJoCo CPU rehearsal with BAM M6 actuator dynamics, real-time keyboard commands, spawn presets on stairs, and balance telemetry.

### R3. Strict Preservation of Existing Code & Repository Invariants
- Zero Breaking Changes: Do not modify existing task configs (velocity, velstand, standup, sitstand, groundpick) or existing simulation scenes.
- 61D Observation Layout: Exactly 48D base proprioception + 13D command block [twist(3), head_pose(4), body_pose(6)].
- 14 BAM Actuators: Voltage-controlled Dynamixel XL330 model with friction DR (expand_bam_friction_fields).
- Export & Normalization: Running observation normalizer baked directly into exported ONNX models via scripts/export.py.

## Acceptance Criteria

### Technical & Functional Guardrails
- [ ] New task Mjlab-StairsClimb-MicroDuck registered in tasks/__init__.py and visible via uv run list-envs.
- [ ] 5-iteration smoke test at 64 envs passes with 0 NaNs and produces a valid baked ONNX export.
- [ ] Dedicated test simulation scripts/infer_stairs.py runs standalone at 50 Hz with scene_stairs.xml and BAM dynamics.
- [ ] Existing test suite (pytest tests/) passes 100% without regressions.
- [ ] Stair height curriculum smoothly schedules transitions across 2.5cm, 3.0cm, and 3.5cm stages.
