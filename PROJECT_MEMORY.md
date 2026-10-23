# PROJECT_MEMORY — microduck_rl

> Auto-generated and maintained via `project-init` skill. Re-run `/project-init` or `/init` to update.

## Project Identity

| Field | Value |
|-------|-------|
| **Name** | `microduck_rl` |
| **Description** | RL training environments for Microduck — a ~800g, ~25cm bipedal robot with 14 Dynamixel XL330 servos built on `mjlab` (MuJoCo Warp) with PPO (`rsl_rl`). 50 Hz control, exported to ONNX for runtime deployment on `pollen-robotics/microduck`. |
| **Root** | `/home/dhp01/microduck_rl` |
| **License** | Apache-2.0 |
| **Remote** | git@github.com:rootkind35-del/Microduck.git |
| **Branch** | `main` |
| **Last Commit** | `04a4e807` by dhp01 — fix(infer): Trigger auto-fetch at t=0.5s when --auto-toss flag is provided |
| **Files** | 2,587 |
| **Lines of Code** | ~477,000 (including MJCF models, YAML/data configs, and scripts) |
| **Last Updated** | 2026-10-14T15:23:00+07:00 |

## Tech Stack

**Languages & Runtimes:**
- Python 3.11+
- CUDA / GPU accelerated via NVIDIA Warp

**Core RL & Simulation Frameworks:**
- **mjlab** (MuJoCo Warp simulation framework)
- **rsl_rl** (PPO runner & neural network policies)
- **MuJoCo** (C / Python bindings, MJCF)
- **ONNX / ONNX Runtime** (Deployment format with baked observation normalizer)
- **uv** (Fast Python package and environment manager)
- **pytest** (Comprehensive CPU regression test suite)
- **Weights & Biases (wandb)** (Experiment tracking, project `mjlab_microduck`)

## Directory Map

```
microduck_rl/
  .agents/        — Agent hooks and configuration (e.g. hooks.json)
  backup_onnx/    — Backup and exported policy checkpoints
  docs/           — Architecture, design, and deployment documentation
  logs/           — Checkpoints and training run artifacts (logs/rsl_rl/<experiment>/)
  scripts/        — Export, deployment rehearsal, evaluation, and wandb utilities
  src/
    mjlab_microduck/
      actuator/   — BAM actuator model + friction domain randomization
      publish/    — Schema-2 manifest builder and Hugging Face Hub publisher
      robot/      — MJCF models (walk, groundcontact, allcollisions, backlash) & constants
      tasks/      — Task configs, custom MDP functions, curricula, task registration
  tests/          — Invariant and regression test suite (CPU executable)
  wandb/          — W&B offline/online run metadata
```

## Key Tasks & Environments

- **Locomotion**: `microduck_velocity_env_cfg.py` (standard flat/rough walking), `microduck_velocity_swizzle_env_cfg.py`, `microduck_velocity_rollers_env_cfg.py`
- **Dynamic Maneuvers**: `microduck_roulade_env_cfg.py` (somersault/roll), `microduck_spin_env_cfg.py`, `microduck_stairs_env_cfg.py`
- **Stance & Transitions**: `microduck_sitstand_env_cfg.py`, `microduck_velstand_env_cfg.py`, `microduck_standup_env_cfg.py`
- **Manipulation / Interactive Fetch**: `microduck_fetch_env_cfg.py` (autonomous ball approach & head tracking), `microduck_ground_pick_env_cfg.py` (beak pick up from ground), `microduck_ball_kick_env_cfg.py`
- **Rollers**: `microduck_roller_slope_env_cfg.py`, `microduck_roller_crouch_env_cfg.py`, `microduck_roller_standup_env_cfg.py`

## Key Files & Entry Points

- [AGENTS.md](file:///home/dhp01/microduck_rl/AGENTS.md) — Comprehensive developer guide: invariants, sim2real rules, reward design rules, training ops.
- [pyproject.toml](file:///home/dhp01/microduck_rl/pyproject.toml) — Package config, direct dependency pins (crucial torch aarch64 cu129 wheel mappings).
- [src/mjlab_microduck/tasks/mdp.py](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/mdp.py) — **Single source of truth** for all custom MDP terms (rewards, observations, commands, curricula, events).
- [src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py) — Base walking environment config and template for other tasks.
- [src/mjlab_microduck/tasks/__init__.py](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/__init__.py) — Task registry (base + `-Backlash-` task variants).
- [src/mjlab_microduck/robot/microduck_constants.py](file:///home/dhp01/microduck_rl/src/mjlab_microduck/robot/microduck_constants.py) — Robot configs, HOME keyframe pose, BAM actuator parameters.
- [src/mjlab_microduck/actuator/friction_dr_bam.py](file:///home/dhp01/microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py) — BAM actuator physics (voltage-controlled Dynamixel XL330 + backlash encoder).
- [scripts/export.py](file:///home/dhp01/microduck_rl/scripts/export.py) — Mandatory ONNX export script that bakes running observation normalizer into ONNX graph.
- [scripts/infer_policy.py](file:///home/dhp01/microduck_rl/scripts/infer_policy.py) — CPU MuJoCo deployment rehearsal tool (includes `--auto-toss` auto-fetch mode).

## Critical Invariants (Must Never Break)

1. **61D Observation Layout (Actor)**:
   - 48D proprioception + 13D command block `[twist(3), head_pose(4), body_pose(6)]` in strict order.
   - Tasks not using a command slot must **zero-pad** (sample near zero); never delete a slot.
2. **14 Actuated Servos**:
   - Left leg: 0–4 (hip_yaw, hip_roll, hip_pitch, knee, ankle)
   - Neck/head: 5–8 (neck_pitch, head_pitch, head_yaw, head_roll)
   - Right leg: 9–13 (hip_yaw, hip_roll, hip_pitch, knee, ankle)
   - Passive joints named `passive_*` interleave on backlash/roller models; always access servos via `_servo_joint_ids` / `_servo_joint_pos` helpers in `mdp.py`.
3. **BAM Actuators**:
   - Actuators operate on voltage control.
   - Standalone envs must register `expand_bam_friction_fields` startup event.
   - Joint friction DR must scale `actuator.friction_scale` (`dof_frictionloss` is zeroed under BAM).
4. **Baked Normalizer**:
   - Running observation normalization is active. ONNX models must have the normalizer baked into the graph via `scripts/export.py`.
5. **No Policy Filtering**:
   - Policies are trained unfiltered (no action EMA low-pass filter).
6. **Reward Sign Convention**:
   - mjlab cost functions return $\ge 0$ $\rightarrow$ **negative weight**.
   - microduck penalties (`*_penalty`, `*_l1`) return $\le 0$ $\rightarrow$ **positive weight**.
   - Check: every `Episode_Reward/<penalty>` logged in wandb must be $\le 0$.

## Dev & Operational Commands

| Command | Description |
|---------|-------------|
| `uv run list-envs` | List all registered tasks |
| `uv run train <TASK_ID> --env.scene.num-envs 4096` | Train policy on GPU (add `--hf-jobs` for remote cluster) |
| `uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5` | **Smoke test** — mandatory before long runs |
| `uv run play <TASK_ID> --wandb-run-path <run_path>` | Visualize trained policy in MuJoCo viewer |
| `uv run scripts/export.py <TASK_ID> --wandb-run-path <run_path>` | Export checkpoint to ONNX with baked normalizer |
| `uv run scripts/infer_policy.py --walking out.onnx` | Deployment rehearsal on CPU MuJoCo |
| `uv run --with pytest pytest tests/` | Run CPU regression and invariant test suite |

---

_This file is injected automatically before every AI invocation via the `project-memory-injector` PreInvocation hook in `.agents/hooks.json`. The AI must read and internalize this context before responding._
