# Handoff Report: Explorer Survey 1 — Stairs RL Architecture & Curriculum

**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1`  
**Date**: 2026-10-11T14:48:30Z  
**Type**: Hard Handoff (Investigation Complete)  
**Deliverable Report**: `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/report.md`  

---

## 1. Observation

1. **Base Environment & Invariants**:
   - `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`: Base locomotion config building on `make_velocity_env_cfg()` (line 263). Wires 61D actor observation (lines 534–704: 48D proprioception + 13D commands: `twist(3)`, `head_pose(4)`, `body_pose(6)`). Delays: `joint_vel` lag = 1 (line 613), IMU lag $[0, 1]$ (lines 567, 571). Actuators: BAM voltage-controlled Dynamixel XL330 with `expand_bam_friction_fields` startup event (line 366). DR: CoM, head CoM, pseudo-inertia mass/inertia, BAM friction, reflected armature (lines 415–514).
   - `src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py`: Walking + protective fall + recovery architecture. Uses `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (`robot_allcollisions.xml`, 70 collision geoms, line 410). Hardened solver settings: `cfg.sim.nconmax = 200` (line 411), `iterations = 30`, `ls_iterations = 50` (lines 778–779). Contact softening: `_soften_terrain_contacts` sets `solref = [0.04, 1.0]` (lines 172–191).
   - Recovery layer in `velstand`: Gated strictly on `tilt > 40°` (`REWARD_GATE_TILT_DEG = 40.0`, line 261). Potential progress: `upright_progress` (weight 5.0, line 462) and `height_progress` (weight 30.0, line 475). Attempt tax relief: `FALLEN_SMOOTHNESS_SCALE = 0.1` scaling `action_rate_l2` and `joint_torque_rate` while fallen (lines 300, 501–510). Recovery economics: `fallen_tax` (-0.5, line 561) and `recovery_success` (+10.0, line 581) releasing at `tilt < 25°` and `z > 0.09 m` (`RECOVERED_UP_TILT_DEG = 25.0`, `RECOVERED_UP_Z = 0.09`, lines 278–279).

2. **Terrain & Stairs Generation**:
   - `mjlab.terrains.primitive_terrains.BoxPyramidStairsTerrainCfg`: Procedural pyramid staircase taking `step_height_range`, `step_width`, `platform_width`, `border_width`. In curriculum mode (`curriculum=True`), row difficulty scales linearly: $\text{step\_height} = \text{range}[0] + \text{difficulty} \times (\text{range}[1] - \text{range}[0])$.
   - `src/mjlab_microduck/tasks/slope_terrain.py`: Demonstrates custom `SubTerrainCfg` subclassing (`FlatRampTerrainCfg`) dynamically creating box geoms directly in `spec.body("terrain")` with `origin` outputs.
   - Lesson on terrain elevations (`microduck_velstand_env_cfg.py:128-134`): Resets must be origin-relative (`env.scene.terrain.env_origins[env_ids, 2]`), otherwise robots spawn inside elevated terrain boxes.

3. **Curricula & Step Counting**:
   - `NUM_STEPS_PER_ENV = 24`. Steps scale as `step = iteration * 24`.
   - `microduck_mdp.reward_weight`: Mutates live RewardManager term config across scheduled steps (`src/mjlab_microduck/tasks/mdp.py:3659-3672`).
   - `microduck_mdp.terrain_levels_slope`: Progression using `terrain.update_env_origins(env_ids, move_up, move_down)` (`mdp.py:3732-3748`).

4. **Task Registration & Runner**:
   - `src/mjlab_microduck/tasks/__init__.py`: Registered via `register_mjlab_task` with `runner_cls=MicroduckOnPolicyRunner` (lines 85–248).
   - Backlash variant registered via `_BACKLASH_TASKS` using `MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG` (lines 268–294).
   - `scripts/export.py`: Wraps `mjlab_microduck.export.run_export`, loading `env_cfg` with `play=True` and baking the running normalizer into ONNX.

---

## 2. Logic Chain

1. **Robot Model**:
   - *Observation*: `MICRODUCK_WALK_ROBOT_CFG` only defines collision geoms on the foot soles (`left_foot_collision`, `right_foot_collision`). `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` has 70 geoms covering shins, knees, thighs, and housings.
   - *Reasoning*: Climbing steps of $2.5\,\text{cm} \to 3.5\,\text{cm}$ (up to 15% robot height) will cause physical contact between stair risers and the lower legs. If the robot falls on stairs, it must physically collide with the step treads.
   - *Deduction*: `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` is strictly required for `microduck_stairs_env_cfg.py`.

2. **Solver Hardening**:
   - *Observation*: Terrain box edges cause normal force discontinuities, leading to solver NaN blowups on GPU Warp unless damped (`_soften_terrain_contacts`, `solref=[0.04, 1.0]`).
   - *Reasoning*: Stairs consist of multiple adjacent box geoms with sharp vertical transitions.
   - *Deduction*: `cfg.scene.spec_fn = _soften_terrain_contacts`, `cfg.sim.nconmax = 200`, `iterations = 30`, `ls_iterations = 50` must be enabled.

3. **Stairs Terrain Strategy**:
   - *Observation*: In `mjlab`, `ManagerBasedRlEnv` requires procedural terrain generation with sub-terrain origins for 4096 parallel GPU environments. Standalone XML (`scene_stairs.xml`) is used for CPU rehearsal.
   - *Reasoning*: Parallel training cannot load arbitrary static scene XMLs per-environment; it requires `TerrainEntityCfg(terrain_type="generator")`.
   - *Deduction*: GPU training in `microduck_stairs_env_cfg.py` should use procedural `TerrainGeneratorCfg` with `BoxPyramidStairsTerrainCfg` (or a straight stairs generator), while CPU interactive evaluation in `scripts/infer_stairs.py` uses `src/mjlab_microduck/robot/microduck/scene_stairs.xml`.

4. **Curriculum Design**:
   - *Observation*: Physical stair riser height is compiled into the MuJoCo model at startup, whereas reward weights and param schedules can be updated per step via `env.common_step_counter`.
   - *Reasoning*: A spatial curriculum (different difficulty rows in the terrain grid from $2.5\,\text{cm}$ to $3.5\,\text{cm}$) allows agents to advance across stair heights as they succeed. Simultaneously, foot clearance and swing height reward parameters can be scheduled in step stages ($0 \to 1500 \to 3000$ iters) to match the increasing step heights.
   - *Deduction*: A two-tier curriculum combining spatial terrain promotion (`terrain.update_env_origins`) with temporal step schedules (`microduck_mdp.reward_weight`) satisfies the progressive $2.5\,\text{cm} \to 3.0\,\text{cm} \to 3.5\,\text{cm}$ requirement cleanly.

5. **Self-Recovery Integration**:
   - *Observation*: Gating recovery on low height caused sitting/squatting exploitation in `velstand` Run 1/2. An unscaled attempt tax caused robots to lie still.
   - *Reasoning*: A stair climbing policy must not learn to sit on steps to farm recovery rewards. It must only trigger recovery when genuinely toppled (`tilt > 40°`).
   - *Deduction*: Gate recovery strictly on `tilt > 40°`, use unfarmable potential rewards (`upright_progress`, `height_progress`), scale smoothness penalties by `FALLEN_SMOOTHNESS_SCALE = 0.1` while fallen, and enforce `fallen_tax` (-0.5) with `recovery_success` bounty (+10.0).

---

## 3. Caveats

1. **Warm Start vs From-Scratch**:
   - While `velstand` supports warm-starting from `441tzs6d@3750`, training from scratch is also fully supported. If warm start is used, `MICRODUCK_WARM_START=1` must be passed so `common_step_counter` resets to 0.
2. **Terrain Grid Dimensions**:
   - `TerrainGeneratorCfg` with `num_rows=6` and `size=(10.0, 10.0)` creates 6 difficulty levels. The box count must be monitored to ensure it stays well within memory limits (avoiding OOM).
3. **Sensor-safe terms**:
   - Raycast sensors (`foot_height_scan`) can return non-finite values if a ray misses geometry. The critic obs must use `foot_height_safe` to prevent rsl_rl `check_nan` crashes.

---

## 4. Conclusion

The architecture for `microduck_stairs_env_cfg.py` and task registration in `tasks/__init__.py` is fully planned and ready for implementation. It preserves all critical repository invariants (61D actor observation, 14 BAM actuators, baked ONNX normalizer) while integrating a calibrated $2.5\,\text{cm} \to 3.0\,\text{cm} \to 3.5\,\text{cm}$ stair curriculum and dynamic tilt-gated self-recovery.

---

## 5. Verification Method

To verify the plan and subsequent implementation:
1. **List Task Registry**:
   ```bash
   uv run list-envs | grep StairsClimb
   ```
   *Expected*: `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck` visible in table.
2. **Run Config Invariant Tests**:
   ```bash
   uv run --with pytest pytest tests/test_stairs_cfg.py
   ```
   *Expected*: Verifies 61D actor obs shape, 14 BAM actuators, negative cost weights, positive self-negating penalty weights.
3. **Run Existing Test Suite (Zero Regressions)**:
   ```bash
   uv run --with pytest pytest tests/
   ```
   *Expected*: 100% pass without regressions.
4. **Run Smoke Test (GPU/CPU)**:
   ```bash
   uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
   ```
   *Expected*: Steps 5 iterations with 0 NaNs and finite loss.
5. **Test ONNX Export**:
   ```bash
   uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/smoke_stairs.onnx
   ```
   *Expected*: Produces valid ONNX graph with baked normalizer.
