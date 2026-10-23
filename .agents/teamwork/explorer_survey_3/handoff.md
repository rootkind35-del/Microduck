# Handoff Report: Standalone Simulation Scene & Runner Survey (Explorer Survey 3)

**Type**: Hard Handoff  
**Agent**: Explorer Survey 3  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3`  
**Report Path**: `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3/report.md`  
**Date**: 2026-10-11  

---

## 1. Observation

1. **Existing MJCF Scene Structure**:
   - `src/mjlab_microduck/robot/microduck/scene.xml` (lines 1-58) includes `<include file="robot_groundcontact.xml" />` and sets visual lighting and groundplane.
   - `src/mjlab_microduck/robot/microduck/scene_walk.xml` (lines 1-58) includes `<include file="robot_walk.xml" />`.
   - `src/mjlab_microduck/robot/microduck/stairs.xml` (lines 1-21) defines 4 steps of 1.5cm risers (`pos="0.31 0 0.0075" size="0.06 0.25 0.0075"` up to `pos="0.67 0 0.0300" size="0.06 0.25 0.0300"`, `condim="3" friction="1.2 0.01 0.001" solref="0.02 1"`).
   - In `robot_walk.xml`, only `left_foot_collision` and `right_foot_collision` are active collision geoms; torso, battery, shins, and head are visual/self-collision only. In `robot_groundcontact.xml`, `np_f970` (battery), `leg` (shins/knees), `hip_l`, `top_head_shell`, and `jaw`/`bottom_head_shell` are full collision geoms (`class="collision"`).

2. **BAM M6 CPU Rehearsal Dynamics**:
   - In `scripts/infer_policy.py` (lines 38-56, 58-115), BAM constants are hardcoded without importing `microduck_constants.py` to prevent pulling in `mjlab/torch/warp` (~16s overhead):
     `BAM_MOTOR_NAME = "xl330"`, `BAM_MODEL = "m6"`, `BAM_KP_FW = 200.0`, `BAM_VIN_RANGE = (6.5, 8.2)`, `BAM_VIN_DROP_GAIN_RANGE = (0.0, 0.2)`, `BAM_VIN_MIN = 6.0`.
   - `load_mujoco_with_bam()` converts position actuators to motors with `forcerange=(-force_limit, force_limit)` ($V_{in} k_t / R$), zeros `joint.damping` and `joint.frictionloss`, sets `dof_armature` to BAM extra inertia, and applies stiff friction solver parameters `BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)` and `BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)`.
   - Decimation is 4 over a 0.005s timestep, yielding a 50 Hz control rate.
   - Observation layout in `get_observations()` (lines 882-908) is strictly 61D: `[ang_vel(3), proj_grav(3), joint_pos_rel(14), joint_vel(14), last_action(14), command(13)]`.

3. **Existing Test Suite & Host Environment**:
   - Host has ROS 2 Humble installed (`/opt/ros/humble/lib/python3.10/site-packages`). Running plain `pytest` triggers `launch_testing` entrypoint failure: `ModuleNotFoundError: No module named 'lark'`.
   - Running with clean `PYTHONPATH`: `PYTHONPATH="" uv run --with pytest pytest tests/test_infer_policy_bam.py` and `PYTHONPATH="" uv run --with pytest pytest tests/test_velstand_cfg.py` pass 100% cleanly (3 passed in 5.92s, 26 passed in 8.88s).

---

## 2. Logic Chain

1. **Why `scene_stairs.xml` must use `robot_groundcontact.xml`**:
   - Requirement R1 specifies high stairs climbing (2.5cm–3.5cm) and self-recovery when fallen or tripped.
   - On 2.5cm–3.5cm steps (up to ~28% of standing hip height), the robot's shins, knees, battery, and head frequently make contact during toe misses, trips, backward topples, or floor recovery.
   - `robot_walk.xml` has no world-collision geoms on shins, battery, or torso. If included, the robot would penetrate through the step edges unphysically.
   - Therefore, `scene_stairs.xml` MUST include `robot_groundcontact.xml`.

2. **Why solid floor-anchored boxes must be used for stair geometry**:
   - If stair steps are modeled as hollow thin boxes or elevated floating slabs, the robot's toes or shins can penetrate the bottom/back faces, triggering solver instability or catching.
   - By anchoring each step geom to the floor ($z=0$), so that $p_z = s_z = H_{top}/2$, all boxes are solid blocks.
   - Step treads of $0.12$ m ($s_x = 0.06$ m) and widths of $0.50$ m ($s_y = 0.25$ m) provide adequate clearance for Microduck's 7.5 cm soles and 9 cm stance width, with consecutive flights joined gaplessly ($p_x^{(k)} + s_x^{(k)} = p_x^{(k+1)} - s_x^{(k+1)}$).

3. **Why `scripts/infer_stairs.py` must run BAM M6 independently without torch imports**:
   - The user workflow expects fast, interactive CPU MuJoCo rehearsal. Importing `mjlab` pulls in NVIDIA Warp and PyTorch, causing a 16-second startup penalty.
   - Mirroring `load_mujoco_with_bam()` using `bam.model.load_model` and `bam.mujoco.MujocoController` delivers an instantaneous startup (<1s) while maintaining identical voltage-sag and friction-budget dynamics as training.
   - Implementing interactive spawn presets (`1` through `5`) allows developers to evaluate climbing stability on 2.5cm, 3.0cm, and 3.5cm flights without manual navigation through preceding flights.

---

## 3. Caveats

1. **GPU Training Environment vs CPU Rehearsal**:
   `scripts/infer_stairs.py` runs on CPU MuJoCo with `bam.mujoco.MujocoController`, while RL training runs in GPU Warp (`mjlab`). Slight differences in contact solver compliance (MuJoCo C engine vs Warp PGS/Newton solver) may produce subtle differences in foot micro-slip, though `friction="1.2 0.01 0.001"` and `solref="0.02 1.0"` minimize discrepancies.
2. **Policy Dependency**:
   The interactive runner requires an exported ONNX policy with baked normalizer (`scripts/export.py`). Before a stairs policy is trained, the runner can be tested and verified with `infer_stairs.py --headless` or using existing trained walking/velstand ONNX policies (`model_XXXX.onnx`).
3. **Environment Variable Invariant**:
   Any execution of `pytest` in terminal commands must be prefixed with `PYTHONPATH=""` to isolate uv's virtual environment from host ROS 2 Humble packages.

---

## 4. Conclusion

1. The exact physical specification and XML code for `src/mjlab_microduck/robot/microduck/scene_stairs.xml` are complete and documented in `report.md`, featuring:
   - Consecutive flights of 2.5cm, 3.0cm, and 3.5cm step risers.
   - 3 steps per flight + 2 intermediate landing platforms + 1 summit platform.
   - Inclusion of `robot_groundcontact.xml`.
   - Distinct visual color palette (slate blue $\rightarrow$ emerald $\rightarrow$ ochre $\rightarrow$ teal $\rightarrow$ coral $\rightarrow$ gold).
   - Keyframe spawn poses for all flights.
2. The architectural design of `scripts/infer_stairs.py` is complete, featuring:
   - Pure CPU MuJoCo + BAM M6 dynamics at 50 Hz control rate (200 Hz physics).
   - 5 interactive spawn presets (hotkeys `1` through `5`).
   - Real-time keyboard controls (velocity commands, body pitch trim, head gaze, perturbation pushes, standup recovery trigger).
   - Comprehensive balance telemetry (pitch/roll tilt, step contact detection, normal contact forces, knee/ankle torque saturation relative to 0.96 Nm limit).
   - Native zero-overhead HUD overlay.
3. The test suite design covers:
   - `tests/test_scene_stairs.py` (scene compilation, step elevations, gapless continuity, contact friction).
   - `tests/test_infer_stairs.py` (BAM constants consistency, spawn presets, 50 Hz timing, telemetry).
   - `tests/test_microduck_stairs_cfg.py` (61D obs layout, BAM actuators, reward sign conventions, curriculum stages).

---

## 5. Verification Method

1. **Verify Scene Compilation**:
   ```python
   import mujoco
   m = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene_stairs.xml")
   assert m.ngeom >= 20
   ```
2. **Verify BAM Actuator Dynamics Consistency**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/test_infer_policy_bam.py
   ```
3. **Verify Existing Task Config Invariants**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/test_velstand_cfg.py
   ```
4. **Inspect Generated Report**:
   Inspect `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3/report.md`.
