# Milestone M1 Implementation & Verification Report

**Author**: M1 Worker (`m1_worker_1`)  
**Date**: 2026-10-11  
**Milestone**: M1 — Core MDP Terms & Standalone Simulation Scene  
**Status**: COMPLETE (Verified 100% Pass, 0 Regressions)

---

## 1. Scope & Owned Deliverables

M1 Worker implemented and verified all four assigned deliverables:
1. `src/mjlab_microduck/robot/microduck/scene_stairs.xml`: Standalone 3-flight stair climbing scene with 2.5cm, 3.0cm, and 3.5cm risers, landing platforms, visual materials, and 8 spawn/settle keyframes.
2. `tests/test_scene_stairs.py`: 10 comprehensive invariant tests for `scene_stairs.xml` validating compilation, kinematics, riser heights, continuity, solid grounding, contact parameters, keyframes, and dynamic settling stability.
3. `src/mjlab_microduck/tasks/mdp.py`: Appended verified stairs MDP functions under `# Stairs Climbing Task — Clearance, Landing Damping, Contact & Recovery Terms` banner:
   - `stairs_foot_clearance_penalty` (one-sided forward swing clearance deficit, $\le 0$)
   - `stairs_foot_landing_damping_penalty` (touchdown force and normal velocity damping, $\le 0$)
   - `stairs_riser_trip_penalty` (horizontal riser shear collision & stubbing penalty, $\le 0$)
   - `stairs_sole_flat_penalty` (foot sole parallel alignment during stance on treads, $\le 0$)
   - `stairs_relative_height_progress` (elevation-invariant potential-based recovery shaping, $\Delta \Phi$)
   - `_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` with pre-initialized `site_ids = [1, 6]` to guarantee site isolation on both real scenes and unit test mocks.
4. `tests/test_stairs_mdp.py`: 11 unit tests verifying tensor shape and device consistency, self-negating penalty invariants ($\le 0$), potential shaping lifecycle, elevation invariance, and site mismatch isolation.

---

## 2. Verification Commands & Execution Logs

### Verification 1: M1 Targeted Unit Test Suite
Command:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py
```
Output:
```
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/dhp01/microduck_rl
configfile: pyproject.toml
plugins: typeguard-4.4.4, anyio-4.12.1
collecting ... collecting 21 items                                                            collected 21 items                                                             

tests/test_scene_stairs.py ..........                                    [ 47%]
tests/test_stairs_mdp.py ...........                                     [100%]

============================== 21 passed in 9.47s ==============================
```

### Verification 2: Full Codebase Regression Suite
Command:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/
```
Output:
```
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/dhp01/microduck_rl
configfile: pyproject.toml
plugins: typeguard-4.4.4, anyio-4.12.1
collecting ... collecting 16 items                                                            collected 277 items                                                            

tests/test_aarch64_cuda_torch.py .....s                                  [  2%]
tests/test_crouch_glide.py ..........                                    [  5%]
tests/test_descent_speed.py ....                                         [  7%]
tests/test_fetch_cfg.py ...                                              [  8%]
tests/test_ground_pick_cfg.py .....                                      [ 10%]
tests/test_ground_pick_pose.py .......                                   [ 12%]
tests/test_head_pose_bias.py ........                                    [ 15%]
tests/test_hf_jobs_flag.py ............                                  [ 19%]
tests/test_infer_fetch.py ..                                             [ 20%]
tests/test_infer_heading_control.py .                                    [ 20%]
tests/test_infer_policy_bam.py ...                                       [ 22%]
tests/test_nan_guard.py ....                                             [ 23%]
tests/test_obs_nan_guard.py ........                                     [ 26%]
tests/test_publish_manifest.py ..............................            [ 37%]
tests/test_roller_crouch_cfg.py ...                                      [ 38%]
tests/test_roller_slope_cfg.py ............                              [ 42%]
tests/test_roller_standup_cfg.py .....................................   [ 55%]
tests/test_scene_stairs.py ..........                                    [ 59%]
tests/test_slope_curriculum.py ....                                      [ 61%]
tests/test_slope_terrain.py ........                                     [ 63%]
tests/test_spin.py ........................                              [ 72%]
tests/test_spin_cfg.py ..........                                        [ 76%]
tests/test_stairs_e2e.py .sss....s.........s.....                        [ 84%]
tests/test_stairs_mdp.py ...........                                     [ 88%]
tests/test_swizzle_head_cfg.py .                                         [ 89%]
tests/test_velstand_cfg.py ..........................                    [ 98%]
tests/test_wheel_glide.py ....                                           [100%]

======================= 271 passed, 6 skipped in 14.94s ========================
```

---

## 3. Key Decisions & Technical Notes

1. **Foot Site Selection (`_FEET_SITE_CFG`)**:
   In `SceneEntityCfg`, `site_ids` defaults to `slice(None)` until `resolve(scene)` is called by an active `ManagerBasedRlEnv`. To support both full simulation resolution and fast CPU mock unit tests (`MockEnv` without full scene resolver), `_FEET_SITE_CFG.site_ids = [1, 6]` is explicitly pre-populated. In real scenes (`robot_groundcontact.xml` and `robot_allcollisions.xml`), `left_foot` is site index 1 and `right_foot` is site index 6, ensuring total consistency with `find_sites` validation.
2. **Strict Preservation of Existing Envs & Invariants**:
   No modifications were made to any existing task configurations or existing simulation scene files.
