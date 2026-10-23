# TEST_INFRA.md — Microduck High Stairs RL & Standalone Simulation Test Infrastructure

**Document Version**: 1.0.0  
**Target Environment**: `Mjlab-StairsClimb-MicroDuck` / `Mjlab-StairsClimb-Backlash-MicroDuck`  
**Framework**: `mjlab` (MuJoCo Warp GPU training) / MuJoCo 3.x CPU Standalone Rehearsal  
**Test Suite Target**: `tests/test_stairs_e2e.py` (E2E Opaque-Box Suite)  
**Author**: E2E Test Writer (`test_writer_e2e_1`)  
**Date**: 2026-10-11  

---

## 1. Executive Overview & Testing Architecture

This document defines the 4-tier testing infrastructure for the Microduck High Stairs RL environment, dedicated standalone simulation scene (`scene_stairs.xml`), and deployment rehearsal runner (`infer_stairs.py`). Microduck is a ~800 g, ~25 cm bipedal robot with 14 Dynamixel XL330 servos. Traversing stairs with 2.5 cm, 3.0 cm, and 3.5 cm risers (up to 15% of body height and 28% of nominal standing hip height) presents extreme challenges for knee clearance, torque limits, dynamic balance, and contact stability.

### The 4-Tier Testing Methodology

```
+-------------------------------------------------------------------------------+
|                       TIER 4: REAL-WORLD SCENARIOS                            |
| Full flight ascent, topple recovery, keyboard teleoperation, sim2real audit  |
+-------------------------------------------------------------------------------+
                                        ^
+-------------------------------------------------------------------------------+
|                    TIER 3: CROSS-FEATURE INTERACTIONS                         |
| Clearance * Curriculum, Landing * Physics, Recovery * Runner HUD, DR * Export  |
+-------------------------------------------------------------------------------+
                                        ^
+-------------------------------------------------------------------------------+
|                 TIER 2: BOUNDARY & CORNER CASES (>=5 per feature)             |
| Zero velocities, force spikes, tilt extremes, step edge discontinuities       |
+-------------------------------------------------------------------------------+
                                        ^
+-------------------------------------------------------------------------------+
|                 TIER 1: FEATURE COVERAGE (>=5 per feature F1-F11)             |
| Primary behavior, nominal parameter ranges, functional contracts             |
+-------------------------------------------------------------------------------+
```

---

## 2. Feature Inventory (F1 – F11)

| Feature ID | Feature Name | Component / Target File | Primary Role |
|---|---|---|---|
| **F1** | MDP Stair Clearance Penalties | `src/mjlab_microduck/tasks/mdp.py` | One-sided foot clearance and swing apex clearance |
| **F2** | MDP Foot Landing Damping | `src/mjlab_microduck/tasks/mdp.py` | Touchdown impact force damping and pre-impact deceleration |
| **F3** | MDP Contact Distribution & Anti-Tripping | `src/mjlab_microduck/tasks/mdp.py` | Tread placement, vertical riser trip penalty, double support |
| **F4** | MDP Elevation-Invariant Recovery Potential | `src/mjlab_microduck/tasks/mdp.py` | Relative height shaping, upright progress, fallen tax/bounty |
| **F5** | Dedicated Standalone MJCF Scene | `src/mjlab_microduck/robot/microduck/scene_stairs.xml` | 3 flights (2.5cm, 3.0cm, 3.5cm), landing platforms, keyframes |
| **F6** | High Stairs RL Environment Config | `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py` | Allcollisions robot, 61D obs, softened contacts, BAM DR |
| **F7** | Progressive Stair Height Curriculum | `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py` | 2-tier spatial and step-staged curriculum (2.5 -> 3.0 -> 3.5 cm) |
| **F8** | Task Registration & Backlash Twin | `src/mjlab_microduck/tasks/__init__.py` | Registry entries for base and `-Backlash-` task variants |
| **F9** | Dedicated Standalone Rehearsal Runner | `scripts/infer_stairs.py` | 50 Hz CPU loop, BAM M6 dynamics, 5 spawn presets, telemetry |
| **F10** | Invariant Unit Test Suite | `tests/test_scene_stairs.py`, `tests/test_stairs_cfg.py`, `tests/test_infer_stairs.py` | Fast CPU invariant unit test coverage across modules |
| **F11** | Full E2E & Smoke Test Verification & Export | `scripts/export.py`, `uv run train`, `tests/test_stairs_e2e.py` | Smoke test (64 envs, 5 iters), baked ONNX export, 100% test pass |

---

## 3. Tier 1: Feature Coverage (>=5 Tests Per Feature across F1–F11)

### F1: MDP Stair Clearance Penalties
- **F1-T1**: `test_foot_clearance_penalty_nominal_swing`  
  - *Description*: Verifies that `stairs_foot_clearance_penalty` returns a negative penalty (value $\le 0$) when foot height is below `target_height` during forward swing.
  - *Expected Source*: `explorer_survey_2/report.md:88`, $P = -\sum \Delta h \cdot \max(0, \|v_{xy}\| - v_{min}) \le 0$.
- **F1-T2**: `test_foot_clearance_penalty_one_sided_free_lifting`  
  - *Description*: Verifies that foot height exceeding `target_height` produces exactly `0.0` penalty (unlike mjlab symmetric clearance penalty).
  - *Expected Source*: `explorer_survey_2/report.md:89`, $\Delta h = \max(0, h_{target} - h) = 0$ for $h \ge h_{target}$.
- **F1-T3**: `test_foot_clearance_penalty_command_gate`  
  - *Description*: Verifies that when the forward velocity command is zero ($\|v_{cmd}\| < \text{threshold}$), the clearance penalty is gated to `0.0`.
  - *Expected Source*: `mdp.py` command gating pattern; stationary robot is not taxed for resting feet on ground.
- **F1-T4**: `test_swing_clearance_penalty_at_touchdown`  
  - *Description*: Verifies that `stairs_swing_clearance_penalty` evaluates only on `first_contact` frames and penalizes apex height deficit.
  - *Expected Source*: `explorer_survey_2/report.md:102`, $P_{peak} = -(\text{deficit}/h_{target})^2 \cdot \mathbb{I}(\text{first\_contact})$.
- **F1-T5**: `test_foot_clearance_batch_tensor_shape_and_device`  
  - *Description*: Verifies that output shape is `(num_envs,)` on the correct torch device without NaNs or Infs.
  - *Expected Source*: `AGENTS.md` and `mdp.py:check_nan`.

### F2: MDP Foot Landing Damping
- **F2-T1**: `test_foot_impact_penalty_above_threshold`  
  - *Description*: Verifies that landing impact forces exceeding $F_{thresh} = 8.0$ N produce self-negating penalty $\le 0.0$.
  - *Expected Source*: `explorer_survey_2/report.md:118`, $P_{impact} = -\max(0, \|F\| - 8.0) \le 0$.
- **F2-T2**: `test_foot_impact_penalty_below_threshold_zero_cost`  
  - *Description*: Verifies that nominal support forces ($\le 8.0$ N) produce exactly `0.0` penalty, ensuring normal standing weight support is unpenalized.
  - *Expected Source*: Microduck robot mass $m \approx 0.8$ kg $\implies m g \approx 7.85$ N; normal contact is free.
- **F2-T3**: `test_landing_velocity_penalty_downward_touchdown`  
  - *Description*: Verifies that touchdown downward velocity $|v_z| > 0.08$ m/s produces quadratic penalty at first contact.
  - *Expected Source*: `explorer_survey_2/report.md:128`, $P_{down\_vel} = -\max(0, -v_z - 0.08)^2 \le 0$.
- **F2-T4**: `test_landing_damping_combined_evaluation`  
  - *Description*: Verifies combined damping formulation evaluating both force excess and downward velocity simultaneously on `first_contact`.
  - *Expected Source*: `explorer_survey_2/report.md:136`.
- **F2-T5**: `test_landing_damping_inactive_during_flight`  
  - *Description*: Verifies that during flight phases (no contact), impact penalties evaluate to `0.0`.
  - *Expected Source*: `first_contact == False` masks all impact penalties to zero.

### F3: MDP Contact Distribution & Anti-Tripping
- **F3-T1**: `test_riser_trip_penalty_horizontal_shear_excess`  
  - *Description*: Verifies that horizontal shear force on foot collision exceeding $\mu_{tread} \cdot F_z$ triggers riser trip penalty.
  - *Expected Source*: `explorer_survey_2/report.md:157`, riser collision generates high $\|F_{xy}\|$ with low $F_z$.
- **F3-T2**: `test_riser_trip_penalty_tread_contact_zero`  
  - *Description*: Verifies that pure normal force on tread ($F_{xy} \approx 0, F_z > 0$) produces zero riser penalty.
  - *Expected Source*: Normal tread contact satisfies $\|F_{xy}\| \le 0.5 F_z$.
- **F3-T3**: `test_sole_flat_penalty_horizontal_gravity_projection`  
  - *Description*: Verifies that tilted foot contact produces penalty proportional to $\|\mathbf{g}_{site, xy}\|^2$.
  - *Expected Source*: `explorer_survey_2/report.md:167`, horizontal gravity projection in foot frame.
- **F3-T4**: `test_sole_flat_penalty_zero_when_horizontal`  
  - *Description*: Verifies that when foot sole is exactly parallel to world XY plane ($\mathbf{g}_{site} = [0, 0, -1]^T$), penalty is exactly `0.0`.
  - *Expected Source*: Flat sole on tread has zero horizontal gravity projection.
- **F3-T5**: `test_double_support_distribution_penalty_load_asymmetry`  
  - *Description*: Verifies that during double support, load imbalance $|F_{z,l} - F_{z,r}| / (F_{z,l} + F_{z,r})$ produces penalty $\le 0.0$.
  - *Expected Source*: `explorer_survey_2/report.md:171`.

### F4: MDP Elevation-Invariant Recovery Potential
- **F4-T1**: `test_relative_height_progress_invariant_to_stair_elevation`  
  - *Description*: Verifies that standing upright on step 0 ($z=0$) and step 3 ($z=0.09$) produce the same relative height potential $z_{rel} \approx 0.115$ m.
  - *Expected Source*: `explorer_survey_2/report.md:196`, $z_{rel} = z_{trunk} - \min(z_{feet})$.
- **F4-T2**: `test_relative_height_progress_zero_during_normal_climbing`  
  - *Description*: Verifies that climbing stairs with steady posture yields $\Delta z_{rel} \approx 0.0$, preventing climbing from farming recovery rewards.
  - *Expected Source*: Feet and trunk elevate simultaneously during ascent.
- **F4-T3**: `test_upright_progress_orientation_potential`  
  - *Description*: Verifies $\Delta \cos(\text{tilt})$ pays positive reward when rising from prone to standing, and charges negative cost when falling.
  - *Expected Source*: `explorer_survey_2/report.md:207`, $\Phi_{upright} = 1.0 - 2.0(q_x^2 + q_y^2)$.
- **F4-T4**: `test_recovery_progress_zero_on_fresh_resets`  
  - *Description*: Verifies that on fresh resets (`episode_length_buf <= 1`), potential differences evaluate to `0.0`.
  - *Expected Source*: Potential differences across resets must be masked to prevent reset jackpot spikes.
- **F4-T5**: `test_fallen_tax_and_recovery_bounty_hysteresis`  
  - *Description*: Verifies fallen tax is armed when $\text{tilt} > 40^\circ$ and released only when $\text{tilt} < 25^\circ$, and one-shot bounty fires on exit.
  - *Expected Source*: `explorer_survey_2/report.md:221-228`.

### F5: Dedicated Standalone MJCF Scene
- **F5-T1**: `test_scene_stairs_loads_in_mujoco`  
  - *Description*: Verifies `src/mjlab_microduck/robot/microduck/scene_stairs.xml` parses and compiles cleanly with `mujoco.MjModel.from_xml_path`.
  - *Expected Source*: `PROJECT.md:8`, standalone MJCF scene compilation.
- **F5-T2**: `test_scene_stairs_three_flight_riser_heights`  
  - *Description*: Verifies that flight 1 risers = 2.5 cm ($0.025 \pm 0.0002$ m), flight 2 risers = 3.0 cm ($0.030 \pm 0.0002$ m), flight 3 risers = 3.5 cm ($0.035 \pm 0.0002$ m).
  - *Expected Source*: `PROJECT.md:17` and `explorer_survey_3/report.md:78-98`.
- **F5-T3**: `test_scene_stairs_landing_platforms`  
  - *Description*: Verifies existence of intermediate landing platforms `landing_1` ($z=0.075$ m), `landing_2` ($z=0.165$ m), and `landing_summit` ($z=0.270$ m).
  - *Expected Source*: `explorer_survey_3/report.md:91-98`.
- **F5-T4**: `test_scene_stairs_gapless_step_continuity`  
  - *Description*: Verifies that all steps and platforms form a continuous, gapless surface along +X without horizontal voids.
  - *Expected Source*: `explorer_survey_3/report.md:16`.
- **F5-T5**: `test_scene_stairs_keyframes_and_collision_model`  
  - *Description*: Verifies all 8 keyframes (`INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`) and inclusion of `robot_groundcontact.xml` or `robot_allcollisions.xml`.
  - *Expected Source*: `PROJECT.md:41-43` and `explorer_survey_3/report.md:54`.

### F6: High Stairs RL Environment Config
- **F6-T1**: `test_stairs_env_cfg_61d_observation_layout`  
  - *Description*: Verifies that the actor observation space has total dimension 61D: 48D base proprioception + 13D commands in exact order.
  - *Expected Source*: `AGENTS.md` and `PROJECT.md:7`.
- **F6-T2**: `test_stairs_env_cfg_robot_allcollisions`  
  - *Description*: Verifies that the robot configuration uses `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` with 70 collision geoms.
  - *Expected Source*: `PROJECT.md:5` and `explorer_survey_1/report.md:74`.
- **F6-T3**: `test_stairs_env_cfg_contact_softening_spec`  
  - *Description*: Verifies that `_soften_terrain_contacts` sets `solref=[0.04, 1.0]` and solver iterations $\ge 30$.
  - *Expected Source*: `explorer_survey_1/report.md:88-93`.
- **F6-T4**: `test_stairs_env_cfg_bam_actuators_and_startup_event`  
  - *Description*: Verifies 14 BAM actuators and registration of `expand_bam_friction_fields` in `events.startup`.
  - *Expected Source*: `AGENTS.md` and `explorer_survey_1/report.md:32`.
- **F6-T5**: `test_stairs_env_cfg_reward_signs_invariant`  
  - *Description*: Verifies that all self-negating penalties have positive weights ($w > 0$) and standard cost terms have negative weights ($w < 0$).
  - *Expected Source*: `AGENTS.md: Reward design` and `tests/test_velstand_cfg.py`.

### F7: Progressive Stair Height Curriculum
- **F7-T1**: `test_stair_curriculum_stages_progression`  
  - *Description*: Verifies step-staged curriculum transitions from 2.5 cm (stage 1) -> 3.0 cm (stage 2) -> 3.5 cm (stage 3).
  - *Expected Source*: `ORIGINAL_REQUEST.md:15` and `PROJECT.md:19`.
- **F7-T2**: `test_stair_curriculum_step_conversion`  
  - *Description*: Verifies that curriculum step counter uses `iteration * 24` (`NUM_STEPS_PER_ENV = 24`).
  - *Expected Source*: `AGENTS.md: Curricula`.
- **F7-T3**: `test_stair_curriculum_clearance_target_scaling`  
  - *Description*: Verifies that `target_height` parameter in clearance reward scales from 0.030 m to 0.045 m across curriculum stages.
  - *Expected Source*: `explorer_survey_2/report.md:98`.
- **F7-T4**: `test_stair_curriculum_manager_mutation`  
  - *Description*: Verifies that curriculum mutates term cfgs via manager instances (`env.event_manager.get_term_cfg`), not silent writes to `env.cfg`.
  - *Expected Source*: `AGENTS.md: Curricula rules`.
- **F7-T5**: `test_stair_curriculum_warm_start_reset`  
  - *Description*: Verifies `MICRODUCK_WARM_START=1` logic resets common step counter so curriculum stages start from stage 1 on warm start.
  - *Expected Source*: `mdp.py: Patch 5` and `AGENTS.md: Training ops`.

### F8: Task Registration & Backlash Twin
- **F8-T1**: `test_task_registered_in_gymnasium`  
  - *Description*: Verifies `Mjlab-StairsClimb-MicroDuck` is registered in `tasks/__init__.py` and accessible in gymnasium registry.
  - *Expected Source*: `ORIGINAL_REQUEST.md:32` and `PROJECT.md:20`.
- **F8-T2**: `test_backlash_twin_task_registered`  
  - *Description*: Verifies `Mjlab-StairsClimb-Backlash-MicroDuck` is registered and tracked in `_BACKLASH_TASKS`.
  - *Expected Source*: `AGENTS.md` and `PROJECT.md:20`.
- **F8-T3**: `test_task_entrypoint_factory_call`  
  - *Description*: Verifies entry point factory function returns an instance of `ManagerBasedRlEnvCfg`.
  - *Expected Source*: `tasks/__init__.py` registration convention.
- **F8-T4**: `test_runner_experiment_name_unique`  
  - *Description*: Verifies runner config sets a unique experiment name (e.g., `microduck_stairs`).
  - *Expected Source*: `AGENTS.md: Building a new env`.
- **F8-T5**: `test_backlash_twin_mirrors_base_robot_model`  
  - *Description*: Verifies backlash twin mirrors base robot collision model (`allcollisions`), preventing confounded A/B comparisons.
  - *Expected Source*: `AGENTS.md: Invariants`.

### F9: Dedicated Standalone Rehearsal Runner
- **F9-T1**: `test_infer_stairs_cli_argparser`  
  - *Description*: Verifies CLI arguments in `scripts/infer_stairs.py` (`--model`, `--task`, `--spawn`, `--no-bam`, `--auto-stand`).
  - *Expected Source*: `PROJECT.md:21` and `explorer_survey_3/report.md:22`.
- **F9-T2**: `test_infer_stairs_five_spawn_presets`  
  - *Description*: Verifies spawn presets: `floor`, `stairs_25`, `stairs_30`, `stairs_35`, `summit`.
  - *Expected Source*: `explorer_survey_3/report.md:22`.
- **F9-T3**: `test_infer_stairs_bam_controller_50hz`  
  - *Description*: Verifies 50 Hz control loop with decimation 4 over 200 Hz physics and BAM M6 voltage controller.
  - *Expected Source*: `AGENTS.md` and `explorer_survey_3/report.md:20`.
- **F9-T4**: `test_infer_stairs_telemetry_hud_fields`  
  - *Description*: Verifies telemetry structure logs tilt angle (pitch/roll), foot contact force, and joint torque saturation ratio.
  - *Expected Source*: `explorer_survey_3/report.md:22`.
- **F9-T5**: `test_infer_stairs_fall_detection_and_recovery_trigger`  
  - *Description*: Verifies that when $|\text{tilt}| > 45^\circ$, fall flag is raised and auto-standup recovery sequence is initiated.
  - *Expected Source*: `explorer_survey_3/report.md:22`.

### F10: Invariant Test Suite
- **F10-T1**: `test_suite_scene_stairs_unit_tests`  
  - *Description*: Verifies `tests/test_scene_stairs.py` executes cleanly on CPU and validates scene geometry and keyframes.
  - *Expected Source*: `PROJECT.md:22, 28`.
- **F10-T2**: `test_suite_stairs_cfg_unit_tests`  
  - *Description*: Verifies `tests/test_stairs_cfg.py` tests env config invariants, obs layout, and reward weights.
  - *Expected Source*: `PROJECT.md:22, 29`.
- **F10-T3**: `test_suite_infer_stairs_unit_tests`  
  - *Description*: Verifies `tests/test_infer_stairs.py` tests runner CLI, keyframe mapping, and BAM parameters.
  - *Expected Source*: `PROJECT.md:22, 30`.
- **F10-T4**: `test_suite_cpu_only_execution`  
  - *Description*: Verifies all invariant tests execute strictly on CPU without requiring CUDA or GPU device allocations.
  - *Expected Source*: `AGENTS.md: tests/`.
- **F10-T5**: `test_suite_existing_tests_zero_regression`  
  - *Description*: Verifies existing test suite (`pytest tests/`) passes 100% without breakage.
  - *Expected Source*: `ORIGINAL_REQUEST.md:35`.

### F11: Full E2E & Smoke Test Verification & Export
- **F11-T1**: `test_export_script_bakes_running_normalizer`  
  - *Description*: Verifies `scripts/export.py` incorporates `RunningMeanStd` observation normalizer into ONNX graph.
  - *Expected Source*: `AGENTS.md: Invariants` and `ORIGINAL_REQUEST.md:27`.
- **F11-T2**: `test_exported_onnx_input_output_shapes`  
  - *Description*: Verifies exported ONNX model takes input shape `(1, 61)` and outputs action shape `(1, 14)`.
  - *Expected Source*: `AGENTS.md` and `PROJECT.md:7`.
- **F11-T3**: `test_exported_onnx_excludes_passive_joints`  
  - *Description*: Verifies `_get_base_metadata_no_passive` filters out `passive_*` joints from action output metadata.
  - *Expected Source*: `mdp.py: Patch 4` and `AGENTS.md`.
- **F11-T4**: `test_smoke_test_command_execution_contract`  
  - *Description*: Verifies 5-iteration smoke test command `uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5` contract.
  - *Expected Source*: `AGENTS.md: Commands` and `ORIGINAL_REQUEST.md:33`.
- **F11-T5**: `test_e2e_nan_guard_in_rewards_and_returns`  
  - *Description*: Verifies MDP rewards pass through Patch 1 and Patch 6 NaN guards, preventing NaN propagation into PPO buffers.
  - *Expected Source*: `mdp.py: Patches 1, 2, 6`.

---

## 4. Tier 2: Boundary & Corner Cases (>=5 Tests Per Feature across F1–F11)

### F1: Boundary & Corner Cases (Clearance Penalties)
- **F1-B1**: `test_clearance_zero_swing_velocity`  
  - *Boundary*: Foot swing velocity is exactly 0.0 m/s while clearance deficit is maximal ($h = 0$).  
  - *Requirement*: Moving gate ($\|v\| > v_{min}$) must completely suppress the penalty to 0.0.
- **F1-B2**: `test_clearance_exact_target_height`  
  - *Boundary*: Foot height is exactly equal to `target_height` ($h = 0.035$ m, deficit = 0.0).  
  - *Requirement*: Clamp produces exactly 0.0 without numerical jitter.
- **F1-B3**: `test_clearance_extreme_negative_height`  
  - *Boundary*: Sensor reports negative height ($h = -0.05$ m) due to mesh penetration or step edge raycast singularity.  
  - *Requirement*: Deficit is safely clamped or bounded, does not cause overflow or NaN.
- **F1-B4**: `test_clearance_supersonic_swing_velocity`  
  - *Boundary*: Foot velocity spikes ($10.0$ m/s) during numerical solver kick.  
  - *Requirement*: Output is finite, device-resident, and does not produce Inf.
- **F1-B5**: `test_clearance_all_envs_disabled_commands`  
  - *Boundary*: Batch where all command vectors are `[0, 0, 0]`.  
  - *Requirement*: Output tensor is strictly zero across all envs.

### F2: Boundary & Corner Cases (Landing Damping)
- **F2-B1**: `test_impact_force_exact_threshold`  
  - *Boundary*: Normal contact force is exactly $F = 8.000$ N.  
  - *Requirement*: Penalty is exactly 0.0.
- **F2-B2**: `test_impact_massive_collision_spike`  
  - *Boundary*: Robot falls from height causing impact force $F = 1000.0$ N.  
  - *Requirement*: Penalty scales linearly (or clamped) without numerical overflow.
- **F2-B3**: `test_landing_velocity_positive_upward`  
  - *Boundary*: Foot makes contact while moving upward ($v_z = +0.2$ m/s) e.g., brushing against step riser bottom.  
  - *Requirement*: $v_{down} = \max(0, -v_z) = 0.0$, velocity damping penalty is 0.0.
- **F2-B4**: `test_first_contact_single_frame_pulse`  
  - *Boundary*: `first_contact` is true on step $t$ and false on step $t+1$.  
  - *Requirement*: Penalty applies on $t$ and immediately ceases on $t+1$ even if foot remains on ground.
- **F2-B5**: `test_zero_contact_sensor_data`  
  - *Boundary*: Sensor returns all zero forces and zero velocities.  
  - *Requirement*: Output tensor is 0.0 without divide-by-zero errors.

### F3: Boundary & Corner Cases (Contact Distribution & Anti-Tripping)
- **F3-B1**: `test_riser_pure_horizontal_collision`  
  - *Boundary*: Toe impacts riser face horizontally ($F_x = 20.0$ N, $F_z = 0.0$ N).  
  - *Requirement*: Shear excess is maximal ($\|F_{xy}\| - \mu F_z = 20.0 - 0 = 20.0$), penalty is strongly negative.
- **F3-B2**: `test_riser_friction_coefficient_boundary`  
  - *Boundary*: Contact force ratio $\|F_{xy}\| / F_z$ is exactly equal to $\mu_{tread} = 0.5$.  
  - *Requirement*: Penalty transition is continuous and evaluates to 0.0 at the boundary.
- **F3-B3**: `test_sole_flat_vertical_wall_contact`  
  - *Boundary*: Foot sole is tilted 90 degrees ($\mathbf{g}_{site} = [1, 0, 0]^T$).  
  - *Requirement*: $\|\mathbf{g}_{xy}\|^2 = 1.0$, maximal flat penalty.
- **F3-B4**: `test_double_support_zero_force_singularity`  
  - *Boundary*: Both feet in contact but forces are near zero ($F_{z,l} = 0, F_{z,r} = 0$).  
  - *Requirement*: Epsilon in denominator ($\epsilon = 1e-4$) prevents zero-division NaN.
- **F3-B5**: `test_single_foot_stance_masking`  
  - *Boundary*: Only one foot is touching the ground during gait stride.  
  - *Requirement*: Double support distribution penalty is masked to 0.0.

### F4: Boundary & Corner Cases (Recovery Potential)
- **F4-B1**: `test_recovery_inverted_robot_tilt_180`  
  - *Boundary*: Robot completely inverted upside down ($\text{tilt} = \pi$, $\cos(\text{tilt}) = -1.0$).  
  - *Requirement*: Upright potential evaluates to -1.0, progress calculation remains bounded.
- **F4-B2**: `test_recovery_perfect_upright_tilt_zero`  
  - *Boundary*: Robot perfectly vertical ($\text{tilt} = 0.0$, $\cos(\text{tilt}) = 1.0$).  
  - *Requirement*: Upright potential evaluates to 1.0, progress is 0.0 at steady state.
- **F4-B3**: `test_relative_height_negative_clearance`  
  - *Boundary*: Foot position is above trunk link (robot flipped on back).  
  - *Requirement*: $z_{rel} \le 0.0$, clamp ensures potential does not crash.
- **F4-B4**: `test_relative_height_ceiling_saturation`  
  - *Boundary*: Robot launched into air with trunk $z_{rel} = 0.5$ m.  
  - *Requirement*: Potential clamps at $h_{ceiling} = 0.115$ m, progress difference evaluates to 0.0.
- **F4-B5**: `test_fallen_tax_hysteresis_boundary`  
  - *Boundary*: Robot tilt is exactly $25.0^\circ$ (boundary between fallen release and latched state).  
  - *Requirement*: State machine preserves previous arm state without flapping.

### F5: Boundary & Corner Cases (MJCF Scene)
- **F5-B1**: `test_scene_stairs_riser_height_tolerances`  
  - *Boundary*: Measure step riser elevations across all 9 step geoms.  
  - *Requirement*: Height deviations must be within $\pm 0.1$ mm of exact design heights (2.5, 3.0, 3.5 cm).
- **F5-B2**: `test_scene_stairs_zero_x_overlap`  
  - *Boundary*: Check adjacent step boundary coordinates $x_{start, i+1} - x_{end, i}$.  
  - *Requirement*: Gap and overlap must be identically 0.0 mm ($|x_{start} - x_{end}| < 1e-6$).
- **F5-B3**: `test_scene_stairs_minimum_surface_friction`  
  - *Boundary*: Inspect tangential friction coefficient across all step and platform geoms.  
  - *Requirement*: Tangential friction $\mu \ge 1.0$ (configured at 1.2) for secure bipedal contact.
- **F5-B4**: `test_scene_stairs_keyframe_joint_limits`  
  - *Boundary*: Check all keyframe qpos coordinates against joint range limits $[q_{min}, q_{max}]$.  
  - *Requirement*: All keyframe joint positions must strictly fall within joint limits.
- **F5-B5**: `test_scene_stairs_material_asset_completeness`  
  - *Boundary*: Inspect all `<material>` and `<texture>` references in `<geom>` tags.  
  - *Requirement*: Zero missing or unresolvable assets during MJCF compilation.

### F6: Boundary & Corner Cases (Environment Config)
- **F6-B1**: `test_env_cfg_obs_dimension_integrity`  
  - *Boundary*: Compute tensor dimensions across all observation groups.  
  - *Requirement*: Actor observation space dimension must be strictly equal to 61 (never 60 or 62).
- **F6-B2**: `test_env_cfg_action_space_excludes_passive`  
  - *Boundary*: Inspect action dimension and joint selectors.  
  - *Requirement*: Action space must be strictly 14 (XL330 servos only), excluding all `passive_*` joints.
- **F6-B3**: `test_env_cfg_zero_command_sampling`  
  - *Boundary*: Check command generator configuration for exact-zero sampling.  
  - *Requirement*: `zero_command_prob > 0.0` to train deployment idle state.
- **F6-B4**: `test_env_cfg_solver_nconmax_upper_bound`  
  - *Boundary*: Check `sim.nconmax` buffer allocation.  
  - *Requirement*: `nconmax >= 200` to prevent GPU contact buffer overflow during multi-geom stair contacts.
- **F6-B5**: `test_env_cfg_friction_dr_non_accumulating`  
  - *Boundary*: Apply domain randomization over 10 consecutive resets.  
  - *Requirement*: Friction scaling must not accumulate or drift away from baseline distribution.

### F7: Boundary & Corner Cases (Curriculum)
- **F7-B1**: `test_curriculum_step_zero_boundary`  
  - *Boundary*: Step counter is exactly 0.  
  - *Requirement*: Curriculum activates stage 1 parameters (2.5 cm target) without uninitialized lookup errors.
- **F7-B2**: `test_curriculum_exact_transition_step`  
  - *Boundary*: Step counter at exact transition boundary ($N = 1500 \times 24 = 36000$).  
  - *Requirement*: Deterministic stage transition to stage 2 without off-by-one errors.
- **F7-B3**: `test_curriculum_large_step_saturation`  
  - *Boundary*: Step counter exceeds final stage ($N = 10000 \times 24 = 240000$).  
  - *Requirement*: Clamps cleanly to stage 3 parameters (3.5 cm target) without array index out of bounds.
- **F7-B4**: `test_curriculum_reverse_spawn_mix_probability`  
  - *Boundary*: Check spawn mix ratios across curriculum stages.  
  - *Requirement*: Spawn probabilities sum to 1.0 on every stage.
- **F7-B5**: `test_curriculum_weight_schedule_monotonicity`  
  - *Boundary*: Audit reward weight schedules across curriculum stages.  
  - *Requirement*: Regularizer attempt taxes do not introduce abrupt discontinuous drops that cause policy collapse.

### F8: Boundary & Corner Cases (Task Registration)
- **F8-B1**: `test_registration_exact_task_id_strings`  
  - *Boundary*: Query gymnasium registry for exact strings `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck`.  
  - *Requirement*: Exact match without trailing spaces or case mismatches.
- **F8-B2**: `test_registration_idempotency`  
  - *Boundary*: Re-import `mjlab_microduck.tasks` multiple times.  
  - *Requirement*: Does not crash with duplicate registration exceptions.
- **F8-B3**: `test_registration_backlash_twin_table_entry`  
  - *Boundary*: Query `_BACKLASH_TASKS` dictionary.  
  - *Requirement*: Maps `Mjlab-StairsClimb-MicroDuck` to `Mjlab-StairsClimb-Backlash-MicroDuck`.
- **F8-B4**: `test_registration_entry_point_signature`  
  - *Boundary*: Inspect registered entry point callable signature.  
  - *Requirement*: Accepts `play: bool` and `rough: bool` keyword arguments.
- **F8-B5**: `test_registration_list_envs_visibility`  
  - *Boundary*: Run CLI registry query.  
  - *Requirement*: Task ID is present in `list-envs` output.

### F9: Boundary & Corner Cases (Runner Rehearsal)
- **F9-B1**: `test_runner_headless_flag_execution`  
  - *Boundary*: Run runner with `--no-gui` / headless mode.  
  - *Requirement*: Steps simulation cleanly without X11 / Wayland display requirement.
- **F9-B2**: `test_runner_invalid_spawn_preset_handling`  
  - *Boundary*: Pass `--spawn invalid_preset`.  
  - *Requirement*: Exits with clear error message listing valid presets without unhandled traceback.
- **F9-B3**: `test_runner_zero_velocity_idle_stability`  
  - *Boundary*: Step simulation for 3.0 seconds with zero velocity command.  
  - *Requirement*: Robot maintains standing pose without drifting or falling.
- **F9-B4**: `test_runner_maximum_torque_saturation_hud`  
  - *Boundary*: Inject high load torque ($1.5$ Nm) into knee servo.  
  - *Requirement*: Telemetry reports saturation ratio $\ge 1.0$ (relative to XL330 0.96 Nm limit) without crash.
- **F9-B5**: `test_runner_timestep_decimation_consistency`  
  - *Boundary*: Check physics timestep $dt = 0.005$ s and decimation $4$.  
  - *Requirement*: Control frequency is exactly $200 / 4 = 50$ Hz.

### F10: Boundary & Corner Cases (Test Suite Architecture)
- **F10-B1**: `test_suite_fast_execution_time`  
  - *Boundary*: Execute complete stairs invariant unit tests.  
  - *Requirement*: Total execution time $< 10$ seconds on standard CPU.
- **F10-B2**: `test_suite_no_gpu_device_dependency`  
  - *Boundary*: Run test suite in an environment with `CUDA_VISIBLE_DEVICES=""`.  
  - *Requirement*: All unit and invariant tests pass without CUDA initialization errors.
- **F10-B3**: `test_suite_isolated_state_no_order_dependence`  
  - *Boundary*: Run tests in randomized order via `pytest --random-order`.  
  - *Requirement*: Zero flaky or order-dependent failures.
- **F10-B4**: `test_suite_import_cleanliness`  
  - *Boundary*: Run tests with `PYTHONPATH=""`.  
  - *Requirement*: Package imports resolve cleanly from installed virtualenv.
- **F10-B5**: `test_suite_assertion_diagnostic_clarity`  
  - *Boundary*: Intentionally fail an invariant assertion in mock test.  
  - *Requirement*: Error message explicitly prints measured value vs expected tolerance.

### F11: Boundary & Corner Cases (Smoke Test & ONNX Export)
- **F11-B1**: `test_export_onnx_input_tensor_dimension_check`  
  - *Boundary*: Feed tensor of shape `(1, 60)` and `(1, 62)` to exported ONNX model.  
  - *Requirement*: ONNX runtime raises shape validation error, confirming strict 61D contract.
- **F11-B2**: `test_export_normalizer_non_identity_transformation`  
  - *Boundary*: Pass non-zero input to ONNX model with trained normalizer.  
  - *Requirement*: Running mean and variance transform input before first linear layer.
- **F11-B3**: `test_smoke_test_nan_detection_abort`  
  - *Boundary*: Inject NaN into observation tensor during smoke test step.  
  - *Requirement*: `nan_state` guard terminates environment or aborts with explicit error.
- **F11-B4**: `test_export_unfiltered_action_verification`  
  - *Boundary*: Inspect export graph for presence of EMA low-pass filter nodes.  
  - *Requirement*: No low-pass filter present in graph, preserving raw policy actions.
- **F11-B5**: `test_export_zero_runtime_dependencies`  
  - *Boundary*: Load exported ONNX model in minimal Python environment without PyTorch or Warp.  
  - *Requirement*: Model executes with `onnxruntime` alone using standard float32 numpy arrays.

---

## 5. Tier 3: Cross-Feature Interactions

### Pairwise Interaction Matrix

| | F1 (Clearance) | F2 (Landing) | F3 (Contact) | F4 (Recovery) | F5 (Scene) | F6 (Env Cfg) | F7 (Curric) | F8 (Task Reg) | F9 (Runner) | F10 (Tests) | F11 (E2E/Export) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **F1** | — | High | High | Low | High | High | **Critical** | Low | Med | High | Med |
| **F2** | High | — | High | Low | **Critical** | High | Med | Low | Med | High | Med |
| **F3** | High | High | — | Med | High | **Critical** | Med | Low | Med | High | Med |
| **F4** | Low | Low | Med | — | High | High | Med | Low | **Critical** | High | Med |
| **F5** | High | **Critical** | High | High | — | **Critical** | High | Med | **Critical** | High | Med |
| **F6** | High | High | **Critical** | High | **Critical** | — | High | **Critical** | High | High | **Critical** |
| **F7** | **Critical** | Med | Med | Med | High | High | — | Med | Med | High | Med |
| **F8** | Low | Low | Low | Low | Med | **Critical** | Med | — | High | High | **Critical** |
| **F9** | Med | Med | Med | **Critical** | **Critical** | High | Med | High | — | High | Med |
| **F10**| High | High | High | High | High | High | High | High | High | — | High |
| **F11**| Med | Med | Med | Med | Med | **Critical** | Med | **Critical** | Med | High | — |

### Detailed Cross-Feature Interaction Scenarios

#### X1. F1 (Clearance) $\times$ F7 (Curriculum): Dynamic Target Clearance Scaling
- **Interaction**: As the stair curriculum progresses across stages (2.5 cm $\to$ 3.0 cm $\to$ 3.5 cm), the clearance reward target height `target_height` must scale from 0.030 m to 0.045 m.
- **Verification**: Verify that mutating the curriculum stage via the event manager properly updates `env.reward_manager.get_term_cfg("stairs_foot_clearance").params["target_height"]` without breaking the positive weight convention.

#### X2. F2 (Landing Damping) $\times$ F5 (MJCF Scene): Rigid Step Contact Dynamics
- **Interaction**: The standalone scene uses rigid step boxes with `solref="0.02 1.0"` and friction $1.2$. Foot landing damping must prevent high contact force spikes on rigid edges while supporting nominal weight.
- **Verification**: Simulate footfall impact onto `scene_stairs.xml` step geoms in MuJoCo CPU loop; verify that damping reward penalizes high-velocity foot drops without suppressing stance support.

#### X3. F3 (Contact Distribution) $\times$ F6 (Env Cfg): Solver Contact Softening Stability
- **Interaction**: `_soften_terrain_contacts` softens constraint compliance (`solref=[0.04, 1.0]`) to avoid GPU solver NaNs. Contact distribution terms (`stairs_riser_trip_penalty`, `stairs_sole_flat_penalty`) must remain numerically stable and sensitive under softened contact conditions.
- **Verification**: Step an environment batch with softened contact parameters; ensure contact normal and shear force calculations do not diverge or trigger NaN guards.

#### X4. F4 (Recovery Potential) $\times$ F9 (Runner): Real-Time Self-Recovery Trigger
- **Interaction**: When a perturbation causes $|\text{tilt}| > 45^\circ$ during interactive execution in `scripts/infer_stairs.py`, the runner triggers the standup recovery mode. The MDP elevation-invariant potential must guide the robot back upright.
- **Verification**: Inject a lateral impulse to topple the robot in `infer_stairs.py`; verify that recovery logic detects the fall, switches mode, and upright progress drives recovery to a standing pose.

#### X5. F5 (MJCF Scene) $\times$ F9 (Runner): Spawn Presets Coordinate Alignment
- **Interaction**: `scene_stairs.xml` defines 8 keyframe coordinates. `scripts/infer_stairs.py` exposes 5 interactive spawn presets. The presets must map exactly to keyframe positions without spawning the robot inside step geometry.
- **Verification**: Iterate over all spawn presets (`floor`, `stairs_25`, `stairs_30`, `stairs_35`, `summit`); verify initial robot height $z \ge z_{step} + 0.12$ m, producing zero initial penetration contacts.

#### X6. F6 (Env Cfg) $\times$ F8 (Task Registration): Registration & Twin Configuration Parity
- **Interaction**: `tasks/__init__.py` registers `Mjlab-StairsClimb-MicroDuck` and its backlash twin. Both must point to the identical base configuration factory and inherit the same observation and reward structures.
- **Verification**: Compare configuration objects of the base task and backlash twin; verify identical 61D observation layouts, identical reward dictionaries, and matching BAM friction events.

#### X7. F6 (Env Cfg) $\times$ F11 (Export): Baked Normalizer and 61D/14D Shape Preservation
- **Interaction**: The environment's 61D observation space and 14D action space must map 1:1 through `scripts/export.py` into the ONNX deployment model with baked normalizer.
- **Verification**: Export a checkpoint configured with `MicroduckStairsRlCfg`; load resulting ONNX model and assert input node shape `(1, 61)` and output node shape `(1, 14)`.

---

## 6. Tier 4: Real-World Application Scenarios

### Scenario RW-1: Autonomous Full Staircase Multi-Flight Ascent
- **Operational Objective**: Microduck must start on the flat apron, ascend Flight 1 (2.5 cm risers), traverse Landing 1, ascend Flight 2 (3.0 cm risers), traverse Landing 2, ascend Flight 3 (3.5 cm risers), and arrive safely on the Top Summit Platform.
- **Trajectory Breakdown**:
  1. $x \in [0.0, 0.30]$ m: Flat apron walk, steady gait establishment.
  2. $x \in [0.30, 0.66]$ m: Flight 1 ascent (three 2.5 cm steps, total rise +7.5 cm).
  3. $x \in [0.66, 1.02]$ m: Platform 1 horizontal transit at $z = 7.5$ cm.
  4. $x \in [1.02, 1.38]$ m: Flight 2 ascent (three 3.0 cm steps, total rise +9.0 cm).
  5. $x \in [1.38, 1.74]$ m: Platform 2 horizontal transit at $z = 16.5$ cm.
  6. $x \in [1.74, 2.10]$ m: Flight 3 ascent (three 3.5 cm steps, total rise +10.5 cm).
  7. $x \in [2.10, 2.58]$ m: Summit Platform arrival at $z = 27.0$ cm.
- **Success Criteria**:
  - Traversal completed without falling or tripping ($|\text{tilt}| < 30^\circ$).
  - Mean forward velocity $\ge 0.15$ m/s.
  - Zero servo stall events ($|\tau| < 0.96$ Nm).

### Scenario RW-2: Perturbation & Mid-Flight Fall Recovery
- **Operational Objective**: During ascent on Flight 2 (3.0 cm steps), a dynamic perturbation (force impulse of $4.0$ N for $0.1$ s) is injected into the trunk, causing the robot to topple onto the step surface. The policy must autonomously recover and regain standing balance on the stairs.
- **Trajectory Breakdown**:
  1. Robot climbing step 2_2 ($z \approx 13.5$ cm).
  2. Perturbation applied $\implies$ robot rolls onto side/back ($\text{tilt} > 45^\circ$, $z_{rel} \approx 0.04$ m).
  3. Recovery layer triggers: fallen tax charges -0.5/step; upright progress and relative height progress provide positive gradient.
  4. Robot uses beak, knees, and sole contacts to push off step surface, tuck legs, and rise.
  5. Return to upright stance ($\text{tilt} < 25^\circ$, $z_{rel} > 0.09$ m) $\implies$ recovery success bounty fires (+10.0).
- **Success Criteria**:
  - Autonomous recovery within $3.0$ seconds of fall.
  - Robot does not tumble down the staircase during recovery attempt.

### Scenario RW-3: Interactive Keyboard Teleoperation in Standalone MuJoCo Runner
- **Operational Objective**: Operator launches `scripts/infer_stairs.py` with BAM actuator dynamics, uses keyboard controls to steer velocity ($v_x \in [-0.2, 0.5]$ m/s, $\omega_z \in [-1.0, 1.0]$ rad/s) and adjust pitch trim, observing live balance telemetry.
- **Verification Elements**:
  - Control loop runs at rock-steady 50 Hz.
  - Telemetry HUD displays pitch/roll tilt, foot contact forces, and torque limits.
  - Pressing `1` through `5` resets spawn to preset locations.

### Scenario RW-4: Automated CI/CD Regression & ONNX Normalizer Verification
- **Operational Objective**: Pre-commit CI execution verifying zero regressions across existing environments and validating that newly trained stairs policies produce deployable ONNX models with baked normalizers.
- **Pipeline Stages**:
  1. `uv run list-envs` includes `Mjlab-StairsClimb-MicroDuck`.
  2. `PYTHONPATH="" uv run --with pytest pytest tests/` passes 100%.
  3. Smoke test: `uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5` completes with exit code 0 and 0 NaNs.
  4. `uv run scripts/export.py` produces `policy.onnx` with input `(1, 61)` and output `(1, 14)`.

### Scenario RW-5: Sim2Real Hardware Deployment Gate
- **Operational Objective**: Verification that the exported policy matches the physical hardware constraints of the real Microduck robot before loading via `robotctl policy add`.
- **Hardware Invariants Audited**:
  - XL330 voltage limit: $7.4$ V nominal, $0.96$ Nm stall torque.
  - Gear backlash: Policy tested against `-Backlash-` twin with 2.0° passive gear clearance.
  - Action rate smoothness: $\Delta u < 0.25$ rad per 20 ms timestep to avoid servo gear stripping.
  - Zero low-pass filtering: Exported policy runs raw without hidden runtime filtering.

---

## 7. Execution Guide & Test Commands

### Running the E2E Test Suite
```bash
# Run the stairs E2E requirement tests
PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_e2e.py -v

# Run the full regression test suite (CPU invariant tests)
PYTHONPATH="" uv run --with pytest pytest tests/
```

### Running the 5-Iteration Smoke Test
```bash
# Fast 64-environment smoke test (always run before full training runs)
uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

### Running the Standalone Simulation Rehearsal
```bash
# CPU MuJoCo deployment rehearsal with BAM M6 actuators on stairs
uv run scripts/infer_stairs.py --spawn stairs_25
```
