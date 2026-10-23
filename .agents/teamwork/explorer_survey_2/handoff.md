# Handoff Report — Explorer Survey 2

**Task**: Investigation of `src/mjlab_microduck/tasks/mdp.py` and Design of High-Stairs MDP Terms  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2`  
**Handoff Type**: Hard (Task Complete)  

---

## 1. Observation
1. **Existing Codebase Structure in `src/mjlab_microduck/tasks/mdp.py`**:
   - `mdp.py` contains 7,686 lines and 234 top-level functions.
   - Lines 25–214 contain Patches 1–7 (reward NaN sanitization, advantage clipping, passive joint filtering in ONNX export, `MICRODUCK_WARM_START=1` curriculum reset, and rsl_rl buffer sanitization).
   - Lines 226–268 provide cached helpers for servo joint index and state resolution: `_servo_joint_ids`, `_servo_joint_pos`, `_servo_joint_vel`, `_servo_default_joint_pos`, and `_env_origin_z`.
   - Lines 639–1050 define locomotion and balance shaping terms: `feet_air_time_upright`, `upright_progress`, `height_progress`, `fallen_state_penalty`, `recovery_success`, `servo_stall_penalty`, `servo_acc_spike_penalty`, and `action_rate_l2_fallen_scaled`.
2. **Standard mjlab Foot Clearance & Landing**:
   - mjlab's `feet_clearance` computes `cost = sum(|foot_height - target_height| * vel_norm)` which penalizes lifting above `target_height`.
   - mjlab's `soft_landing` penalizes total contact force magnitude without an impact threshold and does not account for downward normal touchdown velocity $v_z$.
3. **Height Progress Failure on Stairs**:
   - In `mdp.py:704`, `height_progress` computes `z = asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]` and clamps to `ceiling = 0.115`.
   - On stairs, step risers accumulate elevation. By step 2 or 3 ($z_{riser} = 0.05$–$0.09$ m), a fallen prone robot already has trunk $z > 0.115$ m relative to `env_origin`. Standard `height_progress` saturates and provides zero gradient to stand up.
4. **Robot Model Invariants**:
   - Plain models (`robot_walk.xml`, `robot_allcollisions.xml`) have 14 actuated servos. Backlash models (`robot_allcollisions_backlash.xml`) interleave 14 `passive_*_backlash` hinges.
   - Foot collision geoms (`left_foot_collision`, `right_foot_collision`), ankle bodies (`ankle_left`, `ankle_right`), and foot sites (`left_foot`, `right_foot`) are identical across all variants.
5. **Test Suite Status**:
   - Existing test suite (`PYTHONPATH="" uv run --with pytest pytest -p no:launch_testing tests/`) passes 100%: 231 passed, 1 skipped in 13.07s.

---

## 2. Logic Chain
1. *From Observation 2*: On stairs with 2.5 cm – 3.5 cm risers (~15% of robot height), the robot must execute high-stepping swing trajectories. Symmetrical error in `feet_clearance` punishes lifting above target. Therefore, `stairs_foot_clearance_penalty` must use a one-sided deficit $\max(0, h_{target} - h_{foot})$ so high clearance is free while low scuffing during forward swing is penalized.
2. *From Observation 2*: XL330 gearboxes have plastic gear trains (288:1 ratio) vulnerable to impact backdriving. Penalizing impact force spikes above body weight ($F > 8$ N) and downward vertical velocity ($v_z < -0.08$ m/s) at initial contact (`first_contact`) cushions footfalls on rigid step treads.
3. *From Geometry of Stairs*: Tread contact has vertical normal $n_z \approx 1$ and $F_z > 0$, while riser contact has horizontal normal $n_z \approx 0$ and high horizontal shear force $\|F_{xy}\|$. By penalizing horizontal contact force exceeding the tread friction cone ($\|F_{xy}\| - 0.5 F_z > 0$) or riser normal alignment, toe-stubbing and riser collisions are penalized.
4. *From Observation 3*: To make recovery potential shaping elevation-invariant across arbitrary stair steps, height progress must be evaluated relative to the lowest foot site: $z_{rel} = z_{trunk} - \min(z_{left\_foot}, z_{right\_foot})$. On any step, standing corresponds to $z_{rel} \approx 0.115$ m and prone to $z_{rel} \approx 0.04$ m. Climbing stairs does not alter $z_{rel}$ and cannot farm recovery reward.
5. *From Observation 4 & Repository Invariants*: All new MDP terms must index joints through `_servo_joint_ids` to maintain identical behavior across plain and backlash robot models. All penalties returning $\le 0.0$ must take positive weights in config files.

---

## 3. Caveats
- Raycast height readings from `TerrainHeightSensor` (`foot_height_scan`) require rays aligned with gravity/yaw. If the robot rolls onto its side, ray angles tilt; the safe fallback in `height_safe` handles NaN/miss conditions.
- Stance-relative height $z_{trunk} - \min(z_{feet})$ assumes at least one foot is near the ground. If the robot is airborne during a violent topple, $z_{rel}$ tracks relative limb extension, which still penalizes crouching and rewards extending limbs to catch the ground.
- Stair height curriculum staging ($2.5 \to 3.0 \to 3.5$ cm) must be mirrored in the clearance target parameter $h_{target}$ ($0.032 \to 0.038 \to 0.045$ m) to maintain appropriate gradient pressure.

---

## 4. Conclusion
1. Complete mathematical designs and implementation-ready code templates have been created for:
   - `stairs_foot_clearance_penalty`: One-sided swing clearance deficit penalty.
   - `stairs_foot_landing_damping_penalty`: Excess contact force and vertical touchdown velocity damping.
   - `stairs_riser_trip_penalty`: Shear/normal-based riser collision and tripping penalty.
   - `stairs_sole_flat_penalty`: Stance-gated horizontal sole alignment on treads.
   - `stairs_relative_height_progress`: Elevation-invariant relative height recovery shaping.
2. Nine existing terms in `mdp.py` (`upright_progress`, `feet_air_time_upright`, `action_rate_l2_fallen_scaled`, `joint_torque_rate_l2_fallen_scaled`, `servo_acc_spike_penalty`, `servo_stall_penalty`, `body_impact_cost`, `trunk_vertical_accel_penalty`, `head_pose_bias_penalty`) have been identified and mapped for direct reuse.
3. The comprehensive specification has been written to `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2/report.md`.

---

## 5. Verification Method
1. Inspect the full report:
   ```bash
   cat /home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_2/report.md
   ```
2. Run the test suite:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest -p no:launch_testing tests/
   ```
3. When implemented, run the standard 5-iteration smoke test:
   ```bash
   uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
   ```
