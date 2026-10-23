# Handoff Report: High-Stairs MDP Functions Validation & Test Design

**Agent**: M1 MDP Explorer (`m1_explorer_mdp_1`)  
**Target Milestone**: M1 — Architecture, Assets & Base MDP Verification  
**Date**: 2026-10-11  
**Status**: Hard Handoff (Complete)  

---

## 1. Observation

1. **Codebase Baseline**:
   - `src/mjlab_microduck/tasks/mdp.py` contains 7,687 lines and 234 functions. Grep for `stairs` confirmed 0 existing occurrences.
   - Microduck robot collision models (`robot_allcollisions.xml`) define 7 sites: `imu` (0), `left_foot` (1), `head_camera` (2), `tof` (3), `head_imu` (4), `mouth_tip` (5), and `right_foot` (6).
2. **Sensor Infrastructure**:
   - In `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` (lines 225–255), `feet_ground_cfg` has `primary=("left_foot_collision", "right_foot_collision")`, `fields=("found", "force")`, `reduce="netforce"`, and `track_air_time=True`.
   - `foot_height_scan` is a `TerrainHeightSensorCfg` sampling vertical height for the 2 feet sites.
   - `TerrainHeightSensor.data.heights` has shape `[B, 2]`.
   - `ContactSensor.data.force` has shape `[B, 2, 3]`.
   - `ContactSensor.compute_first_contact` requires `track_air_time=True` and returns shape `[B, 2]` bool.
3. **Multi-Site Broadcast Bug**:
   - In Survey 2 report (line 248), `asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG` was proposed.
   - Calling `SceneEntityCfg("robot").site_ids` produces `slice(None)`, indexing all 7 sites: `asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]` returns shape `(B, 7, 2)`.
   - Attempting `clearance_deficit * moving` produces verbatim error:
     `RuntimeError: The size of tensor a (2) must match the size of tensor b (7) at non-singleton dimension 1`.
4. **Validation Test Run**:
   - Running the unit test suite on CPU via:
     `PYTHONPATH="" uv run --with pytest pytest -p no:launch_testing -v /tmp/test_stairs_mdp_verify.py`
     returned: `11 passed in 8.99s` (100% pass rate).
   - Stress test over 1,000 randomized environments confirmed all 4 penalties return $\le 0.0$ strictly.
   - `stairs_relative_height_progress` produces exactly `0.0` at steps 0 and 1 (`episode_length_buf <= 1`), exactly `0.0` when climbing stairs with constant relative posture, and produces 0 NaNs.

---

## 2. Logic Chain

1. **Device & Dtype Logic**:
   - Observation 2 confirms that sensor outputs and asset data arrive as `torch.float32` on `env.device`.
   - Intermediate calculations (e.g. `clearance_deficit`, `excess_force`, `per_foot`) explicitly allocate tensors on `env.device` with `dtype=torch.float32`.
   - Condition masks (`cmd_active`, `first_contact`, `in_contact`) are explicitly converted with `.float()`, preserving `torch.float32` and dimension matching without implicit dtype casting.
2. **Dimension Matching & Site Isolation Logic**:
   - From Observation 3, `_DEFAULT_ASSET_CFG` resolves to 7 sites, causing dimension mismatch with 2-primary sensors.
   - Defining `_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` explicitly isolates site indices `[1, 6]`.
   - Slicing `asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]` yields shape `(B, 2, 3)`, which matches `foot_height_scan` `(B, 2)` and `feet_ground_contact` `(B, 2)`.
   - Summing over feet `dim=1` produces consistent shape `(B,) = (env.num_envs,)` across all 5 functions.
3. **Self-Negating Mathematical Proof Logic**:
   - `stairs_foot_clearance_penalty`: `torch.clamp(target - h, min=0.0) >= 0.0` and `torch.clamp(vel - thresh, min=0.0) >= 0.0`. Their product and sum are $\ge 0.0$. Multiplying by command gate $\in \{0.0, 1.0\}$ keeps cost $\ge 0.0$. Returning `-cost` is strictly $\le 0.0$.
   - `stairs_foot_landing_damping_penalty`: `excess_force >= 0.0` and `vel_weight * down_vel**2 >= 0.0`. Gated by `first_contact.float()`, sum is $\ge 0.0$. Returning `-sum` is strictly $\le 0.0$.
   - `stairs_riser_trip_penalty`: `f_xy >= 0.0` and `f_z >= 0.0`. `riser_excess = clamp(f_xy - mu * f_z, min=0.0) >= 0.0`. Gated by `in_contact`, sum is $\ge 0.0$. Returning `-sum` is strictly $\le 0.0$.
   - `stairs_sole_flat_penalty`: `proj[:, :2]**2 >= 0.0`. Sum of squared tilts is $\ge 0.0$. Gated by contact, sum is $\ge 0.0$. Returning `-sum` is strictly $\le 0.0$.
   - Therefore, all 4 penalty functions obey the Microduck invariant ($P \le 0.0$) and must be weighted with positive weights ($w > 0$).
4. **Stairs Relative Height Logic**:
   - Standard `height_progress` uses $z_{trunk} - z_{origin}$, which saturates above step 1 ($z_{trunk} > 0.115$) and rewards stair ascent as recovery.
   - `stairs_relative_height_progress` uses $z_{rel} = z_{trunk} - \min(z_L, z_R)$.
   - Ascending stairs increases $z_{trunk}$ and $z_{feet}$ simultaneously, keeping $z_{rel} \approx 0.117$ m constant. Thus $\Delta \Phi = 0.0$, making it unfarmable during stair climbing.
   - Extending legs from crouch/prone increases $z_{rel}$ from $0.04 \to 0.115$ m, generating positive potential shaping $\Delta \Phi > 0$.
   - Masking `episode_length_buf <= 1` resets `_stairs_rel_height_pot_prev` to current potential, preventing initial spawn transients.

---

## 3. Caveats

- **Contact Sensor Configuration**: The landing damping function relies on `contact_sensor.compute_first_contact(dt=env.step_dt)`, which strictly requires `track_air_time=True` on `feet_ground_cfg`. This is already true in `microduck_velocity_env_cfg.py` and must be preserved in `microduck_stairs_env_cfg.py`.
- **Sensors reduced to netforce**: `stairs_riser_trip_penalty` expects global frame forces, which is guaranteed when `reduce="netforce"` is configured on `ContactSensorCfg`.

---

## 4. Conclusion

The 5 high-stairs MDP functions have been mathematically proven, tensor-verified, hardened against multi-site broadcast bugs, and confirmed 100% NaN-safe.
Ready-to-merge implementations and unit tests have been authored:
- Code addition: `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/proposed_stairs_mdp_additions.py`
- Test suite: `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/proposed_test_stairs_mdp.py`
- Comprehensive Report: `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/report.md`

---

## 5. Verification Method

Independently verify by running the test suite on CPU:
```bash
PYTHONPATH="" uv run --with pytest pytest -p no:launch_testing -v /home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/proposed_test_stairs_mdp.py
```
Expected output:
- 11 passed tests in < 10 seconds.
- 0 failures, 0 NaNs, 0 shape mismatch errors.
