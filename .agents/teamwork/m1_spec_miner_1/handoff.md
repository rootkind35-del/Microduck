# Handoff Report: Milestone M1 Specifications (Stairs MDP Functions & Scene)

**Author**: M1 Spec Miner (`m1_spec_miner_1`)  
**Date**: 2026-10-11  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/m1_spec_miner_1`  
**Report Document**: `/home/dhp01/microduck_rl/.agents/teamwork/m1_spec_miner_1/report.md`  
**Status**: COMPLETE (Hard Handoff)

---

## 1. Observation

1. **`src/mjlab_microduck/tasks/mdp.py` Architecture**:
   - Total line count is 7,687 lines. Lines 1–25 define core module imports (`torch`, `Entity`, `ManagerBasedRlEnv`, `SceneEntityCfg`). Line 219 defines `_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")`.
   - Lines 226–235 define `_servo_joint_ids` to safely access 14 servo joints and bypass `passive_*` joints.
   - The end of `mdp.py` (lines 7466–7686) houses the `Fetch task` functions and `FetchTargetCommandCfg`. Line 7687 is an empty trailing line.
   - Section headers in `mdp.py` use double horizontal bars (e.g. line 2043, 3352, 6869):
     ```python
     # ==============================================================================
     # <Task Name>
     # ==============================================================================
     ```
2. **Contact and Terrain Sensors**:
   - `microduck_velocity_env_cfg.py` lines 220–255 configure `feet_ground_contact` as `ContactSensor` with `primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot")`, `secondary=ContactMatch(mode="body", pattern="terrain")`, `fields=("found", "force")`, `reduce="netforce"`, and `track_air_time=True`.
   - `foot_height_scan` is configured as a `TerrainHeightSensor` with `frame` mapped to sites `left_foot` and `right_foot`. In `mjlab.sensor.terrain_height_sensor`, `sensor.data.heights` returns tensor `[B, 2]`.
   - In `mjlab.sensor.contact_sensor`: `ContactSensor.data.found` has shape `[B, 2]`, `data.force` has shape `[B, 2, 3]`, and `contact_sensor.compute_first_contact(dt)` returns `[B, 2]` bool.
3. **Existing MJCF Scene Models**:
   - `src/mjlab_microduck/robot/microduck/scene.xml` includes `robot_groundcontact.xml`, defines `groundplane` material, light at `[0, 0, 3.5]`, floor plane geom, and 4 keyframes (`INIT`, `STAND`, `SIT`, `FOLD`).
   - `stairs.xml` defines step boxes with `condim="3"`, `friction="1.2 0.01 0.001"`, `solref="0.02 1"`, and tread half-size $s_x = 0.06$.
   - Collision models: `robot_groundcontact.xml` provides collision meshes on battery pack (`np_f970`), shins/knees (`leg`), and head/beak, preventing geometry clipping when toppling on stairs.
4. **Current Test Suite**:
   - Running `PYTHONPATH="" uv run --with pytest pytest -p no:launch_testing tests/` executes 232 test items with 231 passed and 1 skipped in 13.07s (100% pass rate).

---

## 2. Logic Chain

1. **Placement of Custom Stairs MDP Functions**:
   - From Observation 1, `mdp.py` organizes task-specific components sequentially by task family at the end of the file.
   - Therefore, appending the 5 stairs MDP functions starting at line 7688 under the heading `# Stairs Climbing Task — Clearance, Landing Damping, Contact & Recovery Terms` is the exact, zero-regression integration point.
2. **One-Sided Swing Clearance (`stairs_foot_clearance_penalty`)**:
   - On 2.5–3.5 cm stairs, standard `feet_clearance` $|h - h_{target}|$ penalizes foot clearance above $h_{target}$, punishing high stepping and causing toe stubbing on step edges.
   - Computing $\Delta h = \max(0.0, h_{target} - h_{sensor})$ and multiplying by horizontal site velocity $> 0.02$ m/s ensures only clearance deficits during forward swing are penalized, while high swings are free ($0.0$).
3. **Landing Impact Damping (`stairs_foot_landing_damping_penalty`)**:
   - Dynamixel XL330 gear trains have a 288:1 reduction ratio. Impulses on rigid steps damage plastic gear teeth.
   - Gating on `compute_first_contact(dt)` evaluates penalties strictly at touchdown. Clamping contact force above $8.0$ N ($\approx 1\times$ body weight) and normal velocity above $0.08$ m/s penalizes hard slams while allowing normal static weight support.
4. **Riser Collision Anti-Tripping (`stairs_riser_trip_penalty`)**:
   - Step risers are vertical surfaces. Collisions with risers generate horizontal shear force $F_{xy}$ with negligible vertical force $F_z$.
   - On treads, friction satisfies $\|F_{xy}\| \le \mu F_z$. Penalizing $\max(0.0, \|F_{xy}\| - 0.5 F_z)$ directly penalizes riser collisions without penalizing tread traction.
5. **Sole Flat Alignment (`stairs_sole_flat_penalty`)**:
   - Projecting gravity into the foot site frame yields $xy = 0$ when the sole is parallel to horizontal treads. Gating on ground contact ensures airborne feet are free to articulate.
6. **Elevation-Invariant Recovery Potential (`stairs_relative_height_progress`)**:
   - Standard `height_progress` ($z_{trunk} - z_{origin}$) saturates on step 2 because stair risers elevate the robot ($z_{fallen} = 0.14 > 0.115$ m), providing zero gradient to stand up.
   - Measuring relative height $z_{rel} = z_{trunk} - \min(z_{left\_foot}, z_{right\_foot})$ ensures upright standing on any step yields $z_{rel} \approx 0.117$ m, while climbing stairs does not alter $z_{rel}$, rendering recovery rewards unfarmable by locomotion.
7. **Standalone Scene `scene_stairs.xml` Geometry**:
   - Placing 12 solid floor-anchored box geoms ($p_z = s_z = H_{top} / 2$) from $x = 0.30$ to $2.58$ m ensures zero hollow cavities and eliminates contact tunneling.
   - Verified continuity condition $p_x^{(k)} + s_x^{(k)} = p_x^{(k+1)} - s_x^{(k+1)}$ guarantees zero gaps and zero overlaps across all 3 flights and 3 landing platforms.

---

## 3. Caveats

1. **Air Time Sensor Prerequisite**: `stairs_foot_landing_damping_penalty` calls `contact_sensor.compute_first_contact(dt)`. In `microduck_stairs_env_cfg.py`, the `feet_ground_contact` sensor must maintain `track_air_time=True` (matching the base configuration in `microduck_velocity_env_cfg.py`).
2. **TerrainHeightSensor Placement**: `stairs_foot_clearance_penalty` requires `height_sensor_name="foot_height_scan"`. The environment configuration must include `foot_height_scan` attached to sites `left_foot` and `right_foot`.
3. **No Caveats on Geometry**: Step riser heights ($0.025, 0.030, 0.035$ m) and gapless boundaries are mathematically closed and proven.

---

## 4. Conclusion

Milestone M1 specifications are completely extracted, verified, and documented in `/home/dhp01/microduck_rl/.agents/teamwork/m1_spec_miner_1/report.md`:
1. **`src/mjlab_microduck/tasks/mdp.py`**:
   - Insertion point: Line 7688.
   - 5 custom MDP functions ready for drop-in: `stairs_foot_clearance_penalty`, `stairs_foot_landing_damping_penalty`, `stairs_riser_trip_penalty`, `stairs_sole_flat_penalty`, and `stairs_relative_height_progress`.
   - All penalties strictly follow the self-negating convention ($\le 0.0 \implies w > 0$).
2. **`src/mjlab_microduck/robot/microduck/scene_stairs.xml`**:
   - Complete 298-line MJCF document including `robot_groundcontact.xml`, visual lighting, 7 color-coded materials, 12 gapless solid step boxes across 3 riser stages (2.5, 3.0, 3.5 cm), 3 navigation site beacons, and 8 keyframes.
3. **`tests/test_scene_stairs.py`**:
   - Concrete test specifications verifying XML compilation, 14 DOFs, geometry riser elevations, gapless continuity, and keyframe dimensions.

---

## 5. Verification Method

To verify the specifications and deliverables:
1. **Inspect Report Document**:
   ```bash
   cat /home/dhp01/microduck_rl/.agents/teamwork/m1_spec_miner_1/report.md
   ```
2. **Test Baseline Pass Rate**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest -p no:launch_testing tests/
   ```
   (Must pass 231 tests).
3. **Verify Insertion Compatibility**:
   Ensure `src/mjlab_microduck/tasks/mdp.py` line 7687 is the current end-of-file and functions can be appended without syntax collisions.
4. **Invalidation Conditions**:
   - Any modification to existing lines 1–7687 in `mdp.py`.
   - Changing the `feet_ground_contact` sensor contract (`track_air_time=False`).
   - Using `robot_walk.xml` instead of `robot_groundcontact.xml` in `scene_stairs.xml`.
