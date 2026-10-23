# Handoff Report: Standalone Simulation Scene & Test Design (`scene_stairs.xml`)

**From**: M1 Scene Explorer (`m1_explorer_scene_1`)  
**To**: Orchestrator (`f3cd8a4f-6d90-4952-bb58-a5be7069a95c`) / M1 Implementation Builder  
**Date**: 2026-10-11  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1`  

---

## 1. Observation

1. **Existing Simulation Scenes**:
   - `src/mjlab_microduck/robot/microduck/scene.xml`: Includes `robot_groundcontact.xml` with groundplane checker material, floor plane at $z=0$, and keyframes `INIT`, `STAND`, `SIT`, `FOLD`.
   - `src/mjlab_microduck/robot/microduck/stairs.xml`: Legacy 4-step mini-stair with 1.5 cm risers (elevations 1.5, 3.0, 4.5, 6.0 cm).
   - `src/mjlab_microduck/robot/microduck/robot_groundcontact.xml`: Contains curated collision geoms for `np_f970` battery, `leg` (shins/knees), `bottom_head_shell`, and `jaw`, while `robot_walk.xml` omits them.
2. **Scene Compilation & MuJoCo Parsing**:
   - Compiling the proposed XML via `mujoco.MjModel.from_xml_string(xml_content)` with working directory set to `src/mjlab_microduck/robot/microduck/` yields:
     ```
     ngeom: 94 nbody: 16 njnt: 15 nkey: 8 nu: 14 nq: 21 nv: 20
     ```
   - When loaded via `mujoco.MjModel.from_xml_path()`, MuJoCo resolves relative includes `<include file="robot_groundcontact.xml" />` relative to the directory containing the XML file. Placing `scene_stairs.xml` in `src/mjlab_microduck/robot/microduck/` resolves all dependencies with zero errors.
3. **Obstacle Geometries & Riser Heights**:
   - 12 consecutive obstacle boxes along $+X$ axis from $x = 0.30$ m to $x = 2.58$ m:
     - Flight 1: `stair_1_1` ($z_{top}=0.025$m, $\Delta z = +2.5$cm), `stair_1_2` ($z_{top}=0.050$m, $\Delta z = +2.5$cm), `stair_1_3` ($z_{top}=0.075$m, $\Delta z = +2.5$cm).
     - Landing 1: `landing_1` ($x \in [0.66, 1.02]$m, $z_{top}=0.075$m, $\Delta z = 0.0$cm).
     - Flight 2: `stair_2_1` ($z_{top}=0.105$m, $\Delta z = +3.0$cm), `stair_2_2` ($z_{top}=0.135$m, $\Delta z = +3.0$cm), `stair_2_3` ($z_{top}=0.165$m, $\Delta z = +3.0$cm).
     - Landing 2: `landing_2` ($x \in [1.38, 1.74]$m, $z_{top}=0.165$m, $\Delta z = 0.0$cm).
     - Flight 3: `stair_3_1` ($z_{top}=0.200$m, $\Delta z = +3.5$cm), `stair_3_2` ($z_{top}=0.235$m, $\Delta z = +3.5$cm), `stair_3_3` ($z_{top}=0.270$m, $\Delta z = +3.5$cm).
     - Summit: `landing_summit` ($x \in [2.10, 2.58]$m, $z_{top}=0.270$m, $\Delta z = 0.0$cm).
   - Gapless alignment: $|x_{min}^{(k+1)} - x_{max}^{(k)}| < 10^{-6}$ m for all $k$.
   - Floor anchoring: $p_z - s_z = 0.0000$ m for all 12 boxes.
   - Contact dynamics: `condim=3`, tangential friction $1.2$, `solref="0.02 1.0"`, `solimp="0.9 0.95 0.001 0.5 2"`.
4. **Keyframe Settling Stability**:
   - Forward kinematics and 50-step physics simulation with built-in PD hold produced:
     ```
     STAND           : init_z=0.125, final_z=0.115, pitch=+1.1deg, roll=-0.0deg
     SPAWN_STAIRS_25 : init_z=0.175, final_z=0.165, pitch=+1.1deg, roll=+0.0deg
     SPAWN_LANDING_1 : init_z=0.200, final_z=0.190, pitch=+1.1deg, roll=+0.0deg
     SPAWN_STAIRS_30 : init_z=0.260, final_z=0.250, pitch=+1.1deg, roll=+0.0deg
     SPAWN_LANDING_2 : init_z=0.290, final_z=0.280, pitch=+1.1deg, roll=+0.0deg
     SPAWN_STAIRS_35 : init_z=0.360, final_z=0.350, pitch=+1.1deg, roll=+0.0deg
     SPAWN_SUMMIT    : init_z=0.395, final_z=0.385, pitch=+1.1deg, roll=+0.0deg
     ```
5. **Test Suite Execution**:
   - Executing the proposed 10-test suite against the validated scene produced 10 passing tests:
     `PASS: test_scene_stairs_compiles`, `PASS: test_scene_stairs_robot_model`, `PASS: test_all_stair_geometries_exist`, `PASS: test_stair_riser_heights`, `PASS: test_stair_gapless_continuity`, `PASS: test_stair_solid_grounding`, `PASS: test_stair_contact_friction_parameters`, `PASS: test_stair_width`, `PASS: test_all_keyframes_exist_and_valid`, `PASS: test_keyframe_dynamic_settle_stability`.

---

## 2. Logic Chain

1. **Model Selection**: From Observation 1, climbing stairs and recovering from falls involves ground impacts on the torso, shins, and battery. `robot_groundcontact.xml` provides collision meshes on all these parts, preventing mesh clipping and ensuring physically faithful contact forces.
2. **XML File Colocation**: From Observation 2, `robot_groundcontact.xml` is located in `src/mjlab_microduck/robot/microduck/`. When `scene_stairs.xml` is placed in this exact folder, the standard `<include file="robot_groundcontact.xml" />` directive resolves natively without requiring XML parser path modifications.
3. **Geometric Soundness**: From Observation 3, the consecutive step half-sizes and positions satisfy both the stair height progression requirement ($2.5 \rightarrow 3.0 \rightarrow 3.5$ cm) and the mathematical gapless condition ($x_{max}^{(k)} = x_{min}^{(k+1)}$). Solid floor grounding ($z_{bottom} = 0.0$) ensures contact solver stability without tunneling through thin shells.
4. **Physical Equilibrium**: From Observation 4, applying the `STAND2` joint configuration with root $z = z_{surface} + 0.125$ m places the robot precisely at its nominal standing height. Simulating 50 steps confirms that each spawn keyframe settles stably into contact equilibrium ($\le 1.1^\circ$ pitch, $0.0^\circ$ roll), fulfilling the stability invariant from `AGENTS.md`.
5. **Test Coverage**: From Observation 5, the 10 test functions provide comprehensive invariant locking over compilation, kinematics, geometry, friction parameters, keyframes, and dynamic settling.

---

## 3. Caveats

1. **Backlash Robot Variant**: `scene_stairs.xml` includes `robot_groundcontact.xml` (the un-backlashed model). For backlash A/B comparisons in M2, a mirrored `scene_stairs_backlash.xml` can be generated using `add_backlash.py` or by replacing the include with `robot_groundcontact_backlash.xml`.
2. **Dynamic Traversal**: The keyframe settling stability test confirms static equilibrium hold; dynamic traversal across flights depends on PPO policy actions trained in M2 and verified in M3/M4.

---

## 4. Conclusion

`scene_stairs.xml` is 100% schema-valid, geometrically precise, and dynamically stable. The complete, drop-in replacement artifacts are prepared:
- XML Scene file: `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1/proposed_scene_stairs.xml`
  (to be written to `src/mjlab_microduck/robot/microduck/scene_stairs.xml`)
- Unit Test file: `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1/proposed_test_scene_stairs.py`
  (to be written to `tests/test_scene_stairs.py`)

---

## 5. Verification Method

Once copied to their target locations, verify independently via:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py -v
```
Expected result: 10 passed in $< 1.0$ s.

Invalidation conditions:
- Any step riser $\Delta z$ differing from $0.025$, $0.030$, or $0.035$ m by more than $10^{-5}$ m.
- Any discontinuity gap along $X$ between $x=0.30$ m and $x=2.58$ m.
- Any spawn keyframe toppling ($|pitch| \ge 10^\circ$ or $|roll| \ge 10^\circ$) during the 50-step settle test.
