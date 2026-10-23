# Comprehensive Survey Report: Standalone Simulation Scene & Inference Runner for Stairs Navigation

**Agent**: Explorer Survey 3  
**Date**: 2026-10-11  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3`  
**Reference Directives**: `AGENTS.md`, `ORIGINAL_REQUEST.md`  

---

## 1. Executive Summary

This report establishes the complete architectural and physical design for the standalone simulation scene `src/mjlab_microduck/robot/microduck/scene_stairs.xml`, the dedicated deployment rehearsal runner `scripts/infer_stairs.py`, and the test verification suite required for the **Microduck Stairs Climbing & Self-Recovery** system (`Mjlab-StairsClimb-MicroDuck`).

### Key Findings & Architectural Decisions
1. **Scene Geometry & Staging**:
   Microduck is a ~25 cm biped (~800 g) with ~12.5 cm nominal standing hip height and ~7.5 cm sole length. Climbing steps of 2.5 cm, 3.0 cm, and 3.5 cm risers (up to 28% of hip height) requires solid, floor-anchored, gapless box geoms along the primary travel axis (+X) with intermediate landing platforms and color-coded materials.
2. **Collision Model Family**:
   The scene must include `robot_groundcontact.xml` (not `robot_walk.xml`). Standalone stairs navigation involves aggressive toe clearances, possible shin-to-riser impacts, and automatic self-recovery from falls. `robot_groundcontact.xml` provides collision meshes on the battery pack (`np_f970`), shins/knees (`leg`), head shells, and beak/jaw (`jaw`, `bottom_head_shell`), preventing the robot from clipping through the steps when toppling or standing up.
3. **Actuator & Physics Rehearsal Invariants**:
   Rehearsal must use the **BAM M6** actuator model with voltage control and load-dependent friction budget (`kt=0.0384`, `R=6.2`, `kp_fw=200.0`, nominal voltage 7.4V, voltage sag gain 0.1 V/Nm), matching the GPU Warp training dynamics. Actuators operate at a 50 Hz control rate with decimation 4 over a 200 Hz (0.005 s) MuJoCo physics timestep.
4. **Interactive Telemetry & Runner Features**:
   `scripts/infer_stairs.py` features 5 interactive spawn presets (flat floor, 2.5 cm flight, 3.0 cm flight, 3.5 cm flight, top summit platform), real-time keyboard velocity & pitch-trim controls, fall detection (`|tilt| > 45°`), standup self-recovery trigger, and real-time balance telemetry (pitch/roll tilt, foot normal contact force per step, knee/ankle torque saturation relative to the XL330 0.96 Nm limit).
5. **Testing & Invariant Verification**:
   The test suite must verify XML compilation, gapless step placement, step riser elevations (exact 2.5, 3.0, 3.5 cm $\pm 0.0001$ m), contact friction $\ge 1.0$, BAM actuator conversion, 61D observation shape, and reward weight signs. Tests run on CPU in seconds with `PYTHONPATH="" uv run --with pytest pytest tests/`.

---

## 2. Investigation of Existing MJCF Scenes

### 2.1 File Catalog & Architecture

Existing scenes in `src/mjlab_microduck/robot/microduck/`:
- `scene.xml` / `scene_walk.xml`: Flat plane ground (`type="plane"`), checker groundplane material, 4 keyframes (`INIT`, `STAND`, `SIT`, `FOLD`). `scene.xml` includes `robot_groundcontact.xml`, while `scene_walk.xml` includes `robot_walk.xml`.
- `scene_interactive.xml`: Includes `robot_groundcontact.xml`, `pick_object.xml`, and the legacy `stairs.xml`.
- `scene_rough.xml`: Includes `robot_groundcontact.xml`. Contains 3 procedural terrain sections directly in `<worldbody>`:
  1. Cobblestone field (49 box geoms, heights 0.4–1.0 cm, size $0.06 \times 0.06$ m).
  2. Mini-stairs along $y = 0.45$ m: 4 steps with 4 mm, 8 mm, 16 mm, 24 mm elevations.
  3. Slopes along $y = -0.45$ m: 4° ramp (`euler="0 -0.0698 0"`).
- `stairs.xml` (legacy mini-stair): 4 steps with 1.5 cm risers (elevations 1.5 cm, 3.0 cm, 4.5 cm, 6.0 cm), tread depth 12 cm (`size="0.06 0.25 <sz>"`), top platform at $x = 0.94$ m, target site at $x = 0.95, z = 0.07$ m.

### 2.2 Collision Families Comparison: `robot_walk.xml` vs `robot_groundcontact.xml`

A direct inspection of the collision geoms reveals the fundamental difference between the models:

| Component | `robot_walk.xml` | `robot_groundcontact.xml` | Physical Role on Stairs |
|---|---|---|---|
| **Soles** (`sole_left`, `sole_right`) | `class="collision"` | `class="collision"` | Primary support contact on step treads |
| **Battery Pack** (`np_f970`) | visual only | `class="collision"` | Protects rear CoM; contacts steps during backward fall |
| **Hips** (`hip_l`) | visual only | `class="collision"` | Prevents pelvic clipping during lateral roll |
| **Shins / Knees** (`leg`) | `self_collision_only` | `class="collision"` | **Critical**: contacts step risers when toe misses tread |
| **Trunk Shells** (`trunk_base`) | visual only | `class="collision"` | Torso protection against step edges |
| **Head Shells** (`top_head_shell`) | visual only | `class="collision"` | Head impact detection; prevents floor clipping |
| **Beak & Jaw** (`jaw`, `bottom_head_shell`) | visual only | `class="collision"` | Front fall contact; tripod posture during standup recovery |

**Conclusion**: For stairs climbing and self-recovery from toppling, `scene_stairs.xml` MUST include `robot_groundcontact.xml`. If `robot_walk.xml` were used, shin impacts on 2.5–3.5 cm step edges would pass through geometrically, producing unphysical dynamics and destroying sim2real transfer.

### 2.3 Contact Dynamics & Solver Parameters

In `scene_rough.xml` and `stairs.xml`:
- `condim="3"`: Enables 3D frictional contact (1 normal force + 2 orthogonal friction forces).
- `friction="1.2 0.01 0.001"`:
  - Tangential sliding friction: $1.2$ (essential for bipedal grip on steps without slip).
  - Torsional friction: $0.01$.
  - Rolling friction: $0.001$.
- `solref="0.02 1.0"`: Solver constraint reference with time constant $\tau = 0.02$ s and damping ratio $d = 1.0$. Prevents step elasticity/bounce while guaranteeing fast constraint stabilization.
- `solimp="0.9 0.95 0.001 0.5 2"`: High constraint impedance, ensuring sharp, rigid step corners without spongy penetration.

---

## 3. Architectural Design of `scene_stairs.xml`

### 3.1 Geometric Layout & Coordinate Strategy

The obstacle course runs continuously along the +X direction from the flat spawning apron ($x = 0$) through three distinct stair flights and intermediate landing platforms to a summit landing.

```
Elevation (Z)
  ^
  |                                                  [Top Summit Platform] (Z=27.0cm)
  |                                                +-----------------------+
  |                                        Step 3_3|
  |                                Step 3_2+-------+
  |                        Step 3_1+-------+
  |        [Platform 2]+-----------+ (Flight 3: 3.5cm risers)
  |            (Z=16.5cm)
  |            +-------+
  |    Step 2_3|
  |Step 2_2----+
  |Step 2_1----+ (Flight 2: 3.0cm risers)
  |    +-------+
  |Platform 1 (Z=7.5cm)
  |    +-------+
  |Step 1_3
  |Step 1_2
  |Step 1_1 (Flight 1: 2.5cm risers)
  |----+
  +----------------------------------------------------------------------------> X (m)
  0.0  0.30    0.66    1.02    1.38    1.74    2.10                    2.58
```

#### Key Geometric Parameters:
- **Tread depth ($L_{tread}$)**: $0.12$ m ($12$ cm) $\rightarrow$ MuJoCo half-size $s_x = 0.06$ m. Microduck sole is ~7.5 cm, leaving ~4.5 cm margin.
- **Stair width ($W_{stair}$)**: $0.50$ m ($50$ cm) $\rightarrow$ MuJoCo half-size $s_y = 0.25$ m. Robot stance width is ~9 cm, giving ~20 cm lateral margin on each side.
- **Intermediate Landing Platforms**: Length $0.36$ m ($36$ cm, 3 steps long) $\rightarrow$ half-size $s_x = 0.18$ m. Allows dynamic gait stabilization before the next flight.
- **Summit Landing Platform**: Length $0.48$ m ($48$ cm) $\rightarrow$ half-size $s_x = 0.24$ m. Generous resting and turnaround area.
- **Solid Floor-Anchored Construction**: Each step box extends from $z = 0$ up to its top elevation $H_{top}$.
  $$\text{Box half-height } s_z = \frac{H_{top}}{2}, \quad \text{Box center } p_z = \frac{H_{top}}{2}$$
  This avoids hollow undersides, preventing contact tunneling.

### 3.2 Precise Specification Table for All Geometries

| Geom Name | Type | $x_{min}$ (m) | $x_{max}$ (m) | Center $p_x$ (m) | Half $s_x$ (m) | Top $H_{top}$ (m) | Half $s_z$ (m) | Center $p_z$ (m) | Riser $\Delta z$ | Material |
|---|---|---|---|---|---|---|---|---|---|---|
| `floor` | plane | $-\infty$ | $+\infty$ | $0.00$ | $0.00$ | $0.000$ | $0.050$ | $0.000$ | — | `groundplane` |
| **Flight 1 (2.5 cm)** | | | | | | | | | | |
| `stair_1_1` | box | $0.30$ | $0.42$ | $0.36$ | $0.06$ | $0.025$ | $0.0125$ | $0.0125$ | $+2.5$ cm | `mat_stair_25` |
| `stair_1_2` | box | $0.42$ | $0.54$ | $0.48$ | $0.06$ | $0.050$ | $0.0250$ | $0.0250$ | $+2.5$ cm | `mat_stair_25` |
| `stair_1_3` | box | $0.54$ | $0.66$ | $0.60$ | $0.06$ | $0.075$ | $0.0375$ | $0.0375$ | $+2.5$ cm | `mat_stair_25` |
| `landing_1` | box | $0.66$ | $1.02$ | $0.84$ | $0.18$ | $0.075$ | $0.0375$ | $0.0375$ | $0.0$ cm | `mat_landing_1` |
| **Flight 2 (3.0 cm)** | | | | | | | | | | |
| `stair_2_1` | box | $1.02$ | $1.14$ | $1.08$ | $0.06$ | $0.105$ | $0.0525$ | $0.0525$ | $+3.0$ cm | `mat_stair_30` |
| `stair_2_2` | box | $1.14$ | $1.26$ | $1.20$ | $0.06$ | $0.135$ | $0.0675$ | $0.0675$ | $+3.0$ cm | `mat_stair_30` |
| `stair_2_3` | box | $1.26$ | $1.38$ | $1.32$ | $0.06$ | $0.165$ | $0.0825$ | $0.0825$ | $+3.0$ cm | `mat_stair_30` |
| `landing_2` | box | $1.38$ | $1.74$ | $1.56$ | $0.18$ | $0.165$ | $0.0825$ | $0.0825$ | $0.0$ cm | `mat_landing_2` |
| **Flight 3 (3.5 cm)** | | | | | | | | | | |
| `stair_3_1` | box | $1.74$ | $1.86$ | $1.80$ | $0.06$ | $0.200$ | $0.1000$ | $0.1000$ | $+3.5$ cm | `mat_stair_35` |
| `stair_3_2` | box | $1.86$ | $1.98$ | $1.92$ | $0.06$ | $0.235$ | $0.1175$ | $0.1175$ | $+3.5$ cm | `mat_stair_35` |
| `stair_3_3` | box | $1.98$ | $2.10$ | $2.04$ | $0.06$ | $0.270$ | $0.1350$ | $0.1350$ | $+3.5$ cm | `mat_stair_35` |
| `landing_summit` | box | $2.10$ | $2.58$ | $2.34$ | $0.24$ | $0.270$ | $0.1350$ | $0.1350$ | $0.0$ cm | `mat_summit` |

*Note*: For all boxes, half-width $s_y = 0.25$ m, $p_y = 0.0$ m, `condim="3"`, `friction="1.2 0.01 0.001"`, `solref="0.02 1.0"`, `solimp="0.9 0.95 0.001 0.5 2"`.

### 3.3 Visual Color Palette & Material Specification
- `mat_stair_25`: Slate Blue (`rgba="0.32 0.48 0.65 1.0"`, reflectance 0.2, roughness 0.5) — Flight 1 (2.5 cm)
- `mat_landing_1`: Emerald Green (`rgba="0.22 0.62 0.42 1.0"`, reflectance 0.25, roughness 0.4) — Landing Platform 1
- `mat_stair_30`: Warm Ochre (`rgba="0.68 0.52 0.26 1.0"`, reflectance 0.2, roughness 0.5) — Flight 2 (3.0 cm)
- `mat_landing_2`: Deep Teal (`rgba="0.18 0.58 0.54 1.0"`, reflectance 0.25, roughness 0.4) — Landing Platform 2
- `mat_stair_35`: Terracotta / Coral (`rgba="0.78 0.34 0.26 1.0"`, reflectance 0.2, roughness 0.5) — Flight 3 (3.5 cm max difficulty)
- `mat_summit`: Champion Gold (`rgba="0.88 0.72 0.20 1.0"`, reflectance 0.3, roughness 0.3) — Top Summit Platform
- `target_site_mat`: Translucent Orange (`rgba="1.0 0.4 0.1 0.7"`, reflectance 0.5) — Navigation site beacons

### 3.4 Keyframe Spawn States in `scene_stairs.xml`
Keyframes provide one-click reset postures in MuJoCo viewer (`[`, `]` keys) and define spawn presets for `infer_stairs.py`:
- `INIT`: Flat spawn at $(0, 0, 0.12)$ with zero joint angles.
- `STAND`: Nominal standing pose (STAND2) on flat ground $(0, 0, 0.125)$.
- `SPAWN_STAIRS_25`: Standing on Step 1_2 $(x=0.48, y=0.0, z=0.175)$.
- `SPAWN_LANDING_1`: Standing on Landing Platform 1 $(x=0.84, y=0.0, z=0.200)$.
- `SPAWN_STAIRS_30`: Standing on Step 2_2 $(x=1.20, y=0.0, z=0.260)$.
- `SPAWN_LANDING_2`: Standing on Landing Platform 2 $(x=1.56, y=0.0, z=0.290)$.
- `SPAWN_STAIRS_35`: Standing on Step 3_2 $(x=1.92, y=0.0, z=0.360)$.
- `SPAWN_SUMMIT`: Standing on Summit Platform $(x=2.34, y=0.0, z=0.395)$.

### 3.5 Complete XML Implementation of `scene_stairs.xml`

```xml
<mujoco model="scene_stairs">
    <include file="robot_groundcontact.xml" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />

        <!-- High stairs materials -->
        <material name="mat_stair_25" rgba="0.32 0.48 0.65 1.0" reflectance="0.2" roughness="0.5" />
        <material name="mat_landing_1" rgba="0.22 0.62 0.42 1.0" reflectance="0.25" roughness="0.4" />
        <material name="mat_stair_30" rgba="0.68 0.52 0.26 1.0" reflectance="0.2" roughness="0.5" />
        <material name="mat_landing_2" rgba="0.18 0.58 0.54 1.0" reflectance="0.25" roughness="0.4" />
        <material name="mat_stair_35" rgba="0.78 0.34 0.26 1.0" reflectance="0.2" roughness="0.5" />
        <material name="mat_summit" rgba="0.88 0.72 0.20 1.0" reflectance="0.3" roughness="0.3" />
        <material name="target_site_mat" rgba="1.0 0.4 0.1 0.7" reflectance="0.5" />
    </asset>

    <worldbody>
        <light pos="1.2 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />

        <!-- ================================================================= -->
        <!-- FLIGHT 1: 2.5cm Risers (x: 0.30m -> 0.66m, Elevation: 0 -> 7.5cm) -->
        <!-- ================================================================= -->
        <geom name="stair_1_1" type="box" pos="0.36 0 0.0125" size="0.06 0.25 0.0125"
              material="mat_stair_25" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <geom name="stair_1_2" type="box" pos="0.48 0 0.0250" size="0.06 0.25 0.0250"
              material="mat_stair_25" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <geom name="stair_1_3" type="box" pos="0.60 0 0.0375" size="0.06 0.25 0.0375"
              material="mat_stair_25" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />

        <!-- LANDING PLATFORM 1: (x: 0.66m -> 1.02m, Height: 7.5cm) -->
        <geom name="landing_1" type="box" pos="0.84 0 0.0375" size="0.18 0.25 0.0375"
              material="mat_landing_1" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <site name="site_landing_1" pos="0.84 0 0.080" type="cylinder" size="0.03 0.005" material="target_site_mat" />

        <!-- ================================================================== -->
        <!-- FLIGHT 2: 3.0cm Risers (x: 1.02m -> 1.38m, Elevation: 7.5cm -> 16.5cm) -->
        <!-- ================================================================== -->
        <geom name="stair_2_1" type="box" pos="1.08 0 0.0525" size="0.06 0.25 0.0525"
              material="mat_stair_30" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <geom name="stair_2_2" type="box" pos="1.20 0 0.0675" size="0.06 0.25 0.0675"
              material="mat_stair_30" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <geom name="stair_2_3" type="box" pos="1.32 0 0.0825" size="0.06 0.25 0.0825"
              material="mat_stair_30" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />

        <!-- LANDING PLATFORM 2: (x: 1.38m -> 1.74m, Height: 16.5cm) -->
        <geom name="landing_2" type="box" pos="1.56 0 0.0825" size="0.18 0.25 0.0825"
              material="mat_landing_2" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <site name="site_landing_2" pos="1.56 0 0.170" type="cylinder" size="0.03 0.005" material="target_site_mat" />

        <!-- ================================================================== -->
        <!-- FLIGHT 3: 3.5cm Risers (x: 1.74m -> 2.10m, Elevation: 16.5cm -> 27.0cm) -->
        <!-- ================================================================== -->
        <geom name="stair_3_1" type="box" pos="1.80 0 0.1000" size="0.06 0.25 0.1000"
              material="mat_stair_35" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <geom name="stair_3_2" type="box" pos="1.92 0 0.1175" size="0.06 0.25 0.1175"
              material="mat_stair_35" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <geom name="stair_3_3" type="box" pos="2.04 0 0.1350" size="0.06 0.25 0.1350"
              material="mat_stair_35" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />

        <!-- TOP SUMMIT PLATFORM: (x: 2.10m -> 2.58m, Height: 27.0cm) -->
        <geom name="landing_summit" type="box" pos="2.34 0 0.1350" size="0.24 0.25 0.1350"
              material="mat_summit" condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2" />
        <site name="site_summit" pos="2.34 0 0.275" type="cylinder" size="0.04 0.005" material="target_site_mat" />
    </worldbody>

    <keyframe>
        <key name="INIT" qpos="0 0 0.12 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" ctrl="0 0 0 0 0 0 0 0 0 0 0 0 0 0" />
        <key name="STAND" qpos="
            0 0 0.125 1 0 0 0
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984"
            ctrl="
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984" />
        <key name="SPAWN_STAIRS_25" qpos="
            0.48 0 0.175 1 0 0 0
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984"
            ctrl="
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984" />
        <key name="SPAWN_LANDING_1" qpos="
            0.84 0 0.200 1 0 0 0
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984"
            ctrl="
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984" />
        <key name="SPAWN_STAIRS_30" qpos="
            1.20 0 0.260 1 0 0 0
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984"
            ctrl="
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984" />
        <key name="SPAWN_LANDING_2" qpos="
            1.56 0 0.290 1 0 0 0
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984"
            ctrl="
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984" />
        <key name="SPAWN_STAIRS_35" qpos="
            1.92 0 0.360 1 0 0 0
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984"
            ctrl="
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984" />
        <key name="SPAWN_SUMMIT" qpos="
            2.34 0 0.395 1 0 0 0
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984"
            ctrl="
            0 -0.087266 -0.457924 -0.004940 0.452984
            0.349066 0.349066 0 0
            0 0.087266 0.457924 0.004940 -0.452984" />
    </keyframe>
</mujoco>
```

---

## 4. Analysis of CPU MuJoCo Inference & BAM Actuator Dynamics

### 4.1 Actuator Mechanics & Model Conversion
In GPU Warp training (`mjlab`), the robot is actuated by the **BAM M6** actuator (`FrictionDRBamActuatorCfg`). In CPU rehearsal scripts (`scripts/infer_policy.py`), `bam.mujoco.MujocoController` mirrors this behavior in software.

#### Parameters locked by `tests/test_infer_policy_bam.py`:
- Motor name: `xl330`
- Model: `m6`
- Torque constant $k_t = 0.0384$ N$\cdot$m/A, Resistance $R = 6.2$ $\Omega$
- Firmware stiffness: $k_{p,fw} = 200.0$
- Battery voltage range: nominal $V_{in} = 7.4$ V (training DR $[6.5, 8.2]$ V)
- Voltage sag gain: $V_{drop} = \text{gain} \times \sum |\tau|$ with default gain $= 0.1$ V/Nm, floor $V_{min} = 6.0$ V
- Voltage-bounded force limit: $\tau_{limit} = V_{in} \cdot k_t / R \approx 7.4 \times 0.0384 / 6.2 \approx 0.0458$ N$\cdot$m (internal gear ratio amplifies to output torque limit ~0.96 N$\cdot$m)
- Stiff friction solver parameters:
  - `BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)`
  - `BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)`

#### Conversion Algorithm in `load_mujoco_with_bam`:
1. Parse XML via `mujoco.MjSpec.from_file(xml_path)`.
2. Filter out passive joints (`passive_*`).
3. Convert position actuators to motors: `act.set_to_motor()`, `act.forcelimited = True`, `act.forcerange = (-force_limit, force_limit)`.
4. Clear MuJoCo internal joint damping and friction loss (`joint.damping = 0`, `joint.frictionloss = 0`).
5. Assign armature from BAM actuator: `dof_armature = bam_model.actuator.get_extra_inertia()`.
6. Compile `model = spec.compile()`.
7. Instantiate `bam_ctrl = MujocoController(...)`.

### 4.2 Control Loop & Timing Invariants
- Physics timestep: $\Delta t_{sim} = 0.005$ s ($200$ Hz).
- Decimation: $4$.
- Control period: $\Delta t_{ctrl} = 4 \times 0.005 = 0.020$ s ($50$ Hz).
- Loop sequence per control step:
  1. Capture raw sensors $\rightarrow$ compute 61D observation.
  2. Evaluate ONNX session $\rightarrow$ obtain joint action target $a \in [-1, 1]^{14}$.
  3. Unscale to joint position target: $q_{target} = q_{default} + a \times \text{scale}$.
  4. Write $q_{target}$ to `bam_ctrl.q_target`.
  5. Run decimation sub-steps:
     - Call `bam_ctrl.update()` (computes firmware P-control, voltage sag, back-EMF, writes torque to `data.ctrl`, updates `dof_frictionloss`).
     - Step MuJoCo physics: `mujoco.mj_step(model, data)`.
  6. Sync viewer: `viewer.sync()`.
  7. Sleep remaining budget to match real-time $50$ Hz.

### 4.3 61D Observation Layout Invariant

The 61D actor observation vector layout:

| Obs Segment | Dimension | Description & Source |
|---|---|---|
| `base_ang_vel` | 3 | Gyroscope angular velocity from `imu_ang_vel` sensor [rad/s] |
| `projected_gravity` | 3 | World gravity $[0, 0, -1]$ rotated into body frame via $q_{trunk}^{-1}$ |
| `joint_pos_rel` | 14 | Joint angles relative to default HOME pose: $q - q_{default}$ [rad] |
| `joint_vel` | 14 | Joint angular velocities [rad/s] from $q_{vel}$ |
| `last_action` | 14 | Previous action vector applied by policy |
| `command_block` | 13 | Unified 13D command block: `[twist(3), head_pose(4), body_pose(6)]` |
| **Total** | **61** | **Strictly preserved across all task variants** |

---

## 5. Architectural Design of `scripts/infer_stairs.py`

### 5.1 Overview & Responsibilities
`scripts/infer_stairs.py` is a standalone, lightweight CPU inference and rehearsal script specifically tailored for stairs navigation. It runs independently of `infer_policy.py`, avoiding bloated fetch/kick logic while introducing dedicated stairs diagnostics.

### 5.2 Command-Line Interface (CLI)

```bash
uv run scripts/infer_stairs.py \
    --policy <path_to_policy.onnx> \
    [--scene src/mjlab_microduck/robot/microduck/scene_stairs.xml] \
    [--spawn bottom|flight1|flight2|flight3|top] \
    [--lin-vel-x 0.15] \
    [--vin 7.4] \
    [--debug] \
    [--telemetry-hz 5] \
    [--headless]
```

### 5.3 Interactive Spawn Presets & Hotkeys

To facilitate rapid rehearsal across difficulty levels, the operator can jump to any flight instantly without manually walking up:

| Hotkey / Flag | Spawn Location | $(X, Y, Z)$ Coordinates | Description |
|---|---|---|---|
| `1` (`--spawn bottom`) | Flat Ground Apron | $(0.00, 0.00, 0.125)$ | Flat ground 30 cm before Flight 1 |
| `2` (`--spawn flight1`) | Flight 1 Mid-Step | $(0.48, 0.00, 0.175)$ | Centered on Step 1_2 (2.5 cm risers) |
| `3` (`--spawn flight2`) | Flight 2 Mid-Step | $(1.20, 0.00, 0.260)$ | Centered on Step 2_2 (3.0 cm risers) |
| `4` (`--spawn flight3`) | Flight 3 Mid-Step | $(1.92, 0.00, 0.360)$ | Centered on Step 3_2 (3.5 cm risers) |
| `5` (`--spawn top`) | Top Summit Platform | $(2.34, 0.00, 0.395)$ | Resting area at top summit (27 cm elevation) |

When any spawn key is pressed:
1. `data.qpos[trunk_adr:trunk_adr+3]` is set to the preset coordinates.
2. Orientation is reset to identity quaternion $[1, 0, 0, 0]$.
3. Joint angles are reset to `STAND2` default pose.
4. Linear and angular velocities are zeroed (`data.qvel[:] = 0`).
5. `bam_ctrl.reset(data.qpos)` is called to clear voltage sag state.
6. `mujoco.mj_forward(model, data)` synchronizes kinematics.

### 5.4 Real-Time Keyboard Controls

Both terminal stdin (via `TerminalInput` raw reader) and MuJoCo viewer key callbacks are captured:
- `UP` / `DOWN`: Increase / decrease forward speed $v_x$ (step $\pm 0.05$ m/s, range $[-0.2, 0.3]$ m/s).
- `LEFT` / `RIGHT`: Lateral strafe $v_y$ (step $\pm 0.02$ m/s).
- `A` / `E`: Yaw angular velocity $\omega_z$ (step $\pm 0.1$ rad/s).
- `SPACE`: Zero all velocity commands (emergency stop).
- `B`: Toggle **Body Pitch Trim Mode**:
  - In mode B, `UP` / `DOWN` trims pitch angle $\Delta \theta_{pitch} \pm 2°$ (allows human operator to assist leaning forward into steep 3.5 cm stairs).
- `H`: Toggle **Head Gaze Mode** (tilt camera/beak pitch).
- `C`: Toggle **Heading-Lock** (locks absolute heading to prevent stair yaw drift).
- `P`: Apply **Perturbation Push** (injects random impulse $v_{xy} \sim 0.6$ m/s to test balance recovery).
- `U`: Trigger **Standup Recovery Policy** (switches to standup session if tripped).
- `T`: Toggle policy inference on/off (pause/resume).
- `Q`: Quit runner cleanly.

### 5.5 Real-Time Balance Telemetry & Stair Metrics

Climbing steep stairs stresses bipedal balance, knee torque limits, and foot placement. `infer_stairs.py` implements a dedicated `StairsTelemetry` monitor that runs every step and outputs both to terminal and to the MuJoCo viewer HUD:

#### Telemetry Metrics Calculated:
1. **Trunk Tilt Angles**:
   $$\text{Pitch } \theta = \arcsin(2(q_0 q_2 - q_3 q_1)), \quad \text{Roll } \phi = \arctan2(2(q_0 q_1 + q_2 q_3), 1 - 2(q_1^2 + q_2^2))$$
   - Alarm: If $|\theta| > 35°$ or $|\phi| > 30°$, display warning `[UNSTABLE]`.
   - Fall detection: If $|\theta| > 50°$ or $|\phi| > 45°$, display `[FALLEN - SELF-RECOVERY NEEDED]`.
2. **Step Contact Identification**:
   Iterate through `data.contact`: identify if `left_foot_collision` or `right_foot_collision` contacts `stair_1_*`, `stair_2_*`, `stair_3_*`, or `landing_*`.
   Reports exact step each foot is resting on.
3. **Contact Normal Forces**:
   Sum 3D contact forces on each foot sole:
   $$F_{norm} = \sum_{\text{sole contacts}} |f_{normal}| \quad [\text{N}]$$
4. **Motor Torque & Current Saturation**:
   Track knee servos (`left_knee`, `right_knee`, joints 3 and 12) and ankle servos (`left_ankle`, `right_ankle`, joints 4 and 13):
   $$\tau_{max} = \max(|\tau_{knee}|, |\tau_{ankle}|), \quad \text{Ratio} = \frac{\tau_{max}}{\tau_{limit}} \times 100\%$$
   - Warning if torque exceeds 85% of stall capacity.
5. **Ascent Progress**:
   - Forward progress along stairs: $X$ coordinate.
   - Height gained: $Z - Z_{floor}$.
   - Vertical climbing rate: $\bar{v}_z$ over rolling 1 s window.

#### In-Viewer HUD Display (via `viewer.set_texts`):
- **Top-Left HUD**:
  ```
  FPS: 50.2 Hz | Sim: 3.2ms | Sync: 1.1ms
  Pos: [X: 1.25m, Z: 0.28m] | Ascent v_z: +0.04 m/s
  Pitch: +4.8° | Roll: -0.6° | Status: BALANCED
  ```
- **Top-Right HUD**:
  ```
  Location: FLIGHT 2 (3.0cm Step 2_2)
  Feet: L:stair_2_2 (3.8N) | R:stair_2_1 (4.2N)
  Knee Tau: L:0.42Nm | R:0.46Nm (48% cap)
  Cmd: [vx: +0.15, vy: 0.00, wz: 0.00]
  ```

---

## 6. Testing & Invariant Verification Suite

### 6.1 Analysis of Existing Test Patterns
Existing tests in `tests/` establish clear invariants:
- `tests/test_infer_policy_bam.py`: Validates BAM constants, torque motor conversion, and a 100-step simulation without NaNs.
- `tests/test_velstand_cfg.py`: Verifies task config creation, 61D observation shape, robot spec functions, and reward weight signs.
- `tests/test_slope_terrain.py`: Verifies procedural geom creation, gapless transitions, and difficulty parameters.

### 6.2 Proposed Test Modules for Stairs System

We define three dedicated test files:

#### 1. `tests/test_scene_stairs.xml`: Scene Invariant Tests
- `test_scene_stairs_compiles`: Loads `scene_stairs.xml` via `mujoco.MjModel.from_xml_path()`, confirms zero compilation errors.
- `test_scene_stairs_robot_model`: Confirms the included robot model has 14 actuated joints, 14 motors, and curated ground-contact collision geoms.
- `test_stair_geometries_exist`: Confirms all 9 stair step geoms (`stair_1_1` through `stair_3_3`), intermediate landings, and summit platform exist in the model.
- `test_stair_riser_heights`: Computes top surface $z_{top} = pos_z + size_z$ for each step:
  - Flight 1: step risers equal $0.025$ m $\pm 10^{-6}$.
  - Flight 2: step risers equal $0.030$ m $\pm 10^{-6}$.
  - Flight 3: step risers equal $0.035$ m $\pm 10^{-6}$.
- `test_stair_gapless_continuity`: Verifies that along the X axis, each step geom's front edge matches the next step's back edge:
  $$pos_x^{(k)} + size_x^{(k)} = pos_x^{(k+1)} - size_x^{(k+1)} \pm 10^{-6}$$
- `test_contact_friction_parameters`: Confirms that every stair geom has `condim == 3` and tangential friction $\ge 1.0$.
- `test_scene_keyframes`: Confirms keyframes `INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_STAIRS_30`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT` exist and have correct qpos dimensions.

#### 2. `tests/test_infer_stairs.py`: Inference Runner Invariant Tests
- `test_infer_stairs_bam_constants`: Confirms hardcoded BAM constants in `scripts/infer_stairs.py` exactly match `_BAM_ACTUATOR_KWARGS` in `microduck_constants.py`.
- `test_infer_stairs_actuator_conversion`: Confirms `load_mujoco_with_bam()` converts all 14 servos to torque motors with correct forcerange.
- `test_infer_stairs_step_loop`: Runs 100 control steps (400 physics steps) with live BAM controller and dummy policy, verifying zero NaNs.
- `test_infer_stairs_spawn_presets`: Exercises all 5 spawn presets, verifying robot position is placed within the bounding box of the selected step/landing.
- `test_infer_stairs_telemetry`: Verifies tilt angle calculation, contact detection, and torque load reporting.

#### 3. `tests/test_microduck_stairs_cfg.py`: Environment Config & Invariant Tests
- `test_stairs_env_cfg_builds`: Builds `make_microduck_stairs_env_cfg(play=False)` and `play=True`.
- `test_stairs_actor_observation_dim`: Confirms actor observation is strictly 61D:
  $$\text{dim} = 3 (\text{ang\_vel}) + 3 (\text{proj\_grav}) + 14 (\text{pos}) + 14 (\text{vel}) + 14 (\text{action}) + 13 (\text{cmd}) = 61$$
- `test_stairs_reward_signs`: Confirms every penalty returns $\le 0$ with positive weight, or costs return $\ge 0$ with negative weight.
- `test_stairs_curriculum_stages`: Confirms stair height curriculum schedules $2.5 \rightarrow 3.0 \rightarrow 3.5$ cm.
- `test_stairs_task_registration`: Confirms `Mjlab-StairsClimb-MicroDuck` is registered in `tasks/__init__.py`.

### 6.3 Test Execution Protocol
Because ROS 2 Humble is installed on the host system, running `pytest` directly causes an environment conflict (`ModuleNotFoundError: No module named 'lark'`).
The verified execution command is:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/
```
All existing tests pass 100% under this command.

---

## 7. Actionable Implementation Plan & Builder Recommendations

| Step | Target File | Purpose | Priority |
|---|---|---|---|
| **Step 1** | `src/mjlab_microduck/robot/microduck/scene_stairs.xml` | Implement complete standalone scene with 3 flights, landing platforms, materials, and keyframes | **P0** |
| **Step 2** | `tests/test_scene_stairs.py` | Add unit tests verifying XML compilation, step riser elevations, and gapless continuity | **P0** |
| **Step 3** | `scripts/infer_stairs.py` | Implement dedicated 50 Hz CPU deployment rehearsal runner with BAM dynamics, spawn presets, keyboard controls, and balance telemetry | **P0** |
| **Step 4** | `tests/test_infer_stairs.py` | Add unit tests for runner BAM conversion, spawn presets, and telemetry functions | **P0** |
| **Step 5** | Integration with RL Env | Ensure `microduck_stairs_env_cfg.py` coordinates seamlessly with `scene_stairs.xml` and exports normalizer-baked ONNX | **P1** |

---

## 8. Conclusion

The standalone simulation scene `scene_stairs.xml` and runner `infer_stairs.py` provide a robust, mathematically precise, and invariant-compliant testbed for Microduck's high-stairs climbing. By adhering strictly to `robot_groundcontact.xml`, BAM M6 voltage control, 61D unified observations, and 50 Hz control decimation, zero regressions to the existing codebase are introduced.
