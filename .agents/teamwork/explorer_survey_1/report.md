# Technical Survey & Architecture Plan: Microduck Stairs RL Environment (`microduck_stairs_env_cfg.py`)

**Author**: Explorer Survey 1  
**Target Path**: `/home/dhp01/microduck_rl/src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`  
**Registration Target**: `/home/dhp01/microduck_rl/src/mjlab_microduck/tasks/__init__.py`  
**Date**: 2026-10-11  

---

## 1. Executive Summary

This report delivers the technical survey and design specification for the new **Mjlab-StairsClimb-MicroDuck** reinforcement learning task. The goal is an all-in-one policy that negotiates high stairs ($2.5\,\text{cm} \to 3.0\,\text{cm} \to 3.5\,\text{cm}$, up to ~15% of the ~25 cm robot height), stabilizes posture against perturbations, and automatically recovers / stands up when tripped or fallen.

### Core Architectural Decisions
1. **Inheritance & Base Template**: Build on `make_microduck_velocity_env_cfg` (`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:193`) as the locomotion foundation, integrating the proven recovery & protective-fall layer from `microduck_velstand_env_cfg.py:380`. This preserves the entire domain randomization (DR), sensor noise, delay buffers, BAM startup events, and NaN guards automatically without duplicating code.
2. **Robot Model**: `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (`robot_allcollisions.xml`, 70 collision geoms). Crucial because stair climbing involves shin/knee contact with step edges, and self-recovery requires trunk/arm/head ground interactions that would clip through the world on the feet-only `robot_walk.xml` model.
3. **Terrain Representation**:
   - **GPU Training (mjlab)**: Procedural `TerrainGeneratorCfg` using either `BoxPyramidStairsTerrainCfg` or a dedicated `BoxStairsTerrainCfg` (subclassing `SubTerrainCfg` following `src/mjlab_microduck/tasks/slope_terrain.py:35`), with `_soften_terrain_contacts` (`solref=[0.04, 1.0]`) to suppress solver NaN spikes on hard stair corners.
   - **CPU Rehearsal (`infer_stairs.py`)**: Standalone MJCF XML scene (`scene_stairs.xml`) containing consecutive flight sections with 2.5 cm, 3.0 cm, and 3.5 cm risers.
4. **Calibrated Stair Height Curriculum**:
   - Structured around `NUM_STEPS_PER_ENV = 24` (`step = iteration * 24`).
   - Two-tier progression:
     - **Terrain level progression**: Multi-row difficulty grid in `TerrainGenerator` where row elevation corresponds to $2.5\,\text{cm} \to 3.0\,\text{cm} \to 3.5\,\text{cm}$.
     - **MDP clearance & reward scheduling**: Phased updates via `microduck_mdp.reward_weight` and term parameter curricula at scheduled iteration boundaries (e.g., 0 $\to$ 1500 $\to$ 3000 iters) targeting $2.5\,\text{cm}$, $3.0\,\text{cm}$, and $3.5\,\text{cm}$ foot swing/clearance.
5. **Self-Recovery Integration**:
   - Gated strictly on **TILT ONLY** (`tilt > 40°`), preventing sitting or squatting exploitation.
   - Potential-based progress rewards (`upright_progress`, `height_progress`) that pay $\Delta\cos(\text{tilt})$ and $\Delta z$, making them unfarmable at steady state.
   - Smoothness attempt tax scaled down to 0.1 (`FALLEN_SMOOTHNESS_SCALE`) during falls so recovery motions are not penalized into inaction.
   - Economics layer: `fallen_tax` (-0.5) and one-shot `recovery_success` bounty (+10.0) with hysteresis.
6. **Invariants Preserved**:
   - Strict 61D actor observation `[48D proprio + 13D commands: twist(3), head_pose(4), body_pose(6)]`.
   - BAM voltage-controlled XL330 actuator model with `expand_bam_friction_fields` startup event and `randomize_bam_friction` DR.
   - Observation normalizer baked into exported ONNX models via `scripts/export.py`.

---

## 2. Environment Base Architecture & Hierarchy

### 2.1 File Map & Inheritance Strategy
Examined files:
- `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`: Base locomotion, 61D obs layout, BAM setup, base DR stack.
- `src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py`: Walking + protective fall + recovery layer, all-collision robot model, impact sensor suite.
- `src/mjlab_microduck/tasks/microduck_standup_env_cfg.py`: Pure episodic sit-to-stand rise, single-target keyframe formulation.
- `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py` & `slope_terrain.py`: Custom terrain generator subclassing pattern and slope curriculum.

```
+-------------------------------------------------------------+
| make_microduck_velocity_env_cfg(play, rough)                |
| - 61D obs contract (48 proprio + 13 commands)               |
| - 14 BAM actuators + expand_bam_friction_fields event       |
| - DR stack: com, head_com, pseudo_inertia, bam_friction     |
| - nan_state termination + sensor safe functions             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| make_microduck_stairs_env_cfg(play, rough)                  |
| - Robot: MICRODUCK_ALLCOLLISIONS_ROBOT_CFG (70 geoms)       |
| - Sensors: servo, head, trunk ground contacts + foot scan   |
| - Solver hardening: nconmax=200, iters=30, ls_iters=50      |
| - Soften contacts: _soften_terrain_contacts(solref=0.04)    |
| - Terrain: Stairs TerrainGenerator (2.5 -> 3.0 -> 3.5 cm)   |
| - Stair clearance & foot swing rewards                      |
| - Recovery layer: tilt-gated progress, tax, bounty          |
| - Topple pushes & prone reset mix                           |
+-------------------------------------------------------------+
```

### 2.2 Robot Model Selection
In `src/mjlab_microduck/robot/microduck_constants.py`:
- `MICRODUCK_WALK_ROBOT_CFG`: Collisions on feet only (`left_foot_collision`, `right_foot_collision`). Inadequate for stairs because shins clip stair edges.
- `MICRODUCK_GROUNDCONTACT_ROBOT_CFG`: 11 geoms (feet, knees, trunk).
- `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG`: 70 geoms (`robot_allcollisions.xml`). Includes all 14 XL330 housings named `*_servo_collision` by `name_servo_collision_geoms`.
**Decision**: Use `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` for `microduck_stairs_env_cfg.py`. When climbing 2.5–3.5 cm steps (10–15% robot height), knee and shin impacts are physical constraints; during falls on stairs, full-body collision is mandatory.

### 2.3 Contact Sensors Suite
Following `microduck_velstand_env_cfg.py:413-437`, `microduck_stairs_env_cfg.py` must register:
1. `feet_ground_contact`: Left and right foot collisions vs `terrain` (with `track_air_time=True`).
2. `self_collision`: `trunk_base` subtree vs `trunk_base` subtree.
3. `foot_height_scan`: `TerrainHeightSensorCfg` ring pattern on `left_foot`, `right_foot` sites for real-time clearance measurement over terrain boxes.
4. `servo_ground_contact`: Matches `rf"^.*{SERVO_GEOM_SUFFIX}$"` vs `terrain` (housing protection).
5. `head_ground_contact`: Matches `HEAD_BODY_NAMES` vs `terrain`.
6. `trunk_ground_contact`: Matches `^trunk_base$` vs `terrain`.

### 2.4 MuJoCo Solver Stability & Contact Softening
Stair edges create normal discontinuities that cause impulsive solver forces and NaNs on GPU Warp:
```python
# Required in make_microduck_stairs_env_cfg:
cfg.scene.spec_fn = _soften_terrain_contacts  # sets solref=[0.04, 1.0], solimp=[0.85, 0.95, 0.001, 0.5, 2.0]
cfg.sim.nconmax = 200                         # prevent contact buffer overflow during multi-geom contact
cfg.sim.mujoco.iterations = 30                # default 10 is too low for step edges
cfg.sim.mujoco.ls_iterations = 50             # default 20
```

---

## 3. Observation Layout (61D Invariant) & Domain Randomization

### 3.1 Strict 61D Actor Observation Layout
As mandated by `AGENTS.md` and verified in `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:534-704`:

| Index Range | Name | Dim | Function / Implementation | Details |
|---|---|---|---|---|
| `0:3` | `base_ang_vel` | 3 | `microduck_mdp.base_ang_vel_imu_misaligned` | Lag $[0, 1]$, noise $[-0.03, 0.03]$, $6^\circ$ IMU misalignment |
| `3:6` | `projected_gravity` | 3 | `microduck_mdp.projected_gravity_imu_misaligned` | Lag $[0, 1]$, noise $[-0.01, 0.01]$, $6^\circ$ IMU misalignment |
| `6:20` | `joint_pos` | 14 | `mdp.joint_pos_rel` (or `joint_pos_rel_backlash`) | Relative to HOME, `biased=True`, noise $[-0.001, 0.001]$ |
| `20:34` | `joint_vel` | 14 | `mdp.joint_vel_rel` (or `joint_vel_rel_backlash`) | Fixed 1-ctrl-step delay (Dynamixel filter emulation), noise $[-0.25, 0.25]$ |
| `34:48` | `actions` | 14 | `mdp.last_action` | Action from $t-1$ |
| **0:48** | **Subtotal Proprio** | **48** | | **Proprioceptive base block** |
| `48:51` | `command` | 3 | `mdp.generated_commands(command_name="twist")` | $[v_x, v_y, \omega_z]$ |
| `51:55` | `head_command` | 4 | `mdp.generated_commands(command_name="head_pose")` | $[\Delta\text{neck\_pitch}, \Delta\text{head\_pitch}, \Delta\text{head\_yaw}, \Delta\text{head\_roll}]$ |
| `55:61` | `body_command` | 6 | `mdp.generated_commands(command_name="body_pose")` | $[\Delta x, \Delta y, \Delta z, \Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}]$ |
| **0:61** | **Total Actor Obs** | **61** | | **Frozen contract across entire policy family** |

### 3.2 Critic Observation Layout (76D)
Privileged information provided to the critic:
- `base_lin_vel` (3D): True linear velocity (actor does not see this).
- True uncorrupted `base_ang_vel` (3D), `projected_gravity` (3D), unbiased `joint_pos` (14D).
- Sensor-safe critic terms: `foot_contact_forces_safe` (2D), `foot_height_safe` (2D), `foot_air_time_safe` (2D), `foot_contact` (2D).
- All command blocks: `twist` (3D), `head_pose` (4D), `body_pose` (6D).

### 3.3 Domain Randomization Suite
Inherited directly from `make_microduck_velocity_env_cfg`:
- `randomize_com`: Trunk CoM offset $\pm 3\,\text{mm}$ initial, ramped to $\pm 15\,\text{mm}$ via curriculum (`dr.body_ipos`, non-accumulating).
- `randomize_head_com`: Head assembly bodies CoM offset $\pm 3\,\text{mm}$ initial, ramped to $\pm 10\,\text{mm}$.
- `randomize_mass_inertia`: Physics-consistent $\pm 5\%$ scaling via `dr.pseudo_inertia`.
- `randomize_bam_friction`: `FrictionDRBamActuator.friction_scale` scaled $[0.9, 1.1]$ on reset.
- `randomize_armature`: Reflected rotor inertia scaled $[0.9, 1.1]$ on reset.
- `encoder_bias`: Startup calibration offset $\pm 0.86^\circ$ ($\pm 0.015\,\text{rad}$) per joint.
- `foot_friction`: Tangential friction sampled in $[0.7, 1.3]$ on reset.
- `push_robot`: Standard stumble pushes ($\pm 0.3\,\text{m/s}$).

---

## 4. Terrain & Scene Integration for Stairs

### 4.1 Mjlab Procedural Terrain Capabilities
`mjlab` provides two procedural terrain mechanisms via `TerrainEntityCfg(terrain_type="generator")`:

#### Approach A: Built-in `BoxPyramidStairsTerrainCfg`
Found in `mjlab.terrains.primitive_terrains.py`:
```python
BoxPyramidStairsTerrainCfg(
    proportion=1.0,
    size=(10.0, 10.0),
    step_height_range=(0.025, 0.035),  # 2.5 cm to 3.5 cm
    step_width=0.20,                   # 20 cm tread depth
    platform_width=1.5,                # 1.5 m flat square top platform
    border_width=1.0,                  # 1.0 m flat apron around staircase
    holes=False,
)
```
- **Strengths**: Built into `mjlab`, proven 4-sided climbable structure, automatic platform origin generation.
- **Curriculum Interaction**: When `curriculum=True` on `TerrainGeneratorCfg`, `TerrainGenerator._generate_curriculum_terrains` assigns `difficulty = sub_row / num_rows`, and `BoxPyramidStairsTerrainCfg` interpolates:
  $$\text{step\_height} = \text{step\_height\_range}[0] + \text{difficulty} \times (\text{step\_height\_range}[1] - \text{step\_height\_range}[0])$$
  Row 0 has $2.5\,\text{cm}$ risers, row 5 has $3.0\,\text{cm}$, row 9 has $3.5\,\text{cm}$.

#### Approach B: Dedicated `BoxStraightStairsTerrainCfg` (`SubTerrainCfg`)
Modeled after `FlatRampTerrainCfg` (`src/mjlab_microduck/tasks/slope_terrain.py:35`):
A custom terrain class subclassing `SubTerrainCfg` can generate a sequential straight staircase flight along $+x$:
1. Flat launch platform ($2.0\,\text{m}$ length).
2. Flight of $N$ steps along $+x$ with riser height $h(\text{difficulty}) \in [0.025, 0.035]$ and tread depth $0.18\,\text{m}$.
3. Elevated summit platform ($2.0\,\text{m}$ length).
4. Descending steps and lower runout platform.

**Recommendation**:
- For **GPU training in `microduck_stairs_env_cfg.py`**: Use `BoxPyramidStairsTerrainCfg` (or a customized straight stairs subterrain) with `TerrainGeneratorCfg(size=(10.0, 10.0), curriculum=True, num_rows=6, num_cols=1, difficulty_range=(0.0, 1.0))`.
- For **CPU rehearsal in `scripts/infer_stairs.py`**: Create the dedicated MJCF `scene_stairs.xml` as specified by Requirement R2.

### 4.2 Elevation-Aware Spawn Coordinates
**Critical Lesson from `microduck_velstand_env_cfg.py:128-132` (wy8gcaus crash)**:
Hardcoding absolute spawn $z$ coordinates ($z \in [0.05, 0.09]$) causes catastrophic value loss divergence on non-flat terrain because subterrain origins have non-zero elevations. All resets (`reset_base`, `random_prone_init`) must compute height relative to `env.scene.terrain.env_origins[env_ids, 2]` (`_env_origin_z`):
$$z_{\text{world}} = z_{\text{nominal}} + z_{\text{origin}}$$

---

## 5. Calibrated Stair Height Curriculum (2.5 cm -> 3.0 cm -> 3.5 cm)

### 5.1 Mechanics of Step Counting & Scheduling
In `mjlab_microduck`:
- `NUM_STEPS_PER_ENV = 24` (24 environment simulation steps per PPO rollout iteration).
- `common_step_counter = iteration * 24`.
- Curricula run per PPO iteration and evaluate against `env.common_step_counter`.

### 5.2 Two-Tier Curriculum Architecture

```
+-------------------------------------------------------------------------------+
| Tier 1: Terrain Difficulty Progression (Spatial / Performance-Driven)         |
| Row 0: 2.5 cm riser -> Row 2: 3.0 cm riser -> Row 5: 3.5 cm riser            |
| Function: terrain_levels_stairs (distance-based promotion via update_env_origins)
+-------------------------------------------------------------------------------+
                                        +
+-------------------------------------------------------------------------------+
| Tier 2: Reward & Clearance Scheduling (Temporal / Step-Staged)                |
| Stage 1 (0 - 1500 iters):    foot_clearance target = 2.5 cm, fell_over @ 70°  |
| Stage 2 (1500 - 3000 iters):  foot_clearance target = 3.0 cm, fell_over -> pi |
| Stage 3 (3000 - 5000+ iters): foot_clearance target = 3.5 cm, recovery live   |
+-------------------------------------------------------------------------------+
```

#### Tier 1: Performance-Driven Terrain Progression (`terrain_levels_stairs`)
Following `terrain_levels_slope` (`src/mjlab_microduck/tasks/mdp.py:3732`):
```python
def terrain_levels_stairs(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # Promote if robot traverses onto the stair platform (> platform boundary)
    move_up = distance > 2.5  # meters
    # Demote if robot falls or covers less than 0.8 m
    move_down = distance < 0.8
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
```

#### Tier 2: Step-Staged Parameter & Reward Schedules
1. **Foot Clearance & Swing Height Target Curriculum**:
   Using `event_param_curriculum` / reward term updates:
   ```python
   # Scheduled clearance targets to match increasing step risers:
   # 0 -> 1500 iters: 2.8 cm target (clears 2.5 cm riser)
   # 1500 -> 3000 iters: 3.3 cm target (clears 3.0 cm riser)
   # 3000+ iters: 3.8 cm target (clears 3.5 cm riser)
   ```
2. **Action Rate Smoothing Curriculum**:
   From `microduck_velocity_env_cfg.py:788`:
   - Start at $-0.1$ to permit high-variance leg flailing needed to explore stepping up.
   - Ramp to $-0.4$ by iter 1500, $-0.8$ by iter 2500, and $-1.0$ by iter 3500.

---

## 6. Dynamic Self-Recovery & Posture Balance Integration

### 6.1 Avoiding the 5 Hard-Won Traps from `VelStand` & `StandUp`
`microduck_velstand_env_cfg.py:64-127` documents 7 development cycles of failure modes. We must bake these lessons directly into the stairs configuration:

1. **The State-Gating Farm Trap**:
   - *Bug*: Gating recovery rewards on low height ($z < 0.10\,\text{m}$) caused the robot to crouch/sit and farm upward velocity and posture rewards forever without climbing stairs.
   - *Fix*: Recovery rewards MUST gate strictly on **TILT ONLY** (`tilt > 40°`). Normal stair climbing operates at $\text{tilt} \in [0^\circ, 15^\circ]$; a robot tilted $> 40^\circ$ is genuinely toppled.
2. **The Attempt Tax Trap**:
   - *Bug*: Smoothness penalties (`action_rate_l2 = -1.0`, `joint_torque_rate = -0.002`) active during a fall taxed explosive get-up attempts by $-2.0/\text{step}$. "Lying still" was the mathematically optimal behavior.
   - *Fix*: Scale smoothness taxes by `FALLEN_SMOOTHNESS_SCALE = 0.1` whenever $\text{tilt} > 40^\circ$ via `action_rate_l2_fallen_scaled` and `joint_torque_rate_l2_fallen_scaled`.
3. **The Unfarmable Potential Formulation**:
   - Use `upright_progress` ($\Delta\cos(\text{tilt})$, weight $+5.0$) and `height_progress` ($\Delta z$, weight $+30.0$, ceiling $0.115\,\text{m}$).
   - Rising pays, falling costs, holding any stationary pose pays zero.
4. **The Crouch Endpoint Trap**:
   - *Bug*: Prone-to-crouch was learned, but crouch-to-stand never fired because recovery bounties demanded $z > 0.105\,\text{m}$, which is higher than the walking equilibrium.
   - *Fix*: Define recovery completion as $\text{tilt} < 25^\circ$ AND $z > 0.09\,\text{m}$ (`RECOVERED_UP_TILT_DEG = 25.0`, `RECOVERED_UP_Z = 0.09`).
5. **Recovery Economics (Tax + Bounty)**:
   - `fallen_tax` (`microduck_mdp.fallen_state_penalty`, weight $-0.5$): Continuous penalty while fallen, with hysteresis (releases only when $\text{tilt} < 25^\circ$ and $z > 0.09\,\text{m}$). Lying still costs $-0.5/\text{step}$.
   - `recovery_success` (`microduck_mdp.recovery_success`, weight $+10.0$): One-shot payout when transitioning from fallen ($\ge 0.5\,\text{s}$) to recovered.

### 6.2 Topple Pushes & Prone Spawns on Stairs
- `topple_push`: Interval pushes ($4.0 \dots 8.0\,\text{s}$) with velocity kicks ramped from $\pm 0.3\,\text{m/s}$ up to $\pm 0.9\,\text{m/s}$. Knocks the robot over occasionally so falling and landing on stairs are on-policy data.
- `random_prone_init`: Resets a fraction ($15\% \dots 35\%$) of environments into prone, supine, or side orientations. Must use origin-relative height (`_env_origin_z + 0.06`).

### 6.3 Servo Protection Costs
To prevent stripping XL330 gears during falls on hard stair edges:
- `servo_impact`: Weight $-0.02$, threshold $2.0\,\text{N}$ on servo housing geoms.
- `head_impact`: Weight $-0.01$, threshold $15.0\,\text{N}$ on head subtree.
- `trunk_impact`: Weight $-0.01$, threshold $20.0\,\text{N}$ on trunk shell.
- `servo_stall`: Weight $-0.05$, flags torque $> 0.4\,\text{N}\cdot\text{m}$ at velocity $< 0.5\,\text{rad/s}$ (pinned limbs).
- `gentle_rise`: Weight $+0.005$ (positive weight for self-negating `trunk_vertical_accel_penalty`, penalizing $|a_z|$).

---

## 7. Task Registration, Backlash Twin, & Runner Conventions

### 7.1 Registration in `src/mjlab_microduck/tasks/__init__.py`
Following lines 84–248 of `tasks/__init__.py`:
```python
from .microduck_stairs_env_cfg import (
    make_microduck_stairs_env_cfg,
    MicroduckStairsRlCfg,
)

# Standard stairs task
register_mjlab_task(
    task_id="Mjlab-StairsClimb-MicroDuck",
    env_cfg=make_microduck_stairs_env_cfg(),
    play_env_cfg=make_microduck_stairs_env_cfg(play=True),
    rl_cfg=MicroduckStairsRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

### 7.2 Backlash Twin Registration
In `tasks/__init__.py:269-286`:
Add to `_BACKLASH_TASKS`:
```python
("Mjlab-StairsClimb-Backlash-MicroDuck", make_microduck_stairs_env_cfg, {}, MicroduckStairsRlCfg, _BL_ALLCOLLISIONS),
```
This automatically invokes `make_backlash_variant(cfg, _BL_ALLCOLLISIONS)` (`tasks/backlash.py:42`), remapping `joint_pos` and `joint_vel` to encoder views through the $\pm 1^\circ$ backlash gears while keeping the 61D observation shape identical.

### 7.3 Runner Configuration (`MicroduckStairsRlCfg`)
```python
MicroduckStairsRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(  # or PpoWithExpertBcCfg if warm-starting
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,  # symmetry OFF for stairs
    ),
    wandb_project="mjlab_microduck",
    experiment_name="stairs_climb",
    run_name="stairs_climb",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
```

### 7.4 Export Compatibility (`scripts/export.py`)
`scripts/export.py` executes:
```bash
uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file output_stairs.onnx
```
Because `make_microduck_stairs_env_cfg(play=True)` produces the exact 61D observation term specification and normalizer configuration, `runner.export_policy_to_onnx` bakes the empirical mean and standard deviation into the ONNX graph with zero manual conversion.

---

## 8. Standalone Test Scene & Runner Specification (`infer_stairs.py`)

As required by Requirement R2:
1. **Scene File**: `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
   - Include `robot_allcollisions.xml` (or `robot_walk.xml` for walk rehearsal).
   - Define consecutive flights:
     - Flight 1: 3 steps with $2.5\,\text{cm}$ ($0.025\,\text{m}$) risers, $0.18\,\text{m}$ treads.
     - Intermediate Landing: $0.6\,\text{m} \times 0.5\,\text{m}$ platform.
     - Flight 2: 3 steps with $3.0\,\text{cm}$ ($0.030\,\text{m}$) risers, $0.18\,\text{m}$ treads.
     - Intermediate Landing: $0.6\,\text{m} \times 0.5\,\text{m}$ platform.
     - Flight 3: 3 steps with $3.5\,\text{cm}$ ($0.035\,\text{m}$) risers, $0.18\,\text{m}$ treads.
     - Summit Platform: $1.0\,\text{m} \times 0.5\,\text{m}$ platform.
2. **Dedicated Runner**: `scripts/infer_stairs.py`
   - Standalone CPU MuJoCo runner mirroring `scripts/infer_policy.py:58-100` with BAM M6 motor model (`load_bam_model(kp_fw=200.0, vin=7.4)`).
   - Keyframe / spawn presets: `SPAWN_GROUND`, `SPAWN_STAIRS_25MM`, `SPAWN_STAIRS_30MM`, `SPAWN_STAIRS_35MM`, `SPAWN_FALLEN_FACEUP`, `SPAWN_FALLEN_FACEDOWN`.
   - Telemetry overlay displaying step height, foot clearance, trunk tilt, and contact normal forces.

---

## 9. Verification & Acceptance Gate Checklist

| Item | Verification Command / Method | Expected Result |
|---|---|---|
| **Task Registration** | `uv run list-envs \| grep StairsClimb` | `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck` appear in task table |
| **Cfg Regression Tests** | `uv run --with pytest pytest tests/test_stairs_cfg.py` | 100% pass: 61D obs layout, all-collisions robot spec, reward weight signs |
| **Full Test Suite** | `uv run --with pytest pytest tests/` | Zero regressions across all existing task tests |
| **Smoke Test** | `uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5` | Completes 5 iterations with 0 NaNs and finite loss |
| **ONNX Export** | `uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/stairs_smoke.onnx` | Generates valid ONNX graph with 61D input, 14D output, baked normalizer |
| **Standalone CPU Runner** | `uv run scripts/infer_stairs.py --help` | CLI flags compile cleanly, loads `scene_stairs.xml` and BAM dynamics |
