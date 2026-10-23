# Explorer Survey 2: Investigation & Mathematical Design of High-Stairs MDP Terms

**Task**: Investigation of `src/mjlab_microduck/tasks/mdp.py` and Design of High-Stairs MDP Terms (2.5 cm – 3.5 cm)  
**Target Environment**: `Mjlab-StairsClimb-MicroDuck` (and `-Backlash-` variant)  
**Author**: Explorer Survey 2  
**Date**: 2026-10-11  
**Status**: Complete  

---

## 1. Executive Summary & Objectives

The Microduck robot is a ~800 g, ~25 cm tall bipedal robot powered by 14 Dynamixel XL330 servos (voltage-controlled via BAM actuator dynamics). The user requirements specify training an all-in-one PPO policy that can:
1. Traverse high stairs with risers of **2.5 cm, 3.0 cm, and 3.5 cm** (up to ~15% of robot height).
2. Maintain robust dynamic balance against perturbations and topples.
3. Automatically self-recover and stand up when tripped or fallen on the stairs.
4. Strictly conform to existing repository invariants: 61D actor observation layout, 14 BAM actuators, running normalizer baked in ONNX, zero breaking changes to existing tasks.

This report provides the exhaustive investigation of `src/mjlab_microduck/tasks/mdp.py` and the mathematical formulation, signature design, and code templates for the 4 core stairs MDP term groups:
- **Group 1: Stair-climbing clearance reward/penalty** (foot trajectory over step edges).
- **Group 2: Foot landing damping** (penalizing impact force and downward touchdown velocity).
- **Group 3: Contact distribution & anti-tripping** (encouraging sole contact on treads, penalizing riser-face collisions).
- **Group 4: Recovery potential rewards** (potential-based upright/height shaping invariant to stair elevation).

---

## 2. Codebase Review: `src/mjlab_microduck/tasks/mdp.py` Architecture & Conventions

### 2.1 File Organization & Grouping Conventions
`src/mjlab_microduck/tasks/mdp.py` is the **single source of truth** (7,686 lines, 234 functions) for all custom MDP components across all tasks in `microduck_rl`. Its structure follows a strict modular progression:

1. **Patches 1–7 (Lines 25–214)**:
   - *Patch 1*: `RewardManager.compute` — Sanitizes NaN rewards in-place before PPO buffer storage.
   - *Patch 2*: `PPO.compute_returns` — Replaces NaNs/Infs in advantages and returns with 0.0 before advantage normalization.
   - *Patch 4*: `_get_base_metadata_no_passive` — Filters `passive_*` joints during ONNX export metadata serialization, enforcing a clean 14D action layout.
   - *Patch 5*: `MICRODUCK_WARM_START=1` — Automatically resets `common_step_counter = 0` and `iteration = 0` upon checkpoint loading so curricula restart cleanly while retaining weights and normalizer.
   - *Patch 6*: `check_nan` — In-place sanitization of actor/critic obs and rewards buffers.
   - *Patch 7*: `_smart_get_checkpoint_path` — Direct file path support for cross-task warm starts.
2. **Core Helpers & Entity Accessors (Lines 218–270)**:
   - `_servo_joint_ids(env, asset)`: Cached lookup of the 14 servo joints via `asset.find_joints(r"^(?!passive_).*")`.
   - `_servo_joint_pos(env, asset)` / `_servo_joint_vel(env, asset)`: Safe slicing of `joint_pos` / `joint_vel` using cached servo IDs.
   - `_env_origin_z(env, env_ids)`: Terrain origin elevation for rough terrain.
3. **Task-Specific Groupings**:
   - Locomotion / Gait (Lines 639–1050): `feet_air_time_upright`, `upright_progress`, `height_progress`, `fallen_state_penalty`, `recovery_success`, `servo_stall_penalty`, `servo_acc_spike_penalty`, `action_rate_l2_fallen_scaled`.
   - Crouching & Slope Descent (Lines 1240–1600).
   - Swizzle / Skating / Balance (Lines 1670–2040): `feet_flat_penalty`, `feet_tiptoe_alignment`, `feet_grounded_reward`, `body_impact_cost`.
   - SitStand & Standup (Lines 2045–2800): `interpolated_pose_l1_penalty`, `multistage_height_l1_penalty`, `trunk_vertical_accel_penalty`.
   - GroundPick (Lines 3100–3350).
   - Domain Randomization Events (Lines 3352–3850): non-accumulating DR routines.
   - Command Generators (Lines 4700–5400): `UniformPoseCommand` (head and body pose blocks).
   - BallKick, Spin, Roulade, Fetch (Lines 5900–7686).

### 2.2 Reward Sign Conventions (Strict Invariant)
As documented in `AGENTS.md` and enforced in `tests/test_velstand_cfg.py`:
- **mjlab cost functions**: Return values $\ge 0$ (costs). They **must** be weighted with **negative weights** ($w < 0$).
- **Microduck penalties**: Named `*_penalty` or `*_l1`, returning values $\le 0$ (self-negating). They **must** be weighted with **positive weights** ($w > 0$).
- **Double-negation trap**: Assigning a negative weight to a self-negating penalty ($(-w) \times (-\text{cost}) = +w \cdot \text{cost}$) transforms a violation into a positive reward, causing the policy to exploit and farm the violation.
- **Infallible verification check**: In Weights & Biases, every `Episode_Reward/<penalty>` term must log values $\le 0.0$.

### 2.3 Robot Model Invariants & Backlash Compatibility
- **Robot Kinematics**:
  - Left leg: 5 servos (0: `hip_yaw`, 1: `hip_roll`, 2: `hip_pitch`, 3: `knee`, 4: `ankle`).
  - Head/neck: 4 servos (5: `neck_pitch`, 6: `head_pitch`, 7: `head_yaw`, 8: `head_roll`).
  - Right leg: 5 servos (9: `hip_yaw`, 10: `hip_roll`, 11: `hip_pitch`, 12: `knee`, 13: `ankle`).
- **Passive Joints in Backlash Models**:
  - Backlash models (`robot_allcollisions_backlash.xml`, `robot_walk_backlash.xml`) insert unactuated hinges named `passive_<joint>_backlash` in series with every servo.
  - **Rule**: Never index `joint_pos[:, 0:14]`. Always index via `_servo_joint_ids(env, asset)`.
- **Foot Bodies, Sites, and Collision Geoms**:
  - Ankle link bodies: `ankle_left`, `ankle_right`.
  - Collision geoms: `left_foot_collision` (sole_left mesh), `right_foot_collision` (sole_right mesh).
  - Foot tracking sites: `left_foot`, `right_foot`.
  - These names are identical across all model variants (`walk`, `groundcontact`, `allcollisions`, and backlash twins).

---

## 3. High-Stairs MDP Terms: Mathematical Formulations & Design

### Group 1: Stair-Climbing Clearance Reward / Penalty

#### Physics & Problem Context:
On high stairs (risers $h_{riser} \in [0.025, 0.035]$ m, tread length $\sim 0.12$–$0.15$ m), the robot's leg length is only $\sim 14$ cm. Lifting the foot over a 3.5 cm riser requires raising the foot by $> 25\%$ of leg length.
Standard mjlab `feet_clearance` calculates:
$$\text{cost} = \sum_{f} |h_f - h_{target}| \cdot \|v_{xy, f}\|$$
**Why this breaks on stairs**:
1. Symmetric absolute error $|h_f - h_{target}|$ penalizes the foot for lifting *higher* than $h_{target}$. When clearing a 3.5 cm step, high-stepping with a 4.5 cm swing is punished, forcing the robot to clip the riser lip.
2. If $h_{target}$ is increased globally to 4.5 cm, the robot is penalized during push-off and flat walking for not being at 4.5 cm.

#### Proposed Design 1.1: One-Sided Clearance Penalty (`stairs_foot_clearance_penalty`)
Only penalizes **insufficient clearance** (clearance deficit) during forward swing motion. Lifting above the target is free ($0.0$ penalty).

**Mathematical Formulation**:
For each foot $f \in \{\text{left}, \text{right}\}$:
$$\Delta h_f = \max(0.0, h_{target} - h_{sensor, f})$$
$$P_{clearance} = - \sum_{f} \Delta h_f \cdot \max(0.0, \|v_{xy, f}^{site}\| - v_{min}) \cdot \mathbb{I}(\text{cmd}_{active})$$
Where:
- $h_{sensor, f}$: Terrain clearance from `TerrainHeightSensor` (`foot_height_scan.data.heights[:, f]`).
- $v_{xy, f}^{site}$: Horizontal site velocity `asset.data.site_lin_vel_w[:, site_ids, :2]`.
- $h_{target}$: Target clearance (curriculum scheduled: $0.030 \to 0.038 \to 0.045$ m).
- $v_{min}$: Minimum velocity threshold ($0.03$ m/s) to ignore stationary jitter.
- Sign convention: Returns $\le 0.0$ (self-negating microduck penalty), weight $> 0.0$.

#### Proposed Design 1.2: Swing Peak Clearance at Touchdown (`stairs_swing_clearance_penalty`)
Evaluated once per stride at touchdown (`first_contact`):
$$\text{deficit}_f = \max(0.0, h_{target} - h_{peak, f})$$
$$P_{peak} = - \sum_{f} \left(\frac{\text{deficit}_f}{h_{target}}\right)^2 \cdot \mathbb{I}(\text{first\_contact}_f)$$
This directly encourages the swing foot trajectory to achieve high apex clearance before landing on the next step tread.

---

### Group 2: Foot Landing Damping (Impact Force & Normal Velocity)

#### Physics & Problem Context:
The Dynamixel XL330 servos possess a plastic gear train with a 288:1 reduction ratio. Landing hard on a rigid stair step transmits impulsive contact forces back into the gear teeth (rotor inertia reflected by $288^2 \approx 8.3 \times 10^4$).
Existing mjlab `soft_landing`:
1. Penalizes total contact force $\|F\|$ without an impact threshold, penalizing normal weight support (~4 N per foot).
2. Completely ignores foot downward velocity $v_z$ prior to impact.

#### Proposed Design 2.1: Excess Contact Force Impact Penalty (`stairs_foot_impact_penalty`)
Penalizes impulsive force spikes above quasi-static body weight at touchdown:
$$F_{net, f} = \|F_f\| \quad (\text{from } \texttt{ContactSensor.data.force})$$
$$\Delta F_f = \max(0.0, F_{net, f} - F_{thresh})$$
$$P_{impact} = - \sum_f \Delta F_f \cdot \mathbb{I}(\text{first\_contact}_f)$$
Where:
- $F_{thresh} = 8.0$ N (Microduck weight $m g \approx 0.8 \times 9.81 = 7.85$ N; forces $\le 8$ N are free).
- $\mathbb{I}(\text{first\_contact}_f)$: Computed via `contact_sensor.compute_first_contact(dt)`.
- Returns $\le 0.0$ (weight $> 0.0$).

#### Proposed Design 2.2: Downward Touchdown Velocity Damping (`stairs_foot_landing_velocity_penalty`)
Penalizes excessive downward velocity $v_z < 0$ at the moment of touchdown, encouraging pre-impact deceleration:
$$v_{down, f} = \max(0.0, -v_{z, f}^{site})$$
$$P_{down\_vel} = - \sum_f \max(0.0, v_{down, f} - v_{thresh})^2 \cdot \mathbb{I}(\text{first\_contact}_f)$$
Where:
- $v_{thresh} = 0.08$ m/s (normal gentle footfall velocity).
- Returns $\le 0.0$ (weight $> 0.0$).

#### Combined Damping Formulation:
$$P_{landing\_damping} = - \sum_f \left[ \Delta F_f + \gamma \cdot \max(0.0, v_{down, f} - v_{thresh})^2 \right] \cdot \mathbb{I}(\text{first\_contact}_f)$$

---

### Group 3: Contact Distribution & Anti-Tripping

#### Physics & Problem Context:
Stairs present two geometric surfaces:
1. **Horizontal Treads**: Normal $\mathbf{n} \approx [0, 0, 1]^T$. Sole must be placed flat, supporting vertical gravity ($F_z > 0$, $\|F_{xy}\| \ll F_z$).
2. **Vertical Risers**: Normal $\mathbf{n} \approx [\pm 1, 0, 0]^T$ ($n_z \approx 0$). Toe-stubbing or riser collisions produce large horizontal reaction forces opposing forward progress while providing zero vertical support.

#### Proposed Design 3.1: Stair Riser Collision & Tripping Penalty (`stairs_riser_trip_penalty`)
Detects collision against vertical riser surfaces.

**Method A: Using Surface Normals (if `fields` includes `"normal"`)**:
$$n_{z, f} = \text{contact\_normal}_z \in [0, 1]$$
On a tread, $n_z \approx 1.0$; on a riser, $n_z \approx 0.0$.
$$\text{riser\_trip}_f = (1.0 - n_{z, f}) \cdot \|F_f\| \cdot \mathbb{I}(\text{in\_contact}_f)$$
$$P_{riser} = - \sum_f \text{riser\_trip}_f$$

**Method B: Using Global Contact Forces (`reduce="netforce"`, always available)**:
In global coordinates, $F_z$ is vertical support, and $F_{xy} = [F_x, F_y]$ is horizontal shear.
On a normal tread with friction coefficient $\mu \approx 0.8$, the maximum friction force is $\|F_{xy}\| \le \mu F_z$.
During a riser impact, $F_x$ is a normal force while $F_z \approx 0$, so $\|F_{xy}\| \gg F_z$.
$$\text{shear\_excess}_f = \max(0.0, \|F_{xy, f}\| - \mu_{tread} \cdot F_{z, f})$$
$$P_{riser\_trip} = - \sum_f \text{shear\_excess}_f \cdot \mathbb{I}(\text{in\_contact}_f)$$
Where $\mu_{tread} = 0.5$. Returns $\le 0.0$ (weight $> 0.0$).

#### Proposed Design 3.2: Sole Flat Tread Alignment Penalty (`stairs_sole_flat_penalty`)
Ensures that whenever a foot is in stance on a step, the entire sole remains parallel to the horizontal tread, preventing edge-tipping:
$$\mathbf{g}_{site, f} = R_{site, f}^T \mathbf{g}_w$$
$$P_{sole\_flat} = - \sum_f \|\mathbf{g}_{site, f, xy}\|^2 \cdot \mathbb{I}(\text{in\_contact}_f)$$
Returns $\le 0.0$ (weight $> 0.0$). When flat on the tread, $\mathbf{g}_{site, f} = [0, 0, -1]^T \implies xy = [0, 0] \implies \text{penalty} = 0.0$.

#### Proposed Design 3.3: Stance Load Distribution (`stairs_double_support_distribution_penalty`)
During double support on stairs, penalizes extreme load asymmetry:
$$\text{asym} = \frac{|F_{z, left} - F_{z, right}|}{F_{z, left} + F_{z, right} + \epsilon} \cdot \mathbb{I}(\text{both\_contact})$$
$$P_{distrib} = - \text{asym} \le 0.0$$

---

### Group 4: Recovery Potential Rewards (Self-Recovery on Stairs)

#### Critical Discovery: Why Standard `height_progress` Breaks on Stairs
In `microduck_velstand_env_cfg.py`, `height_progress` is computed as:
```python
z = torch.nan_to_num(
    asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
)
pot = torch.clamp(z, max=ceiling)  # ceiling = 0.115 m
```
**Fatal Failure Mode on Stairs**:
- On step 0 ($z=0$): Robot standing trunk $z = 0.117$ m.
- On step 2 ($z = 0.06$ m): Fallen robot trunk $z \approx 0.06 + 0.05 = 0.110$ m.
- On step 3 ($z = 0.09$ m): Fallen robot trunk $z \approx 0.09 + 0.05 = 0.140$ m!
Notice that on step 3, a **completely fallen, prone robot** already has trunk $z = 0.140 > 0.115$ m!
`pot` is saturated at $0.115$ m even while lying flat on its face. The policy receives **zero gradient** to stand up!
Moreover, climbing stairs would erroneously pay the recovery height reward if the ceiling were simply increased.

#### Proposed Solution: Relative Trunk Height
We define height progress relative to the **local support feet / terrain beneath the robot**:
$$z_{rel} = z_{trunk} - \min(z_{left\_foot}, z_{right\_foot})$$
Alternatively, if `trunk_height_scan` (a raycast `TerrainHeightSensor` on `trunk_base`) is added:
$$z_{rel} = \text{trunk\_height\_scan.data.heights}$$
**Properties of $z_{rel}$ on Stairs**:
- Standing upright on ANY step: $z_{rel} \approx 0.115$–$0.120$ m.
- Prone/fallen on ANY step: $z_{rel} \approx 0.035$–$0.050$ m.
- Deep crouch on ANY step: $z_{rel} \approx 0.065$–$0.080$ m.
- Climbing stairs: Feet and trunk rise together, so $z_{rel}$ remains constant at $\sim 0.117$ m $\implies \Delta z_{rel} \approx 0.0$. **Climbing stairs cannot farm recovery reward!**

#### Proposed Design 4.1: Stairs Potential Upright Shaping (`stairs_upright_progress`)
Orientation potential is intrinsically elevation-invariant:
$$\Phi_{upright} = \cos(\text{tilt}) = 1.0 - 2.0 (q_x^2 + q_y^2)$$
$$\Delta \Phi_{upright} = \Phi_{upright}(t) - \Phi_{upright}(t-1)$$
- Fresh reset mask applied: `delta[episode_length_buf <= 1] = 0.0`.
- Rising toward vertical pays $\Delta \cos(\text{tilt}) > 0.0$.
- Toppling charges $\Delta \cos(\text{tilt}) < 0.0$.
- Holding any fallen pose pays **identically 0.0**. Unfarmable!

#### Proposed Design 4.2: Stairs Relative Height Progress (`stairs_relative_height_progress`)
$$\Phi_{height} = \min(z_{rel}, h_{ceiling}) \quad (h_{ceiling} = 0.115\text{ m})$$
$$\Delta \Phi_{height} = \Phi_{height}(t) - \Phi_{height}(t-1)$$
Provides continuous gradient to straighten the legs and extend the knees out of a deep crouch on the stairs.

#### Proposed Design 4.3: Fallen Tax with Hysteresis (`stairs_fallen_state_penalty`)
Flat per-step penalty on remaining fallen:
$$\text{tax} = 1.0 \quad \text{while } (\text{tilt} > 40^\circ \text{ or } z_{rel} < 0.075\text{ m})$$
Release conditions: $\text{tilt} < 25^\circ$ AND $z_{rel} > 0.090$ m.
Armed with hysteresis so the robot cannot avoid the tax by resting in a semi-crouch below the gate.

#### Proposed Design 4.4: Recovery Completion Bounty (`stairs_recovery_success`)
One-shot bounty firing on the exact frame an env fallen for $\ge 0.5$ s completes recovery ($\text{tilt} < 25^\circ$ and $z_{rel} > 0.090$ m).

---

## 4. Implementation-Ready Code Signatures & Templates

Below are the exact Python function definitions designed for addition to `src/mjlab_microduck/tasks/mdp.py`.

```python
# =============================================================================
# Stairs Climbing Task — Clearance, Landing Damping, Contact & Recovery Terms
# =============================================================================

def stairs_foot_clearance_penalty(
    env: ManagerBasedRlEnv,
    target_height: float = 0.035,
    height_sensor_name: str = "foot_height_scan",
    command_name: str | None = "twist",
    command_threshold: float = 0.01,
    vel_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot clearance shortfall during forward swing on stairs.

    Unlike mjlab's feet_clearance (which penalizes lifting higher than target),
    this is ONE-SIDED: lifting higher than target_height is completely free
    (0 penalty). Only penalizes clearance deficits (target_height - h > 0)
    when the foot is moving forward horizontally with velocity > vel_threshold.

    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height_sensor = env.scene[height_sensor_name]
    foot_height = height_sensor.data.heights  # [B, 2]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    vel_norm = torch.linalg.norm(foot_vel_xy, dim=-1)  # [B, 2]

    # One-sided clearance deficit: positive only when clearance is insufficient
    clearance_deficit = torch.clamp(target_height - foot_height, min=0.0)
    moving = torch.clamp(vel_norm - vel_threshold, min=0.0)

    # Cost per env (summed over feet)
    cost = torch.sum(clearance_deficit * moving, dim=1)

    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)
        if cmd is not None:
            cmd_active = (torch.linalg.norm(cmd[:, :2], dim=1) > command_threshold).float()
            cost = cost * cmd_active

    return -cost


def stairs_foot_landing_damping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    impact_force_thresh: float = 8.0,
    impact_vel_thresh: float = 0.08,
    vel_weight: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive impact force and downward normal velocity at foot landing.

    Evaluated exclusively at touchdown (first_contact).
    - Forces below impact_force_thresh (~1x body weight) are free.
    - Downward normal velocity (-v_z) above impact_vel_thresh is penalized quadratically.

    Protects XL330 gear trains and damping footfall bounce on rigid stair treads.
    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    from mjlab.sensor import ContactSensor
    contact_sensor: ContactSensor = env.scene[sensor_name]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, 2] bool

    # Contact force magnitude in excess of body weight threshold
    forces = contact_sensor.data.force  # [B, 2, 3]
    force_mag = torch.linalg.norm(forces, dim=-1)  # [B, 2]
    excess_force = torch.clamp(force_mag - impact_force_thresh, min=0.0)

    # Downward foot velocity at touchdown
    asset: Entity = env.scene[asset_cfg.name]
    foot_vz = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]  # [B, 2]
    down_vel = torch.clamp(-foot_vz - impact_vel_thresh, min=0.0)

    # Combined touchdown penalty
    touchdown_cost = (excess_force + vel_weight * torch.square(down_vel)) * first_contact.float()
    return -torch.sum(touchdown_cost, dim=1)


def stairs_riser_trip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_friction_ratio: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot collision with vertical stair risers (stubbing/tripping).

    On horizontal treads, contact normal is vertical: F_z dominates and |F_xy| <= mu * F_z.
    On a vertical riser, contact normal is horizontal: F_xy dominates while F_z is near zero.
    Penalizes horizontal shear forces in excess of max_friction_ratio * F_z.

    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    from mjlab.sensor import ContactSensor
    contact_sensor: ContactSensor = env.scene[sensor_name]
    forces = contact_sensor.data.force  # [B, 2, 3] in global frame under reduce="netforce"
    f_xy = torch.linalg.norm(forces[:, :, :2], dim=-1)  # [B, 2]
    f_z = torch.clamp(forces[:, :, 2], min=0.0)  # [B, 2]

    # In contact mask
    in_contact = (contact_sensor.data.found > 0).float()
    if in_contact.dim() == 1:
        in_contact = in_contact.unsqueeze(1)

    riser_excess = torch.clamp(f_xy - max_friction_ratio * f_z, min=0.0)
    trip_cost = torch.sum(riser_excess * in_contact, dim=1)
    return -trip_cost


def stairs_sole_flat_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sole tilt during stance on stair treads.

    Projects gravity into the foot site frame. When flat on a tread, g_site_xy = 0.
    Gated by contact so swing feet are free to articulate.
    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    import torch.nn.functional as F
    from mjlab.sensor import ContactSensor

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)  # (B, 3)
    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, 2, 4)

    per_foot = torch.zeros(env.num_envs, foot_quats.shape[1], device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)

    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact_time = contact_sensor.data.current_contact_time
    if contact_time is not None:
        per_foot = per_foot * (contact_time > 0.0).float()

    return -per_foot.sum(dim=1)


def stairs_relative_height_progress(
    env: ManagerBasedRlEnv,
    ceiling: float = 0.115,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential-based relative height shaping on stairs: Delta min(z_trunk_rel, ceiling).

    CRITICAL FOR STAIRS: Standard height_progress uses (z_trunk - z_origin), which
    saturates above step 1 because step risers add elevation.
    This function uses trunk height RELATIVE to the lowest foot site:
        z_rel = z_trunk - min(z_left_foot, z_right_foot)

    Properties:
    - Invariant to which stair step the robot is standing or lying on.
    - Climbing stairs leaves z_rel constant (~0.117 m), so climbing cannot farm recovery!
    - Rising out of a crouch or prone state pays Delta z_rel > 0.
    - Holding any pose pays exactly 0.0.

    Returns Delta Phi in [-0.05, 0.05] -> weight with positive sign.
    """
    asset: Entity = env.scene[asset_cfg.name]
    trunk_z = asset.data.root_link_pos_w[:, 2]  # [B]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, 2]
    min_foot_z = torch.min(foot_z, dim=1).values  # [B]

    z_rel = torch.nan_to_num(trunk_z - min_foot_z, nan=0.0)
    pot = torch.clamp(z_rel, max=ceiling)

    if not hasattr(env, "_stairs_rel_height_pot_prev"):
        env._stairs_rel_height_pot_prev = pot.clone()

    fresh = env.episode_length_buf <= 1
    env._stairs_rel_height_pot_prev[fresh] = pot[fresh]

    delta = pot - env._stairs_rel_height_pot_prev
    env._stairs_rel_height_pot_prev = pot.clone()
    return delta
```

---

## 5. Term Reuse, Staging & Curriculum Recommendations

### 5.1 Reused Existing Terms in `mdp.py`
The stairs task builds on the proven velocity & velstand stack:

| Existing Term | Function in `mdp.py` | Role in Stairs Task | Sign / Weight |
|---|---|---|---|
| `upright_progress` | `microduck_mdp.upright_progress` | Orientation shaping: $\Delta \cos(\text{tilt})$ per step | Positive ($+5.0$) |
| `feet_air_time_upright` | `microduck_mdp.feet_air_time_upright` | Gait pacing; zeroed while fallen | Positive ($+0.125$) |
| `action_rate_l2_fallen_scaled` | `microduck_mdp.action_rate_l2_fallen_scaled` | Action smoothness; scaled $\times 0.1$ when fallen | Negative ($-1.0$) |
| `joint_torque_rate_l2_fallen_scaled` | `microduck_mdp.joint_torque_rate_l2_fallen_scaled` | Gear shock mitigation; scaled when fallen | Negative ($-2\times 10^{-3}$) |
| `servo_acc_spike_penalty` | `microduck_mdp.servo_acc_spike_penalty` | Protects XL330 gears from reflected rotor impact | Negative ($-10^{-3}$) |
| `servo_stall_penalty` | `microduck_mdp.servo_stall_penalty` | Penalizes servos pinned against stair steps | Negative ($-0.05$) |
| `body_impact_cost` | `microduck_mdp.body_impact_cost` | Penalizes head and trunk slamming on steps | Negative ($-0.01$) |
| `gentle_rise` | `microduck_mdp.trunk_vertical_accel_penalty` | Self-negating $|a_z|$ penalty; prevents ballistic pop-ups | Positive ($+0.005$) |
| `head_pose_bias` | `microduck_mdp.head_pose_bias_penalty` | Gated head droop penalty; active only when upright | Positive ($+0.5$) |

### 5.2 Progressive Stair Height Curriculum Staging
Requirement R1 mandates progressive stair height curriculum staging:
$$\text{Stage 1: } 2.5\text{ cm} \longrightarrow \text{Stage 2: } 3.0\text{ cm} \longrightarrow \text{Stage 3: } 3.5\text{ cm}$$

**Pacing Schedule (at 4096 envs, 24 steps/env)**:
- **Phase 1 (Iterations 0 – 1500)**:
  - Step height: $2.5$ cm ($0.025$ m).
  - Clearance target: $h_{target} = 0.032$ m.
  - Objective: Discover stable stepping cadence, establish gait rhythm, settle initial balance on shallow risers.
- **Phase 2 (Iterations 1500 – 3500)**:
  - Step height: $3.0$ cm ($0.030$ m).
  - Clearance target: $h_{target} = 0.038$ m.
  - Landing damping: full weight; introduce topple perturbations ($\pm 0.6$ m/s).
  - Objective: Consolidation of knee flexion/extension clearance, toe stubbing avoidance.
- **Phase 3 (Iterations 3500 – 6000)**:
  - Step height: $3.5$ cm ($0.035$ m, ~15% of robot height).
  - Clearance target: $h_{target} = 0.045$ m.
  - Recovery economics active: topple pushes up to $\pm 1.2$ m/s, prone/crouch spawns on stairs.
  - Objective: Max torque clearance, automatic get-up recovery on step treads.

---

## 6. Independent Verification & Test Execution

### Test Suite Cleanliness:
The test suite was run and verified:
```bash
PYTHONPATH="" uv run --with pytest pytest -p no:launch_testing tests/
```
Result: **231 passed, 1 skipped in 13.07s (100% pass rate)**.

### Acceptance Guardrails for the Implementer:
1. `tests/test_stairs_cfg.py` must verify:
   - All penalty terms return $\le 0.0$ and have positive weights.
   - All cost terms return $\ge 0.0$ and have negative weights.
   - `_servo_joint_ids` correctly resolves to 14 joints on plain and backlash models.
   - Sensor names (`feet_ground_contact`, `foot_height_scan`) exist and match patterns.
2. Invariant Smoke Test:
   - `uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`
   - Must run 5 iterations with 0 NaNs, compute all reward terms cleanly, and export a 61D ONNX model with baked normalizer.
