# Specification Report: High Stairs MDP Terms & Standalone Scene

**Author**: M1 Spec Miner (`m1_spec_miner_1`)  
**Date**: 2026-10-11  
**Target Files**:
1. `src/mjlab_microduck/tasks/mdp.py` (5 custom stairs MDP functions)
2. `src/mjlab_microduck/robot/microduck/scene_stairs.xml` (Standalone 3-flight stairs scene)
3. `tests/test_scene_stairs.py` (Unit verification suite for scene)

---

## 1. Executive Summary

This report establishes the exact line-level specification, type signatures, tensor contracts, sensor dependencies, and mathematical implementations for Milestone M1 of the Microduck High Stairs project (`Mjlab-StairsClimb-MicroDuck`).

The specifications strictly adhere to repository invariants in `AGENTS.md`:
- Self-negating penalties return $\le 0.0$ and are paired with positive weights ($w > 0$).
- Potential-based progress terms return step differences $\Delta \Phi$ invariant to absolute elevation.
- Actuator kinematics and sensor lookups index through `_servo_joint_ids` / entity site IDs.
- Dedicated standalone scene `scene_stairs.xml` uses `robot_groundcontact.xml` with zero modifications to existing scenes.

---

## 2. Specification for `src/mjlab_microduck/tasks/mdp.py`

### 2.1 File State & Exact Integration Point
- **Existing File Length**: 7,687 lines (`FetchTargetCommandCfg` concludes at line 7686, followed by trailing newline).
- **Exact Line Number for Insertion**: Line 7688 (appended at bottom of file).
- **Section Heading Banner**:
  ```python
  # ==============================================================================
  # Stairs Climbing Task — Clearance, Landing Damping, Contact & Recovery Terms
  # ==============================================================================
  ```
- **Module-Level Imports Present at Top of `mdp.py`**:
  - `math`, `os`, `pathlib.Path`, `numpy as np`, `torch`
  - `from typing import TYPE_CHECKING, Optional`
  - `import mujoco`
  - `from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv`
  - `from mjlab.managers.scene_entity_config import SceneEntityCfg`
  - `from mjlab.entity import Entity`
  - `_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")` (line 219)
- **Inline Imports Used Inside the 5 Functions**:
  - `from mjlab.sensor import ContactSensor` (for contact and air time tracking)
  - `from mjlab.utils.lab_api.math import quat_apply_inverse` (for sole orientation projection)
  - `import torch.nn.functional as F` (for vector normalization)

---

### 2.2 Function 1: `stairs_foot_clearance_penalty`

#### Purpose & Physics
Penalizes clearance shortfall (deficit below `target_height`) during forward swing on stairs. Unlike mjlab's stock symmetric `feet_clearance` (which penalizes high swings $> h_{target}$), this penalty is strictly **one-sided**: lifting higher than target is completely free ($0.0$ penalty), allowing high-stepping over 3.5 cm step lips. Gated on forward site velocity and active velocity command.

#### Exact Type Signature
```python
def stairs_foot_clearance_penalty(
    env: ManagerBasedRlEnv,
    target_height: float = 0.035,
    height_sensor_name: str = "foot_height_scan",
    command_name: str | None = "twist",
    command_threshold: float = 0.01,
    vel_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
```

#### Tensor Math & Operations
1. `height_sensor = env.scene[height_sensor_name]` $\implies$ `foot_height = height_sensor.data.heights` of shape `[B, 2]`.
2. `asset: Entity = env.scene[asset_cfg.name]` $\implies$ `foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]` of shape `[B, 2, 2]`.
3. `vel_norm = torch.linalg.norm(foot_vel_xy, dim=-1)` of shape `[B, 2]`.
4. Deficit: `clearance_deficit = torch.clamp(target_height - foot_height, min=0.0)`.
5. Motion gate: `moving = torch.clamp(vel_norm - vel_threshold, min=0.0)`.
6. Raw cost: `cost = torch.sum(clearance_deficit * moving, dim=1)` of shape `[B]`.
7. Command gate (if `command_name` specified):
   - `cmd = env.command_manager.get_command(command_name)`
   - `cmd_active = (torch.linalg.norm(cmd[:, :2], dim=1) > command_threshold).float()`
   - `cost = cost * cmd_active`
8. Return value: `-cost` (values $\le 0.0$, self-negating penalty $\implies$ weight $> 0$).

---

### 2.3 Function 2: `stairs_foot_landing_damping_penalty`

#### Purpose & Physics
Penalizes excessive contact force and downward vertical velocity at foot touchdown (`first_contact`). Protects Dynamixel XL330 plastic gear train (288:1 ratio) from impulse spikes on rigid step treads and damps foot bounce. Normal static weight bearing ($\le 8$ N) and continuous stance are completely unpenalized.

#### Exact Type Signature
```python
def stairs_foot_landing_damping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    impact_force_thresh: float = 8.0,
    impact_vel_thresh: float = 0.08,
    vel_weight: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
```

#### Tensor Math & Operations
1. `from mjlab.sensor import ContactSensor`
2. `contact_sensor: ContactSensor = env.scene[sensor_name]`
3. `first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)` of shape `[B, 2]` bool.
4. `forces = contact_sensor.data.force` of shape `[B, 2, 3]`.
   - `force_mag = torch.linalg.norm(forces, dim=-1)` of shape `[B, 2]`.
   - `excess_force = torch.clamp(force_mag - impact_force_thresh, min=0.0)`.
5. `asset: Entity = env.scene[asset_cfg.name]`.
   - `foot_vz = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]` of shape `[B, 2]`.
   - `down_vel = torch.clamp(-foot_vz - impact_vel_thresh, min=0.0)`.
6. Touchdown cost per foot:
   `touchdown_cost = (excess_force + vel_weight * torch.square(down_vel)) * first_contact.float()`.
7. Return value: `-torch.sum(touchdown_cost, dim=1)` of shape `[B]` (values $\le 0.0$).

---

### 2.4 Function 3: `stairs_riser_trip_penalty`

#### Purpose & Physics
Penalizes foot collisions against vertical stair riser faces (stubbing/tripping). On horizontal step treads, contact reaction forces are predominantly vertical ($F_z > 0$, $\|F_{xy}\| \le \mu F_z$). On a vertical riser face, the contact normal is horizontal, generating large horizontal shear forces while $F_z \approx 0$. Penalizes horizontal force in excess of `max_friction_ratio * F_z`.

#### Exact Type Signature
```python
def stairs_riser_trip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_friction_ratio: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
```

#### Tensor Math & Operations
1. `from mjlab.sensor import ContactSensor`
2. `contact_sensor: ContactSensor = env.scene[sensor_name]`
3. `forces = contact_sensor.data.force` of shape `[B, 2, 3]` (global frame under `reduce="netforce"`).
   - If `forces is None`: return `torch.zeros(env.num_envs, device=env.device)`.
   - `f_xy = torch.linalg.norm(forces[:, :, :2], dim=-1)` of shape `[B, 2]`.
   - `f_z = torch.clamp(forces[:, :, 2], min=0.0)` of shape `[B, 2]`.
4. In-contact mask:
   - `in_contact = (contact_sensor.data.found > 0).float()`.
   - If `in_contact.dim() == 1`: `in_contact = in_contact.unsqueeze(1)`.
5. Riser shear excess:
   - `riser_excess = torch.clamp(f_xy - max_friction_ratio * f_z, min=0.0)`.
   - `trip_cost = torch.sum(riser_excess * in_contact, dim=1)`.
6. Return value: `-trip_cost` of shape `[B]` (values $\le 0.0$).

---

### 2.5 Function 4: `stairs_sole_flat_penalty`

#### Purpose & Physics
Penalizes foot sole tilt during stance on stair treads, enforcing flat foot placement and preventing edge-balancing. Projects the normalized gravity vector into each foot site's coordinate frame. When the sole is flat on the horizontal tread, local gravity is $[0, 0, -1]^T$ and horizontal components are zero. Gated on ground contact so the airborne swing foot is free to articulate.

#### Exact Type Signature
```python
def stairs_sole_flat_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
```

#### Tensor Math & Operations
1. `from mjlab.utils.lab_api.math import quat_apply_inverse`
2. `import torch.nn.functional as F`
3. `from mjlab.sensor import ContactSensor`
4. `asset: Entity = env.scene[asset_cfg.name]`
5. `gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)` of shape `[B, 3]`.
6. `foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]` of shape `[B, 2, 4]`.
7. For each foot $i \in \{0, 1\}$:
   - `proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)` of shape `[B, 3]`.
   - `per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)`.
8. Contact gating:
   - `contact_sensor: ContactSensor = env.scene[sensor_name]`
   - `contact_time = contact_sensor.data.current_contact_time`
   - If `contact_time is not None`: `per_foot = per_foot * (contact_time > 0.0).float()`
   - Else: `found = (contact_sensor.data.found > 0).float()`; `if found.dim() == 1: found = found.unsqueeze(1)`; `per_foot = per_foot * found`.
9. Return value: `-per_foot.sum(dim=1)` of shape `[B]` (values $\le 0.0$).

---

### 2.6 Function 5: `stairs_relative_height_progress`

#### Purpose & Physics
Potential-based trunk height shaping invariant to stair elevation:
$$\Phi(t) = \min(z_{rel}(t), \text{ceiling}), \quad z_{rel} = z_{trunk} - \min(z_{left\_foot}, z_{right\_foot})$$
$$\Delta \Phi = \Phi(t) - \Phi(t-1)$$
Standard `height_progress` ($z_{trunk} - z_{origin}$) saturates on step 2 ($z = 0.06$ m) because step risers add elevation, causing a fallen robot on step 3 to have $z_{trunk} = 0.14 > 0.115$ m with zero gradient to stand up. Relative height progress is invariant to which step the robot occupies:
- Standing upright on ANY step: $z_{rel} \approx 0.117$ m (saturated at ceiling $0.115$).
- Prone/fallen on ANY step: $z_{rel} \approx 0.040$ m.
- Climbing stairs: feet and trunk rise together, so $z_{rel}$ remains constant and $\Delta \Phi = 0.0$ (unfarmable by climbing).
- Standing up from a crouch/fall: pays $\Delta z_{rel} > 0.0$. Holding pays $0.0$.

#### Exact Type Signature
```python
def stairs_relative_height_progress(
    env: ManagerBasedRlEnv,
    ceiling: float = 0.115,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
```

#### Tensor Math & Operations
1. `asset: Entity = env.scene[asset_cfg.name]`.
2. `trunk_z = asset.data.root_link_pos_w[:, 2]` of shape `[B]`.
3. `foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]` of shape `[B, 2]`.
4. `min_foot_z = torch.min(foot_z, dim=1).values` of shape `[B]`.
5. `z_rel = torch.nan_to_num(trunk_z - min_foot_z, nan=0.0)` of shape `[B]`.
6. `pot = torch.clamp(z_rel, max=ceiling)` of shape `[B]`.
7. Memory persistence & reset protection:
   ```python
   if not hasattr(env, "_stairs_rel_height_pot_prev"):
       env._stairs_rel_height_pot_prev = pot.clone()
   fresh = env.episode_length_buf <= 1
   env._stairs_rel_height_pot_prev[fresh] = pot[fresh]
   delta = pot - env._stairs_rel_height_pot_prev
   env._stairs_rel_height_pot_prev = pot.clone()
   return delta
   ```
8. Return value: `delta` of shape `[B]`.

---

### 2.7 Complete Code Insertion Block for `mdp.py`

```python
# ==============================================================================
# Stairs Climbing Task — Clearance, Landing Damping, Contact & Recovery Terms
# ==============================================================================


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

    Protects XL330 gear trains and damps footfall bounce on rigid stair treads.
    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    from mjlab.sensor import ContactSensor
    contact_sensor: ContactSensor = env.scene[sensor_name]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, 2] bool

    # Contact force magnitude in excess of body weight threshold
    forces = contact_sensor.data.force  # [B, 2, 3]
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
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
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
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
    else:
        found = (contact_sensor.data.found > 0).float()
        if found.dim() == 1:
            found = found.unsqueeze(1)
        per_foot = per_foot * found

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

## 3. Specification for `src/mjlab_microduck/robot/microduck/scene_stairs.xml`

### 3.1 Structural Blueprint
- **Target File**: `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
- **Included Model**: `<include file="robot_groundcontact.xml" />` (essential for full body collision during topples and standups).
- **Contact Dynamics Parameters**:
  - `condim="3"` (3D frictional contact)
  - `friction="1.2 0.01 0.001"` (1.2 tangential sliding friction for high bipedal grip)
  - `solref="0.02 1.0"` (rigid constraint stabilization, no sponge bounce)
  - `solimp="0.9 0.95 0.001 0.5 2"` (high constraint impedance)
- **Geometry Construction Rules**:
  - Solid floor-anchored construction: $p_z = s_z = H_{top} / 2$ (boxes rest on floor $z=0$, eliminating hollow pockets and contact tunneling).
  - Width: $s_y = 0.25$ m (total lateral span $0.50$ m), centered at $y = 0.0$.
  - Gapless continuity: $x_{max}^{(k)} = x_{min}^{(k+1)}$ across all adjacent boxes along +X.

---

### 3.2 Coordinate & Geometry Proof Table

| Geom Name | $x_{min}$ | $x_{max}$ | Center $p_x$ | Half $s_x$ | Top $H_{top}$ | Half $s_z$ | Center $p_z$ | Riser $\Delta z$ | Material |
|---|---|---|---|---|---|---|---|---|---|
| `floor` | $-\infty$ | $+\infty$ | $0.00$ | $0.00$ | $0.000$ | $0.050$ | $0.000$ | — | `groundplane` |
| **Flight 1: 2.5cm Risers** | | | | | | | | | |
| `stair_1_1` | $0.30$ | $0.42$ | $0.36$ | $0.06$ | $0.025$ | $0.0125$ | $0.0125$ | $+2.5$ cm | `mat_stair_25` |
| `stair_1_2` | $0.42$ | $0.54$ | $0.48$ | $0.06$ | $0.050$ | $0.0250$ | $0.0250$ | $+2.5$ cm | `mat_stair_25` |
| `stair_1_3` | $0.54$ | $0.66$ | $0.60$ | $0.06$ | $0.075$ | $0.0375$ | $0.0375$ | $+2.5$ cm | `mat_stair_25` |
| `landing_1` | $0.66$ | $1.02$ | $0.84$ | $0.18$ | $0.075$ | $0.0375$ | $0.0375$ | $0.0$ cm | `mat_landing_1` |
| **Flight 2: 3.0cm Risers** | | | | | | | | | |
| `stair_2_1` | $1.02$ | $1.14$ | $1.08$ | $0.06$ | $0.105$ | $0.0525$ | $0.0525$ | $+3.0$ cm | `mat_stair_30` |
| `stair_2_2` | $1.14$ | $1.26$ | $1.20$ | $0.06$ | $0.135$ | $0.0675$ | $0.0675$ | $+3.0$ cm | `mat_stair_30` |
| `stair_2_3` | $1.26$ | $1.38$ | $1.32$ | $0.06$ | $0.165$ | $0.0825$ | $0.0825$ | $+3.0$ cm | `mat_stair_30` |
| `landing_2` | $1.38$ | $1.74$ | $1.56$ | $0.18$ | $0.165$ | $0.0825$ | $0.0825$ | $0.0$ cm | `mat_landing_2` |
| **Flight 3: 3.5cm Risers** | | | | | | | | | |
| `stair_3_1` | $1.74$ | $1.86$ | $1.80$ | $0.06$ | $0.200$ | $0.1000$ | $0.1000$ | $+3.5$ cm | `mat_stair_35` |
| `stair_3_2` | $1.86$ | $1.98$ | $1.92$ | $0.06$ | $0.235$ | $0.1175$ | $0.1175$ | $+3.5$ cm | `mat_stair_35` |
| `stair_3_3` | $1.98$ | $2.10$ | $2.04$ | $0.06$ | $0.270$ | $0.1350$ | $0.1350$ | $+3.5$ cm | `mat_stair_35` |
| `landing_summit` | $2.10$ | $2.58$ | $2.34$ | $0.24$ | $0.270$ | $0.1350$ | $0.1350$ | $0.0$ cm | `mat_summit` |

---

### 3.3 Site Beacons & Visual Materials
- **Visual Materials**:
  - `mat_stair_25`: `rgba="0.32 0.48 0.65 1.0"` (Slate Blue)
  - `mat_landing_1`: `rgba="0.22 0.62 0.42 1.0"` (Emerald Green)
  - `mat_stair_30`: `rgba="0.68 0.52 0.26 1.0"` (Warm Ochre)
  - `mat_landing_2`: `rgba="0.18 0.58 0.54 1.0"` (Deep Teal)
  - `mat_stair_35`: `rgba="0.78 0.34 0.26 1.0"` (Terracotta Coral)
  - `mat_summit`: `rgba="0.88 0.72 0.20 1.0"` (Champion Gold)
  - `target_site_mat`: `rgba="1.0 0.4 0.1 0.7"` (Translucent Navigation Beacon)
- **Site Beacons**:
  - `site_landing_1`: pos=`"0.84 0 0.080"`, type=`"cylinder"`, size=`"0.03 0.005"`, material=`"target_site_mat"`
  - `site_landing_2`: pos=`"1.56 0 0.170"`, type=`"cylinder"`, size=`"0.03 0.005"`, material=`"target_site_mat"`
  - `site_summit`: pos=`"2.34 0 0.275"`, type=`"cylinder"`, size=`"0.04 0.005"`, material=`"target_site_mat"`

---

### 3.4 Keyframes Specification
Each keyframe specifies 21 qpos elements ($[x, y, z, q_w, q_x, q_y, q_z]$ + 14 joint positions) and 14 ctrl targets:
- `INIT`: Flat spawn at $(0, 0, 0.12)$ with all zeros.
- `STAND`: Nominal `STAND2` pose on floor at $(0, 0, 0.125)$.
- `SPAWN_STAIRS_25`: Step 1_2 at $(0.48, 0, 0.175)$.
- `SPAWN_LANDING_1`: Landing 1 at $(0.84, 0, 0.200)$.
- `SPAWN_STAIRS_30`: Step 2_2 at $(1.20, 0, 0.260)$.
- `SPAWN_LANDING_2`: Landing 2 at $(1.56, 0, 0.290)$.
- `SPAWN_STAIRS_35`: Step 3_2 at $(1.92, 0, 0.360)$.
- `SPAWN_SUMMIT`: Summit at $(2.34, 0, 0.395)$.

---

### 3.5 Complete XML Content for `scene_stairs.xml`

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

## 4. Verification Suite Specification for `tests/test_scene_stairs.py`

To ensure zero regressions and rigorous validation of Milestone M1 deliverables, the builder should implement `tests/test_scene_stairs.py` containing:

1. `test_scene_stairs_compilation()`: Loads `scene_stairs.xml` via `mujoco.MjModel.from_xml_path()`, asserts model compiles with 0 errors, has 14 actuated DOFs, and includes `robot_groundcontact.xml` geoms (`np_f970`, `leg`, etc.).
2. `test_scene_stairs_geometries_exist()`: Confirms all 12 step and landing box geoms (`stair_1_1` through `stair_3_3`, `landing_1`, `landing_2`, `landing_summit`) exist by name in the compiled model.
3. `test_scene_stairs_riser_heights()`: Computes $z_{top} = p_z + s_z$ for each step:
   - Flight 1: step risers equal $0.025$ m $\pm 10^{-6}$.
   - Flight 2: step risers equal $0.030$ m $\pm 10^{-6}$.
   - Flight 3: step risers equal $0.035$ m $\pm 10^{-6}$.
4. `test_scene_stairs_gapless_continuity()`: Asserts for all adjacent pairs $(k, k+1)$ that $p_x^{(k)} + s_x^{(k)} = p_x^{(k+1)} - s_x^{(k+1)} \pm 10^{-6}$.
5. `test_scene_stairs_contact_parameters()`: Asserts each stair geom has `condim == 3` and tangential friction $\ge 1.0$.
6. `test_scene_stairs_keyframes()`: Asserts keyframes `INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT` exist and have valid qpos length (21) and ctrl length (14).

---

## 5. Formal Discovery Tables

### Features Discovered
| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | MDP Clearance | `stairs_foot_clearance_penalty` | One-sided foot clearance deficit penalty during forward swing on stairs | `env`, `target_height`, `height_sensor_name`, `command_name`, `vel_threshold` | `torch.Tensor [B]` ($\le 0.0$) | Clamped to 0.0 if above target or stationary; NaN-safe | Survey 2 & codebase probing |
| 2 | MDP Damping | `stairs_foot_landing_damping_penalty` | Touchdown penalty on excess impact force (>8N) and downward normal velocity | `env`, `sensor_name`, `impact_force_thresh`, `impact_vel_thresh`, `vel_weight` | `torch.Tensor [B]` ($\le 0.0$) | Evaluated only on `first_contact`; 0.0 if forces is None | Survey 2 & `contact_sensor.py` |
| 3 | MDP Anti-Trip | `stairs_riser_trip_penalty` | Penalty on horizontal shear forces exceeding $\mu F_z$ (riser stubbing/tripping) | `env`, `sensor_name`, `max_friction_ratio` | `torch.Tensor [B]` ($\le 0.0$) | Zero if $F_{xy} \le \mu F_z$ or not in contact | Survey 2 & contact mechanics |
| 4 | MDP Contact | `stairs_sole_flat_penalty` | Enforces sole parallel alignment on treads during stance | `env`, `sensor_name`, `asset_cfg` | `torch.Tensor [B]` ($\le 0.0$) | Gated by ground contact; swing foot free to articulate | Survey 2 & `feet_flat_penalty` |
| 5 | MDP Recovery | `stairs_relative_height_progress` | Potential-based trunk height shaping relative to lowest foot ($\Delta \min(z_{rel}, 0.115)$) | `env`, `ceiling`, `asset_cfg` | `torch.Tensor [B]` ($\Delta \Phi$) | Episode reset masked; constant during stair climbing | Survey 2 & `height_progress` |
| 6 | Scene MJCF | `scene_stairs.xml` | Standalone MuJoCo scene with 3 stair flights (2.5, 3.0, 3.5 cm), landings, materials, and keyframes | MJCF XML parser | Compiled `MjModel` | Raises XML parsing error if invalid | Survey 3 & scene templates |
| 7 | Unit Tests | `test_scene_stairs.py` | CPU unit test suite verifying scene compilation, riser elevations, and continuity | `pytest` | Test assertions pass/fail | Fails if geometry mismatch or missing keyframe | Survey 3 & `tests/` patterns |

### Edge Cases
| # | Feature | Input | Observed Behavior |
|---|---------|-------|-------------------|
| 1 | `stairs_foot_clearance_penalty` | Foot swings above target height ($h > h_{target}$) | `clearance_deficit` clamped to 0.0; penalty is 0.0 (high-stepping is completely free). |
| 2 | `stairs_foot_clearance_penalty` | Robot standing still with zero forward velocity | `moving` clamped to 0.0; penalty is 0.0 (standing still does not incur swing penalty). |
| 3 | `stairs_foot_landing_damping_penalty` | Continuous stance phase (`first_contact == False`) | Masked out by `first_contact`; penalty is 0.0 (normal weight support is unpenalized). |
| 4 | `stairs_riser_trip_penalty` | Pure vertical weight support on flat tread | Horizontal shear $F_{xy} = 0 \implies 0 - \mu F_z \le 0 \implies$ clamped to 0.0. |
| 5 | `stairs_sole_flat_penalty` | Airborne swing foot | `contact_time == 0.0` or `found == 0` $\implies$ penalty is 0.0 (swing foot free to tilt). |
| 6 | `stairs_relative_height_progress` | Episode reset (step $\le 1$) | Masked via `fresh = env.episode_length_buf <= 1` $\implies \Delta \Phi = 0.0$ (no spawn jump). |
| 7 | `stairs_relative_height_progress` | Robot climbing up stairs upright | Feet and trunk elevate together $\implies z_{rel}$ remains $\approx 0.117$ m $\implies \Delta \Phi = 0.0$ (cannot farm recovery). |
| 8 | `scene_stairs.xml` | Boundary transitions between steps | $p_x^{(k)} + s_x^{(k)} = p_x^{(k+1)} - s_x^{(k+1)}$ exactly; zero gap, zero overlap. |
