# Milestone M1 Code Review & Adversarial Stress Test Report

**Reviewer**: M1 Reviewer 1 (Roles: Reviewer, Critic)  
**Date**: 2026-10-11  
**Milestone**: M1 (Core Stairs MDP Terms & Standalone Simulation Scene)  
**Target Files**:
- `src/mjlab_microduck/tasks/mdp.py` (Stairs MDP additions & `_FEET_SITE_CFG`)
- `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
- `tests/test_scene_stairs.py`
- `tests/test_stairs_mdp.py`

---

## 1. Review Summary

**Verdict**: **APPROVE**  
**Integrity Audit**: Clean — No integrity violations, no hardcoded cheats, no dummy facades, no shortcuts, no fabricated outputs. All logic is genuine, mathematically sound, and independently verified.  
**Repository Health**: 100% pass across all 271 unit tests in `microduck_rl` with zero regressions.

---

## 2. Findings

### [Minor] Finding 1: Docstring Nuance in `stairs_foot_clearance_penalty`
- **What**: Docstring mentions "forward swing on stairs", but code evaluates total planar velocity `vel_norm = torch.linalg.norm(foot_vel_xy, dim=-1)`.
- **Where**: `src/mjlab_microduck/tasks/mdp.py:7713-7727`
- **Why**: Evaluates any horizontal planar foot motion ($\sqrt{v_x^2 + v_y^2} > \text{vel\_threshold}$), including lateral and backward movement.
- **Assessment**: Physically advantageous for stairs — scuffing or stubbing during lateral adjustments or turning is just as hazardous as forward stubbing. Harmless nuance, but documentation can be updated in future passes for precision.

### [Minor] Finding 2: Gravity Inversion Corner Case in `stairs_sole_flat_penalty`
- **What**: Penalty measures horizontal projection of gravity in foot frame `torch.square(proj[:, :2])`. An inverted foot ($180^\circ$ flip) produces local gravity $[0, 0, +1]$, where xy components are also zero.
- **Where**: `src/mjlab_microduck/tasks/mdp.py:7824-7826`
- **Why**: Gravity vector projection alone does not distinguish upright $+z$ from inverted $-z$.
- **Assessment**: Non-issue in practice. Dynamixel XL330 ankle pitch and roll joint limits strictly prohibit a $180^\circ$ inversion. Furthermore, full-body tilt termination and height shaping penalize inverted robot states.

---

## 3. Verified Claims

1. **Self-Negating Penalty Sign Convention ($\le 0.0$)**:
   - `stairs_foot_clearance_penalty` returns `-cost <= 0.0`. Verified via analytical code trace and `test_foot_clearance_self_negating_and_bounds`.
   - `stairs_foot_landing_damping_penalty` returns `-torch.sum(touchdown_cost, dim=1) <= 0.0`. Verified via analytical trace and `test_landing_damping_self_negating_and_bounds`.
   - `stairs_riser_trip_penalty` returns `-trip_cost <= 0.0`. Verified via analytical trace and `test_riser_trip_self_negating_and_bounds`.
   - `stairs_sole_flat_penalty` returns `-per_foot.sum(dim=1) <= 0.0`. Verified via analytical trace and `test_sole_flat_self_negating_and_bounds`.
   - All 4 penalties strictly conform to the microduck convention (return $\le 0.0 \implies$ positive configuration weight in `rewards`).
   - `Episode_Reward/<penalty>` wandb metric will be strictly non-positive, preventing reward farming bugs.

2. **Elevation-Invariant Relative Height Shaping**:
   - `stairs_relative_height_progress` computes $\Delta \min(z_{\text{trunk}} - \min(z_{\text{left\_foot}}, z_{\text{right\_foot}}), \text{ceiling})$.
   - On stairs, ascending steps increases both trunk height and foot height identically, maintaining constant relative height (~0.117 m) and yielding exactly 0.0 delta.
   - Climbing stairs cannot farm recovery rewards.
   - Verified via `test_relative_height_progress_elevation_invariance`.

3. **Asynchronous Reset Transient Isolation**:
   - `stairs_relative_height_progress` syncs `_stairs_rel_height_pot_prev` when `episode_length_buf <= 1`.
   - Eliminates spawn drop impulses and handles asynchronous parallel environment resets without cross-talk.
   - Verified via `test_relative_height_progress_lifecycle`.

4. **Dedicated Site Configuration (`_FEET_SITE_CFG`)**:
   - `SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` with `site_ids = [1, 6]`.
   - Verified against all 8 robot model XML variants (`robot_walk.xml`, `robot_groundcontact.xml`, `robot_allcollisions.xml`, and backlash variants): `left_foot` is consistently index 1 and `right_foot` is index 6.
   - Prevents the default `SceneEntityCfg("robot")` from selecting all 7 sites (imu, camera, tof, etc.), avoiding runtime shape mismatch against 2-primary sensors (`foot_height_scan`, `feet_ground_contact`).

5. **Standalone Scene Physics & Geometry (`scene_stairs.xml`)**:
   - Includes `robot_groundcontact.xml` with 14 actuated XL330 servos and correct joint topology (nu=14, njnt=15, nq=21, nv=20).
   - Riser heights: Flight 1 = 3 $\times$ 2.5 cm (elev 7.5 cm), Flight 2 = 3 $\times$ 3.0 cm (elev 16.5 cm), Flight 3 = 3 $\times$ 3.5 cm (elev 27.0 cm), Summit = 27.0 cm.
   - Gapless continuity along X axis from $x = 0.30$ m to $x = 2.58$ m with 0.0 overlap or separation.
   - Solid vertical grounding: Every step box extends down to $z = 0.0$ (`pos_z - size_z == 0.0`), preventing collision tunneling.
   - 3D contact friction: `condim=3`, tangential friction = 1.2, `solref="0.02 1.0"`, `solimp="0.9 0.95 0.001 0.5 2"`.
   - All 8 keyframes (`INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`) compile with valid 21D qpos and 14D ctrl, and settle stably (< 10 deg pitch/roll).

---

## 4. Adversarial Challenge & Stress-Testing

**Overall Risk Assessment**: **LOW**

### Challenge 1: Asynchronous Multi-Environment Step Resets
- **Attack Scenario**: Subsets of 4096 parallel environments reset on different timesteps. Could stale potential buffers in `stairs_relative_height_progress` trigger spurious reward spikes?
- **Analysis**: The check `fresh = env.episode_length_buf <= 1` indexes boolean tensor mask directly: `env._stairs_rel_height_pot_prev[fresh] = pot[fresh]`. Each environment's buffer is independently re-anchored on reset while active environments continue unperturbed.
- **Result**: PASSED. Verified in `test_relative_height_progress_lifecycle` (Step 5).

### Challenge 2: Foot Clearance on Non-Twist or Zero Commands
- **Attack Scenario**: Robot is pushed or perturbed while command is zero (`twist == 0`). Does foot clearance penalty unfairly punish the robot while attempting to stand still?
- **Analysis**: The term checks `cmd_norm > command_threshold`. When `twist` command is near zero, `cmd_active == 0`, completely muting the penalty so stationary balance adjustments are not taxed.
- **Result**: PASSED. Verified via code inspection and test suite.

### Challenge 3: Contact Normal Force Inversion in Riser Trip Penalty
- **Attack Scenario**: Numerical noise in contact sensor reports negative vertical force ($F_z < 0$).
- **Analysis**: `f_z = torch.clamp(forces[:, :, 2], min=0.0)` guarantees non-negative $F_z$. Negative sensor forces cannot inflate `riser_excess = torch.clamp(f_xy - max_friction_ratio * f_z, min=0.0)`.
- **Result**: PASSED.

### Challenge 4: Extreme Impact Spikes on Stair Edges
- **Attack Scenario**: Robot hits step corner, triggering momentary force spikes > 30 N.
- **Analysis**: `stairs_foot_landing_damping_penalty` is gated strictly by `first_contact` (`ContactSensor.compute_first_contact(dt)`). It only penalizes the single instant of touchdown, preventing continuous force penalties during stance.
- **Result**: PASSED.

---

## 5. Coverage Gaps & Unverified Items
- None within Milestone M1 scope.
- Integration with RL training runner (`microduck_stairs_env_cfg.py`) and interactive CPU runner (`scripts/infer_stairs.py`) will be verified in Milestones M2 and M3 respectively.
