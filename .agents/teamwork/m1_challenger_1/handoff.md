# Handoff Report — Milestone M1 Stairs MDP Adversarial Verification

**Author**: M1 Challenger 1 (`critic`, `specialist`)  
**Verdict**: **APPROVE**  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_1`  
**Date**: 2026-10-11  

---

## 1. Observation

1. **Target Implementation**:
   - Location: `src/mjlab_microduck/tasks/mdp.py` lines 7700–7875.
   - Functions implemented:
     - Line 7704: `stairs_foot_clearance_penalty`
     - Line 7744: `stairs_foot_landing_damping_penalty`
     - Line 7776: `stairs_riser_trip_penalty`
     - Line 7806: `stairs_sole_flat_penalty`
     - Line 7838: `stairs_relative_height_progress`
     - Line 7700: `_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` with `site_ids = [1, 6]`.

2. **Test Suites Executed**:
   - Base MDP test suite (`tests/test_stairs_mdp.py`):
     - Command: `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_mdp.py`
     - Result: 11 passed in 10.15s.
   - Adversarial stress test suite (`tests/test_stairs_adversarial.py`):
     - Command: `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_adversarial.py -v`
     - Result: 11 passed in 10.91s across CPU and CUDA (`test_real_scale_batch_and_device[cpu] PASSED`, `test_real_scale_batch_and_device[cuda] PASSED`).
   - Full repository regression suite:
     - Command: `PYTHONPATH="" uv run --with pytest pytest tests/ -k "not test_aarch64_cuda_torch"`
     - Result: 300 passed, 5 skipped, 6 deselected in 16.99s. Zero regressions.

3. **Empirical Boundary & Invariant Observations**:
   - **Sign Oracle**: Across 10,000 Monte Carlo evaluations with extreme randomized inputs ($v \in [-100, 100]\text{ m/s}$, forces $\in [-100, 500]\text{ N}$, arbitrary 3D quaternions), every penalty strictly satisfied $\max(p) \le 0.0$.
   - **Kinematic Stress**: Under supersonic velocities ($v_{xy} = 10,000\text{ m/s}$, $v_z = \pm 10,000\text{ m/s}$), penalties scaled monotonically without numerical overflow or positive values. Zero horizontal speed produced exactly $0.0$.
   - **Contact Force Stress**: Under normal crush forces ($F_z = 100,000\text{ N}$) and pure horizontal riser shear impacts ($F_{xy} = 10,000\text{ N}, F_z = 0\text{ N}$), output was strictly non-positive. Negative normal forces ($F_z = -1,000\text{ N}$) clamped to $0.0$.
   - **Gravitational Stress**: Tested inverted gravity ($\vec{g} = [0, 0, +9.81]$), zero gravity ($\vec{g} = [0, 0, 0]$), and high gravity ($\vec{g} = [0, 0, -10000]$). No NaNs were produced.
   - **NaN/Inf Immunity**: Injected NaNs into `heights`, `forces`, `lin_vel_w`, and `root_link_pos_w`; all outputs remained finite numbers due to `torch.nan_to_num`.
   - **Relative Height Progress Lifecycle**:
     - Reset steps 0 and 1 strictly return $0.0$ via `env.episode_length_buf <= 1` synchronization.
     - Stair climbing with identical trunk and foot rise strictly yields $0.0$ (elevation invariance confirmed).
     - Recovery from inverted posture ($z_{rel} = -0.18$) pays positive progress ($+0.295$).
     - Multi-step rising satisfies the telescopic identity $\sum_{t=0}^T \Delta \Phi_t = \Phi_T - \Phi_0 = 0.115$.

---

## 2. Logic Chain

1. **Sign Safety Invariant (AGENTS.md rule)**:
   - Observation 3 confirms all penalty functions return values $\le 0.0$ under normal, extreme, and randomized conditions.
   - In `stairs_foot_clearance_penalty`: `clearance_deficit >= 0` and `moving >= 0` $\implies \text{cost} \ge 0 \implies \text{return} = -\text{cost} \le 0$.
   - In `stairs_foot_landing_damping_penalty`: $\text{excess\_force} \ge 0$ and $\text{down\_vel}^2 \ge 0 \implies \text{return} \le 0$.
   - In `stairs_riser_trip_penalty`: $\text{riser\_excess} \ge 0 \implies \text{return} \le 0$.
   - In `stairs_sole_flat_penalty`: $proj_{xy}^2 \ge 0 \implies \text{return} \le 0$.
   - Therefore, self-negating penalties with positive reward weights in training will never reward violations or permit reward-farming exploits.

2. **NaN and Crash Immunity**:
   - Observation 3 shows all 5 functions wrap external sensor tensors in `torch.nan_to_num`.
   - `F.normalize(asset.data.gravity_vec_w, dim=-1)` handles zero gravity without division-by-zero NaNs due to default $\epsilon = 10^{-12}$.
   - Missing `current_contact_time` cleanly degrades to `found` flag in `stairs_sole_flat_penalty`.
   - Absence of `command_manager` in evaluation/standalone scenes is handled cleanly in `stairs_foot_clearance_penalty`.

3. **Physical & Potential Shaping Correctness**:
   - `stairs_relative_height_progress` uses $z_{trunk} - \min(z_{left}, z_{right})$, which is invariant to global elevation shifts when ascending consecutive stairs.
   - Reset transients (steps $\le 1$) are zeroed, preventing artificial reward spikes upon spawn.
   - Telescopic summation holds, preventing cyclic reward pumping.

4. **Production Scalability**:
   - Both CPU and CUDA execution at $B=4096$ pass identical shape and numerical checks without memory or device mismatch errors.

---

## 3. Caveats

1. **Sensor Config Prerequisite for M2**:
   - `stairs_foot_landing_damping_penalty` relies on `contact_sensor.compute_first_contact(dt=env.step_dt)`. In `mjlab`, this method raises a `RuntimeError` if the sensor lacks `track_air_time=True`. The M2 task config (`microduck_stairs_env_cfg.py`) must configure `track_air_time=True` on `feet_ground_contact`.
2. **Clearance 2D Norm**:
   - `stairs_foot_clearance_penalty` calculates horizontal speed via `torch.linalg.norm(foot_vel_xy, dim=-1)`. Lateral swing velocities exceeding 0.02 m/s while foot height is below target are penalized. This is desirable for avoiding stair edge snags, but task designers should note that rapid pure-lateral sidestepping without lifting feet will be penalized.
3. **Sole Flat Inverted Orientation**:
   - For an inverted robot ($180^\circ$ roll), $proj_{xy} = 0$, so tilt penalty is 0. However, upside-down robots do not have stance contact (`contact_time = 0`), which properly zeros out the penalty.
4. **Relative Height Lower Bound**:
   - `stairs_relative_height_progress` clamps only the ceiling (`max=ceiling`). If feet kick high above the trunk ($z_{rel} < 0$), potential goes negative and rising yields $> 0.115$ delta. Because discount factor $\gamma < 1$, oscillating loses return, but if extreme leg thrashing is observed in early training, an optional lower clamp (`min=-0.05`) could be added in M2.

---

## 4. Conclusion

**Verdict: APPROVE**

The 5 stairs MDP functions in `src/mjlab_microduck/tasks/mdp.py` are robust, mathematically sound, free of NaNs/Infs under extreme boundary conditions, strictly compliant with Microduck reward sign conventions, invariant to stair elevation, and performant at real training scale ($B=4096$) on GPU. All acceptance criteria for Milestone M1 MDP terms are met.

---

## 5. Verification Method

To independently reproduce the adversarial verification:

```bash
# 1. Run unit MDP test suite
PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_mdp.py -v

# 2. Run adversarial stress test suite (CPU and CUDA)
PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_adversarial.py -v

# 3. Run full regression test suite
PYTHONPATH="" uv run --with pytest pytest tests/ -k "not test_aarch64_cuda_torch"
```

**Invalidation Conditions**:
- Any test in `tests/test_stairs_adversarial.py` fails.
- Any penalty function returns a value $> 0.0$.
- Any function produces `NaN` or `Inf` under randomized or extreme inputs.
- `stairs_relative_height_progress` produces non-zero rewards during stair climbing in constant standing posture.
