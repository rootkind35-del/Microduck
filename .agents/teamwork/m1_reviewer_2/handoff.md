# Handoff Report — Milestone M1 Independent Review

- **Agent**: M1 Reviewer 2 (Reviewer & Adversarial Critic)
- **Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/m1_reviewer_2`
- **Milestone**: M1 (Core MDP Terms & Standalone Simulation Scene)
- **Verdict**: **APPROVE**

---

## 1. Observation

1. **Scene File**: `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
   - Defines 12 box geoms spanning $x \in [0.30, 2.58]\text{ m}$:
     - Flight 1: `stair_1_1` (z: 0.025m), `stair_1_2` (z: 0.050m), `stair_1_3` (z: 0.075m) -> exactly 2.5 cm risers.
     - Landing 1: `landing_1` (z: 0.075m, size: 0.18 x 0.25 x 0.0375).
     - Flight 2: `stair_2_1` (z: 0.105m), `stair_2_2` (z: 0.135m), `stair_2_3` (z: 0.165m) -> exactly 3.0 cm risers.
     - Landing 2: `landing_2` (z: 0.165m, size: 0.18 x 0.25 x 0.0825).
     - Flight 3: `stair_3_1` (z: 0.200m), `stair_3_2` (z: 0.235m), `stair_3_3` (z: 0.270m) -> exactly 3.5 cm risers.
     - Summit: `landing_summit` (z: 0.270m, size: 0.24 x 0.25 x 0.135).
   - Solid grounding: Every geom has $z_{\text{bottom}} = \text{pos}_z - \text{size}_z = 0.000\text{ m}$.
   - Contact physics: `condim="3" friction="1.2 0.01 0.001" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2"`.
   - Keyframes: 8 keyframes (`INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`) with 21D `qpos` and 14D `ctrl`.

2. **MDP Functions**: `src/mjlab_microduck/tasks/mdp.py` (lines 7690–7875)
   - `_FEET_SITE_CFG`: `SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` with explicit `site_ids = [1, 6]` matching robot site indices 1 (`left_foot`) and 6 (`right_foot`).
   - `stairs_foot_clearance_penalty`: One-sided deficit $\max(0, h_{\text{target}} - h_{\text{foot}})$ gated on forward velocity $(\|\mathbf{v}_{xy}\| - v_{\text{thresh}})$ and command activity; uses `torch.nan_to_num`; returns $\le 0$.
   - `stairs_foot_landing_damping_penalty`: Evaluated at touchdown via `contact_sensor.compute_first_contact(dt=env.step_dt)`; penalizes $\|\mathbf{F}\| > 8.0\text{ N}$ and downward $-v_z > 0.08\text{ m/s}$ quadratically; returns $\le 0$.
   - `stairs_riser_trip_penalty`: Computes shear excess $\max(0, F_{xy} - 0.5 F_z)$ without division by zero; handles 1D and 2D `found` sensor shape variations; returns $\le 0$.
   - `stairs_sole_flat_penalty`: Projects normalized per-env gravity (`dim=-1`) into foot site frame; gated on stance contact; returns $\le 0$.
   - `stairs_relative_height_progress`: Potential-based shaping $\Delta \min(z_{\text{rel}}, \text{ceiling})$ with $z_{\text{rel}} = z_{\text{trunk}} - \min(z_{\text{foot\_l}}, z_{\text{foot\_r}})$; resets `_stairs_rel_height_pot_prev` on `episode_length_buf <= 1`.

3. **Test Executions and Results**:
   - `PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`:
     `21 passed in 9.88s`
   - `PYTHONPATH="" uv run --with pytest pytest tests/`:
     `271 passed, 6 skipped in 15.89s` (0 failures, 0 regressions).

---

## 2. Logic Chain

1. **Geometry & Physics Compliance**:
   - Observations 1 show that `scene_stairs.xml` strictly implements the 2.5 cm, 3.0 cm, and 3.5 cm riser sequence, continuous gapless X-extent from 0.30 m to 2.58 m, solid grounding down to z=0, 3D friction with $\mu = 1.2$, and 8 stable spawn keyframes. Dynamic forward stepping confirms robot equilibrium stability.

2. **MDP Correctness & Stability**:
   - Observation 2 confirms that all penalty terms return values $\le 0$ as required by the microduck sign convention.
   - Observation 2 confirms no division by zero in `stairs_riser_trip_penalty` because subtraction $(F_{xy} - \mu F_z)$ is used instead of ratio division.
   - Observation 2 confirms that `stairs_relative_height_progress` is invariant to step elevation changes: stepping onto higher stair treads changes both trunk and feet heights equally, resulting in $\Delta \Phi = 0.0$, preventing reward exploitation.
   - Resets are isolated per environment using the boolean tensor `episode_length_buf <= 1`, eliminating boundary spikes.

3. **Zero Regressions & Non-interference**:
   - Existing codebase was not broken. Observation 3 verifies that 271 regression tests passed cleanly without a single failure.

4. **Integrity Validation**:
   - No hardcoded test stubs, fake implementations, shortcuts, or fabricated outputs exist. All components compute real physics and tensor dynamics.

---

## 3. Caveats

- In `stairs_relative_height_progress`, `_stairs_rel_height_pot_prev` is dynamically initialized on `env`. If an external script mutates `env.num_envs` on an existing instance without destroying the instance, the tensor shape would need to be reallocated. Standard `ManagerBasedRlEnv` instances maintain a fixed batch size throughout their lifecycle.
- 6 tests were skipped in the full pytest run: 1 is an `aarch64` CUDA test skipped on this `x86_64` runner, and 5 are progressive integration tests in `tests/test_stairs_e2e.py` testing M2/M3 artifacts (`microduck_stairs_env_cfg.py`, `infer_stairs.py`, and task registration) that are planned for future milestones.

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone M1 satisfies all requirements set forth in `ORIGINAL_REQUEST.md` and `PROJECT.md`:
- `scene_stairs.xml` provides accurate, robust 3D contact simulation on 2.5 cm, 3.0 cm, and 3.5 cm high stairs.
- `mdp.py` additions provide robust, mathematically sound, and zero-division safe reward/penalty terms adhering to all repository invariants.
- Test coverage is complete and independent verification confirmed 0 failures and 0 regressions across the entire suite.

---

## 5. Verification Method

To independently reproduce this verification:
1. Verify unit tests:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py
   ```
2. Verify full regression suite:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/
   ```
3. Verify elevation invariance:
   ```bash
   uv run python -c "
   import torch
   from mjlab_microduck.tasks import mdp as microduck_mdp
   class D:
       root_link_pos_w = torch.tensor([[0., 0., 0.125]])
       site_pos_w = torch.tensor([[[0.,0.,0.],[0.,0.,0.01],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.01]]])
   env = type('E', (), {'num_envs': 1, 'device': 'cpu', 'episode_length_buf': torch.tensor([2]), 'scene': {'robot': type('A', (), {'data': D()})()}})()
   microduck_mdp.stairs_relative_height_progress(env)
   env.scene['robot'].data.root_link_pos_w[:, 2] += 0.035
   env.scene['robot'].data.site_pos_w[:, [1, 6], 2] += 0.035
   out = microduck_mdp.stairs_relative_height_progress(env)
   assert torch.allclose(out, torch.tensor([0.0])), f'Failed: {out}'
   print('Verified!')
   "
   ```
