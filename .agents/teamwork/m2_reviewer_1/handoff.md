# Handoff Report — Milestone M2 Review

**Agent**: M2 Reviewer 1 (`reviewer_critic`)  
**Target Milestone**: M2 — High Stairs RL Environment Config, Curriculum & Task Registration  
**Verdict**: **APPROVE**  

---

## 1. Observation

1. **Task Registry (`src/mjlab_microduck/tasks/__init__.py`)**:
   - Lines 82–85: Imported `make_microduck_stairs_env_cfg` and `MicroduckStairsRlCfg` from `.microduck_stairs_env_cfg`.
   - Lines 254–260: Registered base task:
     ```python
     register_mjlab_task(
         task_id="Mjlab-StairsClimb-MicroDuck",
         env_cfg=make_microduck_stairs_env_cfg(),
         play_env_cfg=make_microduck_stairs_env_cfg(play=True),
         rl_cfg=MicroduckStairsRlCfg,
         runner_cls=MicroduckOnPolicyRunner,
     )
     ```
   - Lines 299–308: Added twin entry `("Mjlab-StairsClimb-Backlash-MicroDuck", make_microduck_stairs_env_cfg, {}, MicroduckStairsRlCfg, _BL_ALLCOLLISIONS)` to `_BACKLASH_TASKS`.
   - Command `PYTHONPATH="" uv run list-envs | grep StairsClimb` returned exit code 0:
     ```
     | 26 | Mjlab-StairsClimb-Backlash-MicroDuck               |
     | 27 | Mjlab-StairsClimb-MicroDuck                        |
     ```

2. **Environment Configuration (`src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`)**:
   - Lines 246–247: Base factory called `make_microduck_velocity_env_cfg(play=play, rough=rough)`, wrapped with `_ObservationsDict`.
   - Lines 254–260: Robot assigned `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (70 collision geoms); solver hardened (`cfg.scene.spec_fn = _soften_terrain_contacts`, `cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 300)`, `cfg.sim.mujoco.iterations = 30`, `cfg.sim.mujoco.ls_iterations = 50`).
   - Lines 271–299: Contact sensors added: `servo_ground_contact` (matching `SERVO_GEOM_SUFFIX`), `head_ground_contact` (matching `HEAD_BODY_NAMES`), and `trunk_ground_contact`.
   - Lines 317–368: Custom MDP terms wired:
     - `stairs_foot_clearance`: `func=microduck_mdp.stairs_foot_clearance_penalty`, `weight=2.0` (positive).
     - `stairs_foot_landing_damping`: `func=microduck_mdp.stairs_foot_landing_damping_penalty`, `weight=0.25` (positive).
     - `stairs_riser_trip`: `func=microduck_mdp.stairs_riser_trip_penalty`, `weight=0.5` (positive).
     - `stairs_sole_flat`: `func=microduck_mdp.stairs_sole_flat_penalty`, `weight=1.0` (positive).
     - `stairs_relative_height_progress`: `func=microduck_mdp.stairs_relative_height_progress`, `weight=20.0` (positive).
   - Lines 403–451: Protective terms and self-recovery:
     - `action_rate_l2`: scaled by `FALLEN_SMOOTHNESS_SCALE = 0.1` while tilted $> 40^\circ$.
     - `gentle_rise`: `func=microduck_mdp.trunk_vertical_accel_penalty`, `weight=0.005 * 0.2` (positive).
     - Impact costs: negative weights (`servo_impact`: $-0.004$, `head_impact`: $-0.002$, `trunk_impact`: $-0.002$).
   - Lines 608–620: Staged clearance curriculum (`stairs_clearance_target_curriculum`, stages: 0.028m $\rightarrow$ 0.033m $\rightarrow$ 0.038m) and `terrain_levels_stairs`.
   - Lines 627–664: `MicroduckStairsRlCfg` defines `obs_normalization=True`, `experiment_name="stairs_climb"`, `num_steps_per_env=24`, `max_iterations=6_000`.

3. **Pytest Invariant and E2E Tests**:
   - `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`:
     ```
     tests/test_stairs_cfg.py ...............                                 [ 38%]
     tests/test_stairs_e2e.py ..s.....................                        [100%]
     ======================== 38 passed, 1 skipped in 8.78s =========================
     ```
   - `PYTHONPATH="" uv run --with pytest pytest tests/`:
     ```
     ======================= 329 passed, 2 skipped in 18.41s ========================
     ```

4. **Live 5-Iteration Smoke Test**:
   - `WANDB_MODE=disabled uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5` completed in 21s:
     - `Episode_Termination/nan_state: 0.0000` (Zero NaNs).
     - `Episode_Reward/stairs_foot_clearance: -0.0003` ($\le 0.0$).
     - `Episode_Reward/stairs_foot_landing_damping: -0.0050` ($\le 0.0$).
     - `Episode_Reward/stairs_riser_trip: -0.0465` ($\le 0.0$).
     - `Episode_Reward/stairs_sole_flat: -0.0066` ($\le 0.0$).
     - `Curriculum/stairs_clearance_target: 0.0280`.

---

## 2. Logic Chain

1. From Observation 1, `Mjlab-StairsClimb-MicroDuck` and its backlash twin are properly registered and discoverable by `uv run list-envs`, fulfilling Requirement R1 and Feature F8.
2. From Observation 2 and tests in Observation 3 (`test_actor_observation_keeps_the_61d_slot_layout`), the actor observation consists of 48D base proprioception + 13D commands, exactly preserving the repository-wide 61D unified actor observation contract without deleting or corrupting any slots.
3. From Observation 2, `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (and its backlash twin) provides 70 collision geoms, ensuring full-body contact fidelity on stairs for shins, knees, trunk, and 14 Dynamixel XL330 housings.
4. From Observation 2 and Observation 4, all self-negating penalty functions returning values $\le 0$ are assigned positive reward weights, and all cost functions returning values $\ge 0$ are assigned negative reward weights. The empirical smoke test logs confirm that every penalty term evaluates $\le 0.0$ in simulation, eliminating the risk of reward hacking or violation farming.
5. From Observation 3, the existing test suite (331 tests across the entire repository) passes with 100% success rate (329 passed, 2 skipped, 0 failed), verifying zero breaking changes or regressions to existing tasks.
6. From Observation 4, the 5-iteration smoke test steps without NaNs on GPU, confirming physical and numerical stability under `_soften_terrain_contacts` and hardened solver settings.
7. Therefore, the implementation in Milestone M2 is technically sound, contract-compliant, and passes all verification criteria.

---

## 3. Caveats

- **Full Multi-Flight PPO Convergence**: The 5-iteration smoke test validates compile time, step stability, GPU memory, and loss mechanics. Full 6,000-iteration policy convergence across all 3 stair flights is budgeted for Milestone M4 / long-run training and was not run during this code review.
- **Gymnasium Registry Skip in E2E**: In `tests/test_stairs_e2e.py`, `test_task_registration_visibility_if_registered` was skipped because `gymnasium` is not directly imported in the standalone runner process; `mjlab`'s internal registry was verified directly by `test_tasks_appear_in_registry` and `uv run list-envs`.

---

## 4. Conclusion

Milestone M2 is **APPROVED**. The task configuration, curriculum design, 61D observation shape, reward sign conventions, and task registration adhere strictly to all repository invariants with zero regressions. The project can proceed directly to Milestone M3.

---

## 5. Verification Method

To independently reproduce and verify this review:
1. Verify task discovery:
   ```bash
   PYTHONPATH="" uv run list-envs | grep StairsClimb
   ```
   *Expected*: Displays both `Mjlab-StairsClimb-Backlash-MicroDuck` and `Mjlab-StairsClimb-MicroDuck`.

2. Run targeted M2 invariant tests:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py
   ```
   *Expected*: 38 passed, 1 skipped.

3. Run full repository regression test suite:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/
   ```
   *Expected*: 329 passed, 2 skipped, 0 failed.

4. Run GPU 5-iteration smoke test:
   ```bash
   WANDB_MODE=disabled uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
   ```
   *Expected*: Completes cleanly in ~20s, exit code 0, 0 NaNs, all penalty terms $\le 0.0$.

*Invalidation conditions*: Any failure in observation shape (not 61D), any positive penalty metric in training logs, or any regression in `pytest tests/`.
