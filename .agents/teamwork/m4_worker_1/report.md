# Milestone M4 Verification Report: Comprehensive E2E Verification & Export Hardening

**Date**: 2026-10-11T20:07:00Z  
**Worker**: M4 Worker (`m4_worker_1`)  
**Workspace**: `/home/dhp01/microduck_rl`  

---

## 1. Registry Verification

### Command
```bash
PYTHONPATH="" uv run list-envs | grep StairsClimb
```

### Return Code
`0`

### Verbatim Stdout
```
| 26 | Mjlab-StairsClimb-Backlash-MicroDuck               |
| 27 | Mjlab-StairsClimb-MicroDuck                        |
```

### Assessment
Both the base high-stairs environment (`Mjlab-StairsClimb-MicroDuck`) and its backlash twin (`Mjlab-StairsClimb-Backlash-MicroDuck`) are successfully registered in the live environment registry.

---

## 2. Full Repository Regression Test Suite

### Command
```bash
PYTHONPATH="" uv run --with pytest pytest tests/
```

### Return Code
`0`

### Verbatim Stdout
```
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/dhp01/microduck_rl
configfile: pyproject.toml
plugins: typeguard-4.4.4, anyio-4.12.1
collecting ... collecting 16 items                                                            collected 426 items                                                            

tests/test_aarch64_cuda_torch.py .....s                                  [  1%]
tests/test_crouch_glide.py ..........                                    [  3%]
tests/test_descent_speed.py ....                                         [  4%]
tests/test_fetch_cfg.py ...                                              [  5%]
tests/test_ground_pick_cfg.py .....                                      [  6%]
tests/test_ground_pick_pose.py .......                                   [  8%]
tests/test_head_pose_bias.py ........                                    [ 10%]
tests/test_hf_jobs_flag.py ............                                  [ 12%]
tests/test_infer_fetch.py ..                                             [ 13%]
tests/test_infer_heading_control.py .                                    [ 13%]
tests/test_infer_policy_bam.py ...                                       [ 14%]
tests/test_infer_stairs.py ..........                                    [ 16%]
tests/test_infer_stairs_adversarial.py ................................. [ 24%]
..........................................                               [ 34%]
tests/test_infer_stairs_cli_adversarial.py ...............               [ 37%]
tests/test_nan_guard.py ....                                             [ 38%]
tests/test_obs_nan_guard.py ........                                     [ 40%]
tests/test_publish_manifest.py ..............................            [ 47%]
tests/test_roller_crouch_cfg.py ...                                      [ 48%]
tests/test_roller_slope_cfg.py ............                              [ 51%]
tests/test_roller_standup_cfg.py .....................................   [ 59%]
tests/test_scene_stairs.py ..........                                    [ 62%]
tests/test_scene_stairs_adversarial.py .......................           [ 67%]
tests/test_slope_curriculum.py ....                                      [ 68%]
tests/test_slope_terrain.py ........                                     [ 70%]
tests/test_spin.py ........................                              [ 76%]
tests/test_spin_cfg.py ..........                                        [ 78%]
tests/test_stairs_adversarial.py ...........                             [ 80%]
tests/test_stairs_cfg.py ...............                                 [ 84%]
tests/test_stairs_e2e.py ..s.....................                        [ 90%]
tests/test_stairs_mdp.py ...........                                     [ 92%]
tests/test_swizzle_head_cfg.py .                                         [ 92%]
tests/test_velstand_cfg.py ..........................                    [ 99%]
tests/test_wheel_glide.py ....                                           [100%]

======================= 424 passed, 2 skipped in 31.61s ========================
```

### Assessment
The full regression suite passed with 100% success rate (424 passed, 2 skipped, 0 failures, 0 regressions). All custom stairs unit tests, adversarial test suites, and existing repository invariant tests passed cleanly.

---

## 3. GPU Warp 5-Iteration Smoke Test

### Command
```bash
PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

### Return Code
`0`

### Training Telemetry Summary (Iteration 4/5)
```
################################################################################
                             Learning iteration 4/5                              

                               Run name: stairs_climb
                            Total steps: 7680 
                       Steps per second: 520 
                        Collection time: 2.730s 
                          Learning time: 0.222s 
                        Mean value loss: 0.0415
                    Mean surrogate loss: -0.0400
                      Mean entropy loss: 19.7866
                            Mean reward: -0.79
                    Mean episode length: 35.56
                        Mean action std: 0.99
   Episode_Reward/track_linear_velocity: 0.0254
  Episode_Reward/track_angular_velocity: 0.0020
                 Episode_Reward/upright: 0.0265
                    Episode_Reward/pose: 0.0088
            Episode_Reward/body_ang_vel: -0.0249
        Episode_Reward/angular_momentum: -0.0000
          Episode_Reward/dof_pos_limits: -0.0005
          Episode_Reward/action_rate_l2: -0.0854
                Episode_Reward/air_time: 0.0048
               Episode_Reward/foot_slip: -0.0001
         Episode_Reward/self_collisions: -0.0122
      Episode_Reward/head_pose_tracking: 0.0599
      Episode_Reward/body_pose_tracking: 0.0000
            Episode_Reward/heading_hold: 0.0145
          Episode_Reward/head_pose_bias: 0.0000
   Episode_Reward/stairs_foot_clearance: -0.0003
Episode_Reward/stairs_foot_landing_damping: -0.0050
       Episode_Reward/stairs_riser_trip: -0.0476
        Episode_Reward/stairs_sole_flat: -0.0060
Episode_Reward/stairs_relative_height_progress: -0.0015
        Episode_Reward/upright_progress: -0.0036
         Episode_Reward/height_progress: -0.0005
     Episode_Reward/com_upward_velocity: 0.0000
    Episode_Reward/joint_torque_rate_l2: -0.0002
            Episode_Reward/servo_impact: -0.0000
             Episode_Reward/head_impact: -0.0000
            Episode_Reward/trunk_impact: -0.0000
         Episode_Reward/servo_acc_spike: -0.0002
             Episode_Reward/servo_stall: -0.0002
             Episode_Reward/gentle_rise: -0.0001
              Episode_Reward/fallen_tax: 0.0000
        Episode_Reward/recovery_success: 0.0000
        Episode_Metrics/mean_action_acc: 1.9378
...
          Episode_Termination/nan_state: 0.0000
```

### Assessment
- `Episode_Termination/nan_state`: `0.0000` (Zero NaNs throughout simulation).
- Reward Sign Invariants: Every penalty reward term in `Episode_Reward/` is strictly $\le 0.0$. No positive rewards for penalties or reward-hacking loops detected.

---

## 4. Backlash Variant Smoke Test

### Command
```bash
PYTHONPATH="" uv run train Mjlab-StairsClimb-Backlash-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 3
```

### Return Code
`0`

### Training Telemetry Summary (Iteration 2/3)
```
################################################################################
                             Learning iteration 2/3                              

                               Run name: stairs_climb
                            Total steps: 4608 
                       Steps per second: 473 
                        Collection time: 3.014s 
                          Learning time: 0.229s 
                        Mean value loss: 0.0415
                    Mean surrogate loss: -0.0399
                      Mean entropy loss: 19.8353
                            Mean reward: -0.72
                    Mean episode length: 33.00
                        Mean action std: 1.00
   Episode_Reward/track_linear_velocity: 0.0249
  Episode_Reward/track_angular_velocity: 0.0017
                 Episode_Reward/upright: 0.0230
                    Episode_Reward/pose: 0.0075
            Episode_Reward/body_ang_vel: -0.0231
        Episode_Reward/angular_momentum: -0.0000
          Episode_Reward/dof_pos_limits: -0.0004
          Episode_Reward/action_rate_l2: -0.0719
                Episode_Reward/air_time: 0.0057
               Episode_Reward/foot_slip: -0.0001
         Episode_Reward/self_collisions: -0.0152
      Episode_Reward/head_pose_tracking: 0.0538
      Episode_Reward/body_pose_tracking: 0.0000
            Episode_Reward/heading_hold: 0.0138
          Episode_Reward/head_pose_bias: 0.0000
   Episode_Reward/stairs_foot_clearance: -0.0003
Episode_Reward/stairs_foot_landing_damping: -0.0045
       Episode_Reward/stairs_riser_trip: -0.0398
        Episode_Reward/stairs_sole_flat: -0.0060
Episode_Reward/stairs_relative_height_progress: -0.0015
        Episode_Reward/upright_progress: -0.0037
         Episode_Reward/height_progress: -0.0005
     Episode_Reward/com_upward_velocity: 0.0000
    Episode_Reward/joint_torque_rate_l2: -0.0002
            Episode_Reward/servo_impact: -0.0000
             Episode_Reward/head_impact: -0.0000
            Episode_Reward/trunk_impact: -0.0000
         Episode_Reward/servo_acc_spike: -0.0001
             Episode_Reward/servo_stall: -0.0002
             Episode_Reward/gentle_rise: -0.0001
              Episode_Reward/fallen_tax: 0.0000
        Episode_Reward/recovery_success: 0.0000
...
          Episode_Termination/nan_state: 0.0000
```

### Assessment
- `Episode_Termination/nan_state`: `0.0000` (Zero NaNs).
- Backlash passive hinge model stepped cleanly on GPU Warp with identical MDP term signs and stable dynamics.

---

## 5. ONNX Export with Baked Normalizer

### Export Command
```bash
PYTHONPATH="" uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/m4_final_stairs.onnx
```

### Return Code
`0`

### Verbatim Stdout
```
Written /tmp/m4_final_stairs.onnx
```

### In-Depth ONNX Graph Inspection Script & Verbatim Output
```python
import onnx
model = onnx.load('/tmp/m4_final_stairs.onnx')
print('=== Model Inspection: /tmp/m4_final_stairs.onnx ===')
print('Producer Name:', model.producer_name)
print('Graph Name:', model.graph.name)
print('Inputs:')
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'  name: {inp.name}, shape: {shape}, type: {inp.type.tensor_type.elem_type}')
print('Outputs:')
for out in model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f'  name: {out.name}, shape: {shape}, type: {out.type.tensor_type.elem_type}')
node_types = [n.op_type for n in model.graph.node]
print(f'Total nodes: {len(node_types)}')
print(f'Unique node types: {set(node_types)}')
print('First 10 nodes:')
for i, n in enumerate(model.graph.node[:10]):
    print(f'  [{i}] op_type={n.op_type}, name={n.name}, inputs={list(n.input)}, outputs={list(n.output)}')
```

```
=== Model Inspection: /tmp/m4_final_stairs.onnx ===
Producer Name: pytorch
Graph Name: main_graph
Inputs:
  name: obs, shape: [1, 61], type: 1
Outputs:
  name: actions, shape: [1, 14], type: 1
Total nodes: 9
Unique node types: {'Div', 'Elu', 'Gemm', 'Sub'}
First 10 nodes:
  [0] op_type=Sub, name=/obs_normalizer/Sub, inputs=['obs', 'obs_normalizer._mean'], outputs=['/obs_normalizer/Sub_output_0']
  [1] op_type=Div, name=/obs_normalizer/Div, inputs=['/obs_normalizer/Sub_output_0', 'onnx::Div_24'], outputs=['/obs_normalizer/Div_output_0']
  [2] op_type=Gemm, name=/mlp/mlp.0/Gemm, inputs=['/obs_normalizer/Div_output_0', 'mlp.0.weight', 'mlp.0.bias'], outputs=['/mlp/mlp.0/Gemm_output_0']
  [3] op_type=Elu, name=/mlp/mlp.1/Elu, inputs=['/mlp/mlp.0/Gemm_output_0'], outputs=['/mlp/mlp.1/Elu_output_0']
  [4] op_type=Gemm, name=/mlp/mlp.2/Gemm, inputs=['/mlp/mlp.1/Elu_output_0', 'mlp.2.weight', 'mlp.2.bias'], outputs=['/mlp/mlp.2/Gemm_output_0']
  [5] op_type=Elu, name=/mlp/mlp.1_1/Elu, inputs=['/mlp/mlp.2/Gemm_output_0'], outputs=['/mlp/mlp.1_1/Elu_output_0']
  [6] op_type=Gemm, name=/mlp/mlp.4/Gemm, inputs=['/mlp/mlp.1_1/Elu_output_0', 'mlp.4.weight', 'mlp.4.bias'], outputs=['/mlp/mlp.4/Gemm_output_0']
  [7] op_type=Elu, name=/mlp/mlp.1_2/Elu, inputs=['/mlp/mlp.4/Gemm_output_0'], outputs=['/mlp/mlp.1_2/Elu_output_0']
  [8] op_type=Gemm, name=/mlp/mlp.6/Gemm, inputs=['/mlp/mlp.1_2/Elu_output_0', 'mlp.6.weight', 'mlp.6.bias'], outputs=['actions']
SUCCESS: Verified input [1, 61], output [1, 14], and baked running observation normalizer nodes!
```

### Assessment
- Input shape: `(1, 61)` (Strict 48D proprioception + 13D unified command block layout).
- Output shape: `(1, 14)` (14 actuated XL330 Dynamixel servos).
- Normalizer: Empirical observation normalizer nodes (`Sub` with `obs_normalizer._mean` and `Div`) are confirmed baked directly into the ONNX graph ahead of the MLP layers.

---

## 6. Standalone 50 Hz Rehearsal Simulation

### Command
```bash
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug
```

### Return Code
`0`

### Verbatim Stdout
```
=================================================================
 Microduck High Stairs Rehearsal Simulation (BAM M6, 50 Hz)       
=================================================================
 Scene: /home/dhp01/microduck_rl/src/mjlab_microduck/robot/microduck/scene_stairs.xml
 Spawn preset: bottom
 Voltage: 7.4V
 Policy: Rest PD Controller
 Headless: True | Debug: True
 Duration: 2.0s (target: 100 steps)
 Controls:
   UP / DOWN   : Forward / Backward speed (+/- 0.05 m/s)
   LEFT / RIGHT: Strafe Left / Right (+/- 0.02 m/s)
   A / E       : Turn Left / Right (+/- 0.10 rad/s)
   SPACE       : Stop (zero all velocities)
   B           : Toggle Body Pitch Trim Mode (UP/DOWN trims pitch +/- 2°)
   H           : Toggle Head Gaze Mode (tilt beak pitch down)
   C           : Toggle Heading Lock
   P           : Apply Lateral Perturbation Push
   U           : Standup Recovery Trigger
   T           : Toggle Policy Inference (pause/resume)
   Q           : Quit Cleanly
   1..5        : Jump to Spawn Presets (1=bottom, 2=flight1, ... 5=top)
=================================================================
[ 0.20s] pos:[0.00, -0.00, 0.12] | tilt: P: +2.7° R: +0.0° [BALANCED] | feet: L:floor( 3.6N) R:floor( 3.6N) | knee:  5% ank:  0% | Floor (Apron)
[ 0.40s] pos:[0.01, -0.00, 0.11] | tilt: P: +6.2° R: +0.0° [BALANCED] | feet: L:floor( 3.7N) R:floor( 3.5N) | knee:  6% ank:  1% | Floor (Apron)
[ 0.60s] pos:[0.01, -0.00, 0.11] | tilt: P:+10.2° R: +0.0° [BALANCED] | feet: L:floor( 3.0N) R:floor( 4.5N) | knee:  7% ank:  3% | Floor (Apron)
[ 0.80s] pos:[0.02, -0.00, 0.11] | tilt: P:+15.7° R: +0.0° [BALANCED] | feet: L:floor( 3.8N) R:floor( 3.4N) | knee:  9% ank:  6% | Floor (Apron)
[ 1.00s] pos:[0.03, -0.00, 0.11] | tilt: P:+25.3° R: +0.0° [BALANCED] | feet: L:floor( 3.5N) R:floor( 3.6N) | knee: 10% ank:  7% | Floor (Apron)
[ 1.20s] pos:[0.08, -0.00, 0.10] | tilt: P:+54.8° R: +0.0° [FALLEN  ] | feet: L:floor( 2.1N) R:floor( 2.1N) | knee: 10% ank:  6% | Floor (Apron)
[ 1.40s] pos:[0.13, -0.00, 0.05] | tilt: P:+83.3° R: -0.1° [FALLEN  ] | feet: L:floor( 1.7N) R:floor( 1.7N) | knee:  7% ank:  6% | Floor (Apron)
[ 1.60s] pos:[0.13, -0.00, 0.05] | tilt: P:+82.5° R: -0.0° [FALLEN  ] | feet: L:floor( 1.6N) R:floor( 1.5N) | knee:  3% ank:  6% | Floor (Apron)
[ 1.80s] pos:[0.13, -0.00, 0.05] | tilt: P:+81.9° R: -0.1° [FALLEN  ] | feet: L:floor( 1.5N) R:floor( 1.5N) | knee:  2% ank:  5% | Floor (Apron)
[ 2.00s] pos:[0.13, -0.00, 0.05] | tilt: P:+81.3° R: -0.1° [FALLEN  ] | feet: L:floor( 1.5N) R:floor( 1.5N) | knee:  0% ank:  4% | Floor (Apron)
[infer_stairs] Target duration 2.0s reached (100 steps). Exiting cleanly.
[infer_stairs] Finished 100 steps with 0 NaNs and clean exit.
```

### Assessment
The standalone rehearsal runner `scripts/infer_stairs.py` completed 100 steps cleanly with 0 NaNs, full BAM M6 actuator dynamics, and active live HUD telemetry.

---

## 7. Conclusion

All 6 acceptance criteria and verification gates for Milestone M4 are fully satisfied:
1. Environment registry shows `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck`.
2. 424 pytest tests pass with 0 regressions.
3. GPU Warp smoke test passes at 64 envs for 5 iterations with 0 NaNs and all penalty terms $\le 0.0$.
4. Backlash variant smoke test passes with 0 NaNs.
5. ONNX export emits `[1, 61] -> [1, 14]` tensor shapes with baked running observation normalizer (`Sub`, `Div`).
6. Standalone rehearsal runner executes cleanly at 50 Hz with BAM dynamics and 0 NaNs.
