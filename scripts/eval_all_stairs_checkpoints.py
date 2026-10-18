#!/usr/bin/env python3
"""Comprehensive checkpoint evaluation and coordinate verification on real simulation.

Evaluates all checkpoints from model_2000.pt to model_6000.pt in:
1. Warp environment (ManagerBasedRlEnv on GPU) - measures spawn coordinates vs setup,
   step-by-step trajectory, displacement (dx, dy, dz), tilt stability, and survival.
2. MuJoCo stairs testbench (scene_stairs.xml with BAM M6 actuators) - measures
   stepping from strict setup coordinates (0, 0, 0.125) towards the stairs at x=0.30m.
"""

from __future__ import annotations

import glob
import math
import os
import re
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import asdict
import numpy as np
import torch
import mujoco

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

# Scripts / MuJoCo helpers
from scripts.infer_stairs import (
    DEFAULT_SCENE_XML,
    BAM_KP_FW,
    BAM_VIN,
    BAM_MAX_CURRENT,
    PHYSICS_DT,
    DECIMATION,
    CONTROL_DT,
    DEFAULT_JOINT_POS,
    load_bam_model,
    load_mujoco_with_bam,
    reset_to_preset,
    build_61d_observation,
    compute_euler_angles,
)


def get_all_checkpoints(log_dir: str) -> list[str]:
    files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    def extract_iter(p: str) -> int:
        m = re.search(r"model_(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    return sorted(files, key=extract_iter)


def run_eval():
    log_dir = "logs/rsl_rl/stairs_climb/2026-10-17_22-40-04_stairs_climb"
    checkpoints = get_all_checkpoints(log_dir)
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {os.path.basename(cp)}")

    # ─────────────────────────────────────────────────────────────────────────────
    # PART 1: Real Warp Simulation (ManagerBasedRlEnv)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PHẦN 1: MÔ PHỎNG THỰC TẾ TRÊN WARP GPU (ManagerBasedRlEnv - Mjlab-StairsClimb)")
    print("="*80)

    task_id = "Mjlab-StairsClimb-MicroDuck"
    device = "cuda:0"
    num_envs = 16
    eval_steps = 100  # 100 steps * 0.02s = 2.0s simulation time

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(task_id)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)

    # Invariants from env_cfg for coordinate comparison
    expected_z_range = (0.12, 0.13)
    expected_xy_range = (-0.5, 0.5)

    warp_results = []

    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        iter_num = int(re.search(r"\d+", ckpt_name).group())

        # Load weights into runner
        runner.load(ckpt_path, load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)

        # Reset environment cleanly
        raw_env.reset()
        obs = env.get_observations()
        
        # Check initial coordinates
        pos_w = raw_env.scene["robot"].data.root_link_pos_w.clone()
        origins = raw_env.scene.env_origins.clone()
        pos_rel_0 = pos_w - origins  # shape: (num_envs, 3)

        x0 = pos_rel_0[:, 0].cpu().numpy()
        y0 = pos_rel_0[:, 1].cpu().numpy()
        z0 = pos_rel_0[:, 2].cpu().numpy()

        # Step policy for eval_steps
        rewards_list = []
        for step in range(eval_steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, rew, done, info = env.step(actions)
            rewards_list.append(rew.cpu().numpy())

        # Final coordinates
        pos_w_final = raw_env.scene["robot"].data.root_link_pos_w.clone()
        pos_rel_final = pos_w_final - origins
        xf = pos_rel_final[:, 0].cpu().numpy()
        yf = pos_rel_final[:, 1].cpu().numpy()
        zf = pos_rel_final[:, 2].cpu().numpy()

        dx = xf - x0
        dy = yf - y0
        dz = zf - z0

        # Projected gravity gives tilt: gravity_b[:, 2] is approx -1 when upright
        proj_grav = raw_env.scene["robot"].data.projected_gravity_b.cpu().numpy()
        upright_cos = -proj_grav[:, 2]
        fallen = upright_cos < 0.5  # tilt > 60 deg

        mean_dx = float(np.mean(dx))
        std_dx = float(np.std(dx))
        mean_dz = float(np.mean(dz))
        mean_zf = float(np.mean(zf))
        fallen_pct = float(np.mean(fallen) * 100.0)
        mean_rew = float(np.mean(rewards_list))

        warp_results.append({
            "iter": iter_num,
            "name": ckpt_name,
            "x0_range": (float(np.min(x0)), float(np.max(x0))),
            "y0_range": (float(np.min(y0)), float(np.max(y0))),
            "z0_range": (float(np.min(z0)), float(np.max(z0))),
            "mean_dx": mean_dx,
            "std_dx": std_dx,
            "mean_dz": mean_dz,
            "mean_zf": mean_zf,
            "fallen_pct": fallen_pct,
            "mean_rew": mean_rew,
        })
        print(f"[{ckpt_name}] z0:[{np.min(z0):.3f}, {np.max(z0):.3f}] | dx: {mean_dx:+.3f}m | dz: {mean_dz:+.3f}m | zf: {mean_zf:.3f}m | Ngã: {fallen_pct:.1f}% | MeanRew: {mean_rew:.2f}")

    raw_env.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # PART 2: Real MuJoCo Stairs Testbench Simulation (scene_stairs.xml, BAM M6)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PHẦN 2: MÔ PHỎNG THỰC TẾ TRÊN MUJOCO TESTBENCH (scene_stairs.xml + BAM M6)")
    print("="*80)
    print("Setup chuẩn tọa độ gốc ban đầu: x = 0.000m, y = 0.000m, z = 0.125m (Keyframe STAND)")
    print("Bậc thang số 1 (Flight 1, 2.5cm) bắt đầu tại x = 0.300m, độ cao +0.025m")

    bam_model = load_bam_model(BAM_KP_FW, BAM_VIN, BAM_MAX_CURRENT)
    model, data, bam_ctrl, act_names = load_mujoco_with_bam(
        DEFAULT_SCENE_XML, bam_model=bam_model, timestep=PHYSICS_DT
    )

    mujoco_results = []
    cmd_twist = np.array([0.20, 0.0, 0.0], dtype=np.float32)  # commanded forward 0.20 m/s
    cmd_head = np.zeros(4, dtype=np.float32)
    cmd_body = np.zeros(6, dtype=np.float32)

    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        iter_num = int(re.search(r"\d+", ckpt_name).group())

        # Load actor PyTorch weights and normalizer
        ckpt = torch.load(ckpt_path, map_location="cpu")
        actor_dict = ckpt["actor_state_dict"]

        mean = actor_dict["obs_normalizer._mean"].numpy()
        var = actor_dict["obs_normalizer._var"].numpy()
        std = np.sqrt(var + 1e-8)

        # Build PyTorch sequential or MLP
        w0 = actor_dict["mlp.0.weight"].numpy()
        b0 = actor_dict["mlp.0.bias"].numpy()
        w2 = actor_dict["mlp.2.weight"].numpy()
        b2 = actor_dict["mlp.2.bias"].numpy()
        w4 = actor_dict["mlp.4.weight"].numpy()
        b4 = actor_dict["mlp.4.bias"].numpy()
        w6 = actor_dict["mlp.6.weight"].numpy()
        b6 = actor_dict["mlp.6.bias"].numpy()

        def forward_policy(raw_obs: np.ndarray) -> np.ndarray:
            # normalize
            norm_obs = (raw_obs - mean) / std
            # MLP with ELU activations
            h1 = np.maximum(0, norm_obs @ w0.T + b0) + np.minimum(0, np.exp(norm_obs @ w0.T + b0) - 1)
            h2 = np.maximum(0, h1 @ w2.T + b2) + np.minimum(0, np.exp(h1 @ w2.T + b2) - 1)
            h3 = np.maximum(0, h2 @ w4.T + b4) + np.minimum(0, np.exp(h2 @ w4.T + b4) - 1)
            out = h3 @ w6.T + b6
            return out

        # Reset to bottom / STAND preset
        reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

        # Settle
        for _ in range(15):
            bam_ctrl.q_target[:] = DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        init_x = float(data.qpos[0])
        init_y = float(data.qpos[1])
        init_z = float(data.qpos[2])

        last_action = np.zeros(14, dtype=np.float32)
        max_x = init_x
        min_z = init_z
        max_z = init_z
        fell = False
        step_fell = -1

        for step in range(eval_steps):
            # Read observation
            obs = build_61d_observation(model, data, last_action, cmd_twist, cmd_head, cmd_body)
            action = forward_policy(obs).flatten()
            last_action = action.copy()

            # Apply BAM decimation steps
            q_target = DEFAULT_JOINT_POS + action
            bam_ctrl.q_target[:] = q_target

            for _ in range(DECIMATION):
                bam_ctrl.update()
                mujoco.mj_step(model, data)

            cur_x = float(data.qpos[0])
            cur_y = float(data.qpos[1])
            cur_z = float(data.qpos[2])
            roll, pitch, yaw = compute_euler_angles(data.qpos[3:7])
            pitch_deg = abs(math.degrees(pitch))
            roll_deg = abs(math.degrees(roll))

            max_x = max(max_x, cur_x)
            min_z = min(min_z, cur_z)
            max_z = max(max_z, cur_z)

            if (pitch_deg > 50.0 or roll_deg > 50.0 or cur_z < 0.08) and not fell:
                fell = True
                step_fell = step

        final_x = float(data.qpos[0])
        final_y = float(data.qpos[1])
        final_z = float(data.qpos[2])
        f_roll, f_pitch, f_yaw = compute_euler_angles(data.qpos[3:7])

        reached_stairs = max_x >= 0.300
        climbed_step1 = max_z >= 0.140  # riser 2.5cm -> z >= 0.145

        mujoco_results.append({
            "iter": iter_num,
            "name": ckpt_name,
            "init_pos": (init_x, init_y, init_z),
            "final_pos": (final_x, final_y, final_z),
            "max_x": max_x,
            "max_z": max_z,
            "fell": fell,
            "step_fell": step_fell,
            "pitch_deg": math.degrees(f_pitch),
            "roll_deg": math.degrees(f_roll),
            "reached_stairs": reached_stairs,
            "climbed_step1": climbed_step1,
        })

        status_str = "NGÃ" if fell else "ĐỨNG/DI CHUYỂN"
        if step_fell >= 0:
            status_str += f" (tại bước {step_fell}, t={step_fell*0.02:.2f}s)"
        print(f"[{ckpt_name}] Init: ({init_x:.3f}, {init_y:.3f}, {init_z:.3f}) | Final: ({final_x:.3f}, {final_y:.3f}, {final_z:.3f}) | MaxX: {max_x:.3f}m | Trạng thái: {status_str}")

    return warp_results, mujoco_results


if __name__ == "__main__":
    run_eval()
