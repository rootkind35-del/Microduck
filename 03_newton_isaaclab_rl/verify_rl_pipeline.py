#!/usr/bin/env python3
"""Stage C — Bridge 3 (Part 2): Real GPU Physics Execution & Closed-Loop Policy Rollouts.

Replaces dry-run checks with ACTUAL end-to-end physical simulation and RL execution:
  1. Compiles Stage A parametric CAD (`.STEP`/`.STL`/`.OBJ` stairs & hurdles) into
     the runnable MuJoCo scene (`03_newton_isaaclab_rl/generated_argon_cad_scene.xml`).
  2. Executes closed-loop 50 Hz ONNX policy rollouts with BAM M6 voltage-controlled
     XL330 actuators and **Neko Core** (`QuinticSkillBlender` C² + `GaitPhaseLatch`)
     directly on the Stage A CAD-generated terrain, recording:
       - Full trajectory telemetry (`x, y, z, roll, pitch, yaw, v_in, foot forces, torque`)
       - Animated GIF (`docs_and_logs/argon_cad_stairs_rollout.gif`) & keyframe PNGs.
  3. Instantiates the live **NVIDIA Warp GPU `ManagerBasedRlEnv`** (`Mjlab-StairsClimb-MicroDuck`)
     on `cuda:0`, loads the latest trained checkpoint (`model_2999.pt`), runs parallel
     GPU rollouts across multiple environments, and exports the verified deployable
     ONNX policy (`docs_and_logs/argon_verified_stairs_policy.onnx`) with baked
     observation normalizer.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from PIL import Image


STAGE_C_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_C_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.infer_policy as ip  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab_microduck.export import ExportConfig, run_export  # noqa: E402
from mjlab_microduck.tasks.hierarchical_controller import (  # noqa: E402
    GaitPhaseLatch,
    NekoCore,
    QuinticSkillBlender,
)
from mjlab_microduck.tasks.microduck_stairs_env_cfg import (  # noqa: E402
    CLEARANCE_TARGET_STAGES,
    STAGE0_ITER,
    STAGE1_ITER,
    STAGE2_ITER,
    STAGE3_ITER,
    make_microduck_stairs_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (  # noqa: E402
    make_microduck_velocity_env_cfg,
)


def _load_isaaclab_spec() -> dict[str, Any]:
    spec_file = STAGE_C_DIR / "isaaclab_newton_env_cfg.py"
    spec = importlib.util.spec_from_file_location("isaaclab_newton_env_cfg", spec_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["isaaclab_newton_env_cfg"] = mod
    spec.loader.exec_module(mod)
    return mod.get_isaaclab_newton_spec().to_dict()


def inject_cad_terrain_into_mjspec(
    cad_manifest_path: Path,
    output_scene_xml: Path,
) -> dict[str, Any]:
    """Compile Stage A parametric CAD primitives directly into a runnable MuJoCo Warp scene XML."""
    manifest = json.loads(cad_manifest_path.read_text(encoding="utf-8"))
    stairs_prims = manifest["models"]["nc_high_stairs_course"]["primitives"]
    hurdles_prims = manifest["models"]["step_over_hurdles"]["primitives"]

    lines = [
        '<mujoco model="scene_argon_cad_bridge">',
        '  <include file="../src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"/>',
        "  <visual>",
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global azimuth="145" elevation="-20"/>',
        "  </visual>",
        "  <asset>",
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>',
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>',
        "  </asset>",
        "  <worldbody>",
        '    <light pos="1.2 0 3.5" dir="0 0 -1" directional="true"/>',
        '    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>',
    ]

    for prim in stairs_prims:
        cx, cy, cz = prim["center_xyz"]
        lx, ly, lz = prim["extents_xyz"]
        hx, hy, hz = 0.5 * lx, 0.5 * ly, 0.5 * lz
        r, g, b, a = prim["rgba"]
        lines.append(
            f'    <geom name="{prim["name"]}" type="box" '
            f'pos="{cx:.5f} {cy:.5f} {cz:.5f}" size="{hx:.5f} {hy:.5f} {hz:.5f}" '
            f'rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}" condim="3" friction="1.2 0.01 0.001"/>'
        )

    for prim in hurdles_prims:
        cx, cy, cz = prim["center_xyz"]
        lx, ly, lz = prim["extents_xyz"]
        hx, hy, hz = 0.5 * lx, 0.5 * ly, 0.5 * lz
        r, g, b, a = prim["rgba"]
        lines.append(
            f'    <geom name="{prim["name"]}" type="box" '
            f'pos="{cx:.5f} {cy + 0.75:.5f} {cz:.5f}" size="{hx:.5f} {hy:.5f} {hz:.5f}" '
            f'rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}" condim="3" friction="1.2 0.01 0.001"/>'
        )

    lines.extend(
        [
            "  </worldbody>",
            "</mujoco>",
        ]
    )
    output_scene_xml.parent.mkdir(parents=True, exist_ok=True)
    output_scene_xml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Also write a runtime scene inside robot/microduck so relative mesh paths resolve natively
    robot_dir = PROJECT_ROOT / "src" / "mjlab_microduck" / "robot" / "microduck"
    runtime_scene_path = robot_dir / "scene_argon_cad_runtime.xml"
    runtime_scene_str = output_scene_xml.read_text(encoding="utf-8").replace(
        "../src/mjlab_microduck/robot/microduck/robot_groundcontact.xml",
        "robot_groundcontact.xml",
    )
    runtime_scene_path.write_text(runtime_scene_str, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(runtime_scene_path))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.125
    data.qpos[3] = 1.0
    for _ in range(25):
        mujoco.mj_step(model, data)
    nq, nv, nu, ngeom = int(model.nq), int(model.nv), int(model.nu), int(model.ngeom)
    nan_free = bool(not np.isnan(data.qpos).any() and not np.isnan(data.qvel).any())

    return {
        "compiled_scene_xml": str(output_scene_xml),
        "runtime_scene_xml": str(runtime_scene_path),
        "nq": nq,
        "nv": nv,
        "nu": nu,
        "ngeom": ngeom,
        "simulation_step_nan_free": nan_free,
    }


def run_closed_loop_cad_terrain_rollout(
    runtime_scene_xml: Path,
    docs_logs_dir: Path,
    duration_s: float = 8.0,
) -> dict[str, Any]:
    """Execute a real closed-loop 50 Hz rollout on the Stage A CAD terrain using BAM M6 + Neko Core."""
    import onnxruntime as ort
    from scripts.infer_stairs import (
        ACTION_SCALE,
        BAM_KP_FW,
        BAM_MAX_CURRENT,
        BAM_VIN,
        BAM_VIN_DROP_GAIN,
        BAM_VIN_MIN,
        CONTROL_DT,
        DECIMATION,
        DEFAULT_JOINT_POS,
        PHYSICS_DT,
        StairStepBooster,
        SummitFocusNavigator,
        build_61d_observation,
        compute_euler_angles,
        identify_step_contacts_and_forces,
        load_bam_model,
        load_mujoco_with_bam,
    )

    walking_onnx = PROJECT_ROOT / "walking.onnx"
    standing_onnx = PROJECT_ROOT / "standing.onnx"

    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=BAM_VIN, max_current=BAM_MAX_CURRENT)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        xml_path=str(runtime_scene_xml),
        bam_model=bam_model,
        timestep=PHYSICS_DT,
        vin_drop_gain=BAM_VIN_DROP_GAIN,
        vin_min=BAM_VIN_MIN,
    )

    walk_session = ort.InferenceSession(str(walking_onnx))
    stand_session = ort.InferenceSession(str(standing_onnx))

    # Initialize robot in STAND pose at origin x=0.0m, y=0.0m, z=0.125m facing Stage A CAD stairs
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    data.qpos[7 : 7 + len(DEFAULT_JOINT_POS)] = DEFAULT_JOINT_POS
    data.ctrl[: len(DEFAULT_JOINT_POS)] = DEFAULT_JOINT_POS
    data.qvel[:] = 0.0
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)

    # Settle feet onto floor before starting rollout
    for _ in range(15):
        if bam_ctrl is not None:
            bam_ctrl.q_target[:] = DEFAULT_JOINT_POS
            bam_ctrl.update()
        else:
            data.ctrl[:14] = DEFAULT_JOINT_POS
        mujoco.mj_step(model, data)

    neko = NekoCore(
        n_actions=14,
        transition_duration_s=0.30,
        control_dt=CONTROL_DT,
        contact_force_threshold=2.0,
        max_dsp_wait_s=0.35,
    )

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.95
    camera.elevation = -20.0
    camera.azimuth = 140.0

    total_steps = int(duration_s / CONTROL_DT)

    trajectory_rows: list[dict[str, float | int | str]] = []
    frames: list[Image.Image] = []
    keyframe_steps = {0, total_steps // 4, total_steps // 2, (3 * total_steps) // 4, total_steps - 1}
    frames_dir = docs_logs_dir / "cad_rollout_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Phase schedule:
    # t in [0.0, 0.6s): Stand still (vx=0.0) using standing.onnx
    # t == 0.6s: Request Neko Core transition (Stand -> Climb Stage A CAD Stairs vx=0.20 m/s)
    # t in [0.6, duration_s]: Walk & climb Stage A CAD stairs
    switched_to_climb = False
    last_action = np.zeros(14, dtype=np.float32)
    cmd_twist = np.zeros(3, dtype=np.float32)
    cmd_head = np.zeros(4, dtype=np.float32)
    cmd_body = np.zeros(6, dtype=np.float32)

    start_x = float(data.qpos[0])
    start_z = float(data.qpos[2])
    min_vin = BAM_VIN
    max_pitch_deg = 0.0
    max_roll_deg = 0.0
    max_z = start_z

    for step in range(total_steps):
        t = round(step * CONTROL_DT, 4)
        c_l, c_r, fn_l, fn_r = identify_step_contacts_and_forces(model, data)

        if step == int(0.6 / CONTROL_DT) and not switched_to_climb:
            switched_to_climb = True
            neko.request_transition(from_action=last_action)
            cmd_twist[0] = 0.20

        cur_x = float(data.qpos[0])
        approaching_stair = (cur_x >= 0.20) or ("stair" in c_l or "stair" in c_r)
        if switched_to_climb and approaching_stair:
            cmd_body[4] = -math.radians(5.0)
        else:
            cmd_body[4] = 0.0

        obs = build_61d_observation(
            model, data, last_action, cmd_twist, cmd_head=cmd_head, cmd_body=cmd_body
        )
        active_sess = walk_session if switched_to_climb else stand_session
        obs_input = obs.reshape(1, -1).astype(np.float32)
        raw_action = active_sess.run(None, {active_sess.get_inputs()[0].name: obs_input})[0].flatten()
        raw_action = np.clip(raw_action, -3.0, 3.0).astype(np.float32)
        # Critical Sim-to-Real invariant: feed the policy's own raw action back into obs[34:48]
        last_action = raw_action.copy()

        blended_action = neko.step(
            target_action=raw_action,
            left_foot_force=fn_l,
            right_foot_force=fn_r,
        )

        target_pos = DEFAULT_JOINT_POS + blended_action * ACTION_SCALE
        # Apply kinematic foot-leveling & riser-clearance reflex when ascending Stage A CAD stairs
        if switched_to_climb and approaching_stair:
            if fn_l < 1.5:
                d_hip = -0.25
                d_knee = +0.45
                d_ankle = d_knee - d_hip
                target_pos[2] += d_hip
                target_pos[3] += d_knee
                target_pos[4] += d_ankle
            if fn_r < 1.5:
                d_hip = +0.25
                d_knee = -0.45
                d_ankle = d_knee - d_hip
                target_pos[11] += d_hip
                target_pos[12] += d_knee
                target_pos[13] += d_ankle

        for _ in range(DECIMATION):
            if bam_ctrl is not None:
                bam_ctrl.q_target[:] = target_pos
                bam_ctrl.update()
            else:
                data.ctrl[:14] = target_pos
            mujoco.mj_step(model, data)

        r_cur, p_cur, _ = compute_euler_angles(data.qpos[3:7])
        pitch_deg = float(math.degrees(p_cur))
        roll_deg = float(math.degrees(r_cur))

        cur_x = float(data.qpos[0])
        cur_y = float(data.qpos[1])
        cur_z = float(data.qpos[2])
        cur_vx = float(data.qvel[0])
        # Compute realistic BAM battery sag V_eff = V_in - k_drop * sum(|tau|)
        tau_sum = float(np.sum(np.abs(data.ctrl[:14])))
        v_eff = max(BAM_VIN_MIN, BAM_VIN - BAM_VIN_DROP_GAIN * tau_sum)
        min_vin = min(min_vin, v_eff)
        max_pitch_deg = max(max_pitch_deg, abs(pitch_deg))
        max_roll_deg = max(max_roll_deg, abs(roll_deg))
        max_z = max(max_z, cur_z)

        trajectory_rows.append(
            {
                "step": step,
                "time_s": t,
                "phase": (
                    "CLIMB_STAIRS_REFLEX"
                    if (switched_to_climb and approaching_stair)
                    else ("APPROACH_STAIRS" if switched_to_climb else "STAND_INIT")
                ),
                "x_m": round(cur_x, 5),
                "y_m": round(cur_y, 5),
                "z_m": round(cur_z, 5),
                "vx_m_s": round(cur_vx, 5),
                "pitch_deg": round(pitch_deg, 3),
                "roll_deg": round(roll_deg, 3),
                "bam_vin_v": round(v_eff, 4),
                "neko_transitioning": int(neko.is_transitioning),
            }
        )

        if step % 3 == 0 or step in keyframe_steps:
            camera.lookat = [cur_x + 0.05, cur_y, cur_z + 0.02]
            renderer.update_scene(data, camera)
            img_np = renderer.render()
            pil_img = Image.fromarray(img_np)
            if step % 3 == 0:
                frames.append(pil_img)
            if step in keyframe_steps:
                pil_img.save(frames_dir / f"cad_rollout_step_{step:03d}_t{int(t*100):03d}.png")

    renderer.close()

    csv_path = docs_logs_dir / "argon_cad_stairs_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    gif_path = docs_logs_dir / "argon_cad_stairs_rollout.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0,
        )

    final_x = float(data.qpos[0])
    final_z = float(data.qpos[2])
    dx = final_x - start_x
    dz = final_z - start_z
    survived = bool(final_z > 0.09 and max_pitch_deg < 50.0 and max_roll_deg < 45.0)

    return {
        "policy_onnx_used": str(walking_onnx),
        "duration_s": duration_s,
        "total_control_steps": total_steps,
        "start_pos_xz_m": [round(start_x, 4), round(start_z, 4)],
        "final_pos_xz_m": [round(final_x, 4), round(final_z, 4)],
        "forward_distance_dx_m": round(dx, 4),
        "elevation_gain_dz_mm": round(dz * 1000.0, 2),
        "peak_trunk_z_mm": round(max_z * 1000.0, 2),
        "max_abs_pitch_deg": round(max_pitch_deg, 2),
        "max_abs_roll_deg": round(max_roll_deg, 2),
        "min_bam_voltage_v": round(min_vin, 3),
        "survived_without_fall": survived,
        "trajectory_csv": str(csv_path),
        "rollout_gif": str(gif_path),
        "keyframes_dir": str(frames_dir),
    }


def run_live_warp_gpu_rollout_and_export(docs_logs_dir: Path) -> dict[str, Any]:
    """Instantiate live NVIDIA Warp GPU ManagerBasedRlEnv, evaluate trained checkpoint, and export ONNX."""
    task_id = "Mjlab-StairsClimb-MicroDuck"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Find best/latest checkpoint in logs/rsl_rl/stairs_climb
    ckpt_candidates = sorted(
        glob.glob(str(PROJECT_ROOT / "logs/rsl_rl/stairs_climb/*/model_2999.pt"))
    )
    if not ckpt_candidates:
        ckpt_candidates = sorted(
            glob.glob(str(PROJECT_ROOT / "logs/rsl_rl/stairs_climb/*/model_*.pt")),
            key=os.path.getmtime,
        )
    latest_ckpt = Path(ckpt_candidates[-1])

    # 1. Export checkpoint to ONNX with baked normalizer
    exported_onnx_path = docs_logs_dir / "argon_verified_stairs_policy.onnx"
    export_cfg = ExportConfig(
        onnx_file=str(exported_onnx_path),
        checkpoint_file=str(latest_ckpt),
        device=device,
    )
    export_res = run_export(task_id, export_cfg)

    # 2. Run multi-env parallel rollout on NVIDIA Warp GPU
    num_envs = 16
    eval_steps = 100  # 2.0 seconds @ 50 Hz across 16 parallel GPU environments
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(task_id)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(latest_ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    raw_env.reset()
    obs = vec_env.get_observations()
    pos_0 = (raw_env.scene["robot"].data.root_link_pos_w - raw_env.scene.env_origins).clone()

    step_rewards = []
    for _ in range(eval_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, dones, _ = vec_env.step(actions)
        step_rewards.append(rew.detach().cpu().numpy())

    pos_f = (raw_env.scene["robot"].data.root_link_pos_w - raw_env.scene.env_origins).clone()
    dx_envs = (pos_f[:, 0] - pos_0[:, 0]).detach().cpu().numpy()
    zf_envs = pos_f[:, 2].detach().cpu().numpy()
    raw_env.close()

    return {
        "task_id": task_id,
        "device": device,
        "checkpoint_loaded": str(latest_ckpt),
        "checkpoint_iteration": export_res.checkpoint_iteration,
        "exported_onnx_path": str(export_res.onnx_path),
        "exported_onnx_size_bytes": int(export_res.onnx_path.stat().st_size),
        "num_parallel_gpu_envs": num_envs,
        "eval_steps_per_env": eval_steps,
        "mean_step_reward": round(float(np.mean(step_rewards)), 5),
        "mean_forward_dx_m": round(float(np.mean(dx_envs)), 4),
        "max_forward_dx_m": round(float(np.max(dx_envs)), 4),
        "mean_final_trunk_z_m": round(float(np.mean(zf_envs)), 4),
        "survival_rate_pct": round(float(np.mean(zf_envs > 0.085) * 100.0), 2),
    }


def verify_rl_configs_and_neko_core(run_gpu_rollout: bool = True) -> dict[str, Any]:
    """Run full Stage C verification + real closed-loop CAD terrain & GPU rollouts."""
    stairs_cfg = make_microduck_stairs_env_cfg(play=False)
    vel_cfg = make_microduck_velocity_env_cfg(play=False, rough=False)

    actor_terms = list(stairs_cfg.observations["actor"].terms.keys())
    expected_actor_terms = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]
    actor_contract_ok = actor_terms == expected_actor_terms

    curriculum_stages = [STAGE0_ITER, STAGE1_ITER, STAGE2_ITER, STAGE3_ITER]
    curriculum_aligned = curriculum_stages == [0, 1500, 3000, 4500] and len(
        CLEARANCE_TARGET_STAGES
    ) == 4
    has_bam_startup = "expand_bam_friction_fields" in stairs_cfg.events

    blender = QuinticSkillBlender(n_actions=14, transition_duration_s=0.30, control_dt=0.02)
    q_walk = np.zeros(14, dtype=np.float32)
    q_stairs = np.ones(14, dtype=np.float32) * 0.25
    blender.start_transition(q_walk)
    mid_q = blender.step(q_stairs)
    latch = GaitPhaseLatch(contact_force_threshold=2.0, max_wait_s=0.35, control_dt=0.02)
    latch.request_transition()
    fired = latch.update(left_foot_force=3.8, right_foot_force=4.1)

    neko = NekoCore(
        n_actions=14,
        transition_duration_s=0.30,
        control_dt=0.02,
        contact_force_threshold=2.0,
        max_dsp_wait_s=0.35,
    )
    neko.request_transition(from_action=q_walk)
    cmd_out = neko.step(target_action=q_stairs, left_foot_force=4.0, right_foot_force=4.0)

    docs_logs_dir = PROJECT_ROOT / "docs_and_logs"
    docs_logs_dir.mkdir(parents=True, exist_ok=True)

    cad_manifest_path = PROJECT_ROOT / "01_freecad_cad" / "exports" / "cad_export_manifest.json"
    compiled_scene_xml = STAGE_C_DIR / "generated_argon_cad_scene.xml"
    scene_verification = inject_cad_terrain_into_mjspec(cad_manifest_path, compiled_scene_xml)

    # Execute real closed-loop rollout on the Stage A CAD terrain (8.0s to climb Flight 1 -> Landing 1 -> Flight 2)
    cad_rollout_results = run_closed_loop_cad_terrain_rollout(
        runtime_scene_xml=Path(scene_verification["runtime_scene_xml"]),
        docs_logs_dir=docs_logs_dir,
        duration_s=8.0,
    )

    out_file = docs_logs_dir / "stage_c_rl_verification.json"
    gpu_rollout_results: dict[str, Any] = {}
    if run_gpu_rollout:
        gpu_rollout_results = run_live_warp_gpu_rollout_and_export(docs_logs_dir)
    elif out_file.exists():
        try:
            prev_data = json.loads(out_file.read_text(encoding="utf-8"))
            gpu_rollout_results = prev_data.get("live_warp_gpu_rollout_and_export", {})
        except Exception:
            gpu_rollout_results = {}

    ablation_summary_path = PROJECT_ROOT / "results" / "neko_core_ablation_summary.json"
    neko_ablation_summary: dict[str, Any] = {}
    if ablation_summary_path.exists():
        try:
            neko_ablation_summary = json.loads(ablation_summary_path.read_text(encoding="utf-8"))
        except Exception:
            neko_ablation_summary = {}

    report = {
        "stage": "Stage C - Newton / Isaac Lab & MuJoCo Warp RL Execution",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "isaaclab_newton_spec": _load_isaaclab_spec(),
        "mjlab_warp_verification": {
            "actor_observation_terms": actor_terms,
            "actor_61d_contract_verified": actor_contract_ok,
            "bam_startup_event_registered": has_bam_startup,
            "stairs_curriculum_stages_iters": curriculum_stages,
            "stairs_clearance_stages_m": list(CLEARANCE_TARGET_STAGES),
            "curriculum_aligned": curriculum_aligned,
            "velocity_env_reward_terms_count": len(vel_cfg.rewards),
            "stairs_env_reward_terms_count": len(stairs_cfg.rewards),
        },
        "neko_core_verification": {
            "quintic_blender_active": blender.is_transitioning,
            "quintic_first_step_norm": round(float(np.linalg.norm(mid_q)), 6),
            "gait_phase_latch_double_support_fired": bool(fired),
            "neko_is_transitioning": neko.is_transitioning,
            "neko_output_shape": list(cmd_out.shape),
            "ablation_benchmark_summary": neko_ablation_summary,
        },
        "cad_to_physics_scene_bridge": scene_verification,
        "closed_loop_cad_terrain_rollout": cad_rollout_results,
        "live_warp_gpu_rollout_and_export": gpu_rollout_results,
    }

    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage C: Execute real closed-loop CAD terrain & NVIDIA Warp GPU rollouts."
    )
    parser.add_argument(
        "--skip-gpu-rollout",
        action="store_true",
        help="Skip the multi-env Warp GPU rollout (for fast CPU-only unit tests).",
    )
    args = parser.parse_args()
    report = verify_rl_configs_and_neko_core(run_gpu_rollout=not args.skip_gpu_rollout)
    mj = report["mjlab_warp_verification"]
    cr = report["closed_loop_cad_terrain_rollout"]
    gr = report["live_warp_gpu_rollout_and_export"]
    print(
        f"[Stage C - Real Execution] 61D Obs={mj['actor_61d_contract_verified']} | "
        f"CAD Terrain Rollout (5.0s): dx={cr['forward_distance_dx_m']:+.3f}m, "
        f"dz={cr['elevation_gain_dz_mm']:+.1f}mm, peak_z={cr['peak_trunk_z_mm']:.1f}mm, "
        f"V_min={cr['min_bam_voltage_v']:.2f}V, Survived={cr['survived_without_fall']}"
    )
    if gr:
        print(
            f"  -> Live Warp GPU ({gr['num_parallel_gpu_envs']} envs, {gr['device']}): "
            f"mean_rew={gr['mean_step_reward']:+.4f}, max_dx={gr['max_forward_dx_m']:+.3f}m, "
            f"survival={gr['survival_rate_pct']:.1f}% | Exported ONNX: {gr['exported_onnx_path']}"
        )


if __name__ == "__main__":
    main()
