import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, PathPatch
from matplotlib.path import Path
import imageio.v2 as iio

# 1920x1080 setup
DPI = 100
WIDTH_PX = 1920
HEIGHT_PX = 1080
FIGSIZE = (WIDTH_PX / DPI, HEIGHT_PX / DPI)
FPS = 30
DURATION_SEC = 16  # 16s animation rich in detail
TOTAL_FRAMES = DURATION_SEC * FPS

print(f"Generating Neko Core Architecture Animation: {TOTAL_FRAMES} frames @ {FPS} fps...")

# Color Palette: White & Tech Blue (Clean Academic & High-tech)
BG_COLOR = '#F8FAFC'       # Slate-50 Pure Tech White
CARD_BG = '#FFFFFF'        # Crisp Card White
PRIMARY_BLUE = '#0284C7'   # Sky/Tech Blue
DARK_BLUE = '#0F172A'      # Deep Navy Text/Header
NAVY_SURFACE = '#1E293B'   # Dark Slate
CYAN_ACCENT = '#0EA5E9'    # Active Electric Cyan
CYAN_GLOW = '#38BDF8'      # Soft Cyan
LIGHT_BLUE_BG = '#F0F9FF'  # Sky-50 Accent surface
BORDER_COLOR = '#E2E8F0'   # Subtle Slate Border
SUCCESS_TEAL = '#0D9488'   # Safety Governor Green-Teal
ORANGE_WARN = '#EA580C'    # Bang-bang shock indicator

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

def draw_frame(frame_idx):
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.axis('off')

    t = frame_idx / FPS  # Current time in seconds
    progress = frame_idx / TOTAL_FRAMES

    # Background
    ax.fill([0, 1920, 1920, 0], [0, 0, 1080, 1080], color=BG_COLOR)

    # Subtle Grid Pattern
    for x in range(0, 1920, 60):
        ax.plot([x, x], [0, 1080], color='#F1F5F9', lw=1, zorder=1)
    for y in range(0, 1080, 60):
        ax.plot([0, 1920], [y, y], color='#F1F5F9', lw=1, zorder=1)

    # Top Header Banner
    ax.text(80, 1020, "NEKO CORE · MULTI-MODEL POLICY ORCHESTRATION ARCHITECTURE", 
            fontsize=17, fontweight='bold', color=DARK_BLUE, fontfamily='DejaVu Sans', zorder=10)
    ax.text(80, 995, "Unified 61D Observation Contract  |  S-Curve Slewed Blending  |  Kinematic Phase-Lock Governor", 
            fontsize=11.5, color='#64748B', fontfamily='DejaVu Sans', zorder=10)

    # Live Status Badges
    badge_x = 1620
    freq_text = "SYNC: 50 Hz  (dt = 20ms)"
    p_badge = FancyBboxPatch((badge_x - 140, 1000), 220, 36, boxstyle="round,pad=5,rounding_size=8",
                             facecolor=LIGHT_BLUE_BG, edgecolor=PRIMARY_BLUE, lw=1.5, zorder=10)
    ax.add_patch(p_badge)
    ax.text(badge_x - 30, 1018, freq_text, fontsize=9.5, fontweight='bold', color=PRIMARY_BLUE, 
            ha='center', va='center', fontfamily='DejaVu Sans', zorder=11)

    # Time / Phase indicator
    phase_str = "MONOLITHIC RUN" if t < 3.0 else ("COMMAND TRIGGER: ROULADE" if t < 6.5 else ("S-CURVE BLENDING (0-100%)" if t < 11.5 else "POST-TRANSITION STEADY"))
    ax.text(badge_x + 130, 1018, f"T={t:04.1f}s", fontsize=10, color=DARK_BLUE, fontweight='bold', ha='left', va='center', zorder=11)

    # -------------------------------------------------------------
    # 4 ARCHITECTURAL PILLARS (ANATOMY VIEW)
    # -------------------------------------------------------------

    # PILLAR 1: SENSORY & 61D CONTRACT (Left Column: x=70 to 450)
    p1 = FancyBboxPatch((70, 160), 380, 780, boxstyle="round,pad=10,rounding_size=16",
                        facecolor=CARD_BG, edgecolor=BORDER_COLOR, lw=2, zorder=2)
    ax.add_patch(p1)
    ax.text(90, 905, "01 / OBSERVATION CONTRACT", fontsize=12, fontweight='bold', color=PRIMARY_BLUE, zorder=5)
    ax.text(90, 882, "Unified 61D Actor Space", fontsize=16, fontweight='bold', color=DARK_BLUE, zorder=5)

    # 48D Proprioception Box
    p1_sub1 = FancyBboxPatch((90, 520), 340, 330, boxstyle="round,pad=6,rounding_size=10",
                             facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.2, zorder=3)
    ax.add_patch(p1_sub1)
    ax.text(105, 825, "48D Proprioception (Continuous)", fontsize=11, fontweight='bold', color=DARK_BLUE, zorder=5)
    
    proprio_items = [
        ("Base Ang Vel (IMU 3D)", "[-0.02, +0.14, -0.05] rad/s"),
        ("Projected Gravity (3D)", "[gx, gy, gz] ~ [0, 0, -1]"),
        ("Joint Positions (14 DOF)", "Left(5) + Neck(4) + Right(5)"),
        ("Joint Velocities (14 DOF)", "Encoder reading @ 50Hz"),
        ("Action History (14 DOF)", "Prev ctrl step t-1")
    ]
    for i, (item, val) in enumerate(proprio_items):
        y_pos = 785 - i * 48
        # Signal pulse indicator
        pulse = 0.5 + 0.5 * math.sin(t * 8 + i)
        ax.plot([105, 115], [y_pos + 6, y_pos + 6], color=CYAN_ACCENT, lw=3, alpha=pulse, zorder=5)
        ax.text(125, y_pos + 8, item, fontsize=9.5, fontweight='bold', color='#334155', zorder=5)
        ax.text(125, y_pos - 6, val, fontsize=8.5, color='#64748B', fontfamily='DejaVu Sans Mono', zorder=5)

    # 13D Command Box (Zero-padded slots)
    p1_sub2 = FancyBboxPatch((90, 190), 340, 305, boxstyle="round,pad=6,rounding_size=10",
                             facecolor=LIGHT_BLUE_BG, edgecolor='#BAE6FD', lw=1.2, zorder=3)
    ax.add_patch(p1_sub2)
    ax.text(105, 465, "13D Command Bus (Zero-Padded)", fontsize=11, fontweight='bold', color=PRIMARY_BLUE, zorder=5)

    cmd_items = [
        ("Twist Command [3D]", "vx, vy, yaw_rate", "Active [0.25, 0, 0]"),
        ("Head Posture [4D]", "neck_pitch, yaw, roll", "Hold Neutral / Gaze"),
        ("Body 6D Pose [6D]", "x, y, z, roll, pitch, yaw", "Zero-Padded Alive")
    ]
    for i, (c_name, c_desc, c_state) in enumerate(cmd_items):
        y_pos = 425 - i * 65
        ax.text(105, y_pos + 12, c_name, fontsize=9.5, fontweight='bold', color=DARK_BLUE, zorder=5)
        ax.text(105, y_pos - 2, c_desc, fontsize=8.5, color='#64748B', zorder=5)
        # Active tag
        tag_color = CYAN_ACCENT if i == 0 else '#94A3B8'
        ax.text(105, y_pos - 18, f"● {c_state}", fontsize=8, color=tag_color, fontweight='bold', zorder=5)


    # PILLAR 2: NEKO CORE ARBITER & TRANSITION ENGINE (Center-Left: x=480 to 970)
    p2 = FancyBboxPatch((480, 160), 490, 780, boxstyle="round,pad=10,rounding_size=16",
                        facecolor=CARD_BG, edgecolor=BORDER_COLOR, lw=2, zorder=2)
    ax.add_patch(p2)
    ax.text(505, 905, "02 / TRANSITION & GOVERNANCE", fontsize=12, fontweight='bold', color=PRIMARY_BLUE, zorder=5)
    ax.text(505, 882, "Neko Core Arbiter Engine", fontsize=16, fontweight='bold', color=DARK_BLUE, zorder=5)

    # State Machine & Arbitration Mechanism
    ax.text(505, 835, "Arbitration State Machine", fontsize=11.5, fontweight='bold', color='#1E293B', zorder=5)
    
    # State pills
    states = ["IDLE/STANCE", "LOCOMOTION", "SLEWED BLEND", "DYNAMIC TRICK"]
    # active state index based on time
    if t < 3.0:
        active_st = 1
    elif t < 6.5:
        active_st = 2
    elif t < 11.5:
        active_st = 2
    else:
        active_st = 3

    for idx, st_name in enumerate(states):
        st_x = 505 + (idx % 2) * 225
        st_y = 795 - (idx // 2) * 44
        is_active = (idx == active_st)
        bg = PRIMARY_BLUE if is_active else '#F1F5F9'
        fg = '#FFFFFF' if is_active else '#475569'
        bd = PRIMARY_BLUE if is_active else '#E2E8F0'
        p_st = FancyBboxPatch((st_x, st_y), 215, 34, boxstyle="round,pad=4,rounding_size=8",
                              facecolor=bg, edgecolor=bd, lw=1.2, zorder=5)
        ax.add_patch(p_st)
        ax.text(st_x + 107, st_y + 17, st_name, fontsize=9, fontweight='bold', color=fg,
                ha='center', va='center', zorder=6)

    # Mathematical Transition Graph: S-Curve Slewed Blending Formula & Curve
    ax.text(505, 685, "Zero-Jerk S-Curve Quintic Ramp", fontsize=11.5, fontweight='bold', color=DARK_BLUE, zorder=5)
    ax.text(505, 665, r"$\alpha(t) = 10\tau^3 - 15\tau^4 + 6\tau^5 \quad (\dot{\alpha}(0)=\dot{\alpha}(1)=0, \ddot{\alpha}(0)=\ddot{\alpha}(1)=0)$",
            fontsize=9.5, color='#0284C7', fontfamily='DejaVu Sans', zorder=5)

    # Mini graph box
    p_graph = FancyBboxPatch((505, 410), 440, 240, boxstyle="round,pad=6,rounding_size=10",
                             facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.2, zorder=3)
    ax.add_patch(p_graph)

    # Plot S-Curve inside mini graph
    gx = np.linspace(0, 1, 100)
    gy_scurve = 10 * gx**3 - 15 * gx**4 + 6 * gx**5
    gy_bang = np.where(gx < 0.5, 0.0, 1.0)

    # Graph axis coordinate mapping
    # box is x=525 to 925 (width 400), y=430 to 620 (height 190)
    px_curve = 535 + gx * 380
    py_scurve = 440 + gy_scurve * 160
    py_bang = 440 + gy_bang * 160

    # Draw Naive Bang-bang (dashed red)
    ax.plot(px_curve, py_bang, color=ORANGE_WARN, lw=1.8, linestyle='--', label="Naive Hard Switch (Jerk shock)", zorder=5)
    # Draw Neko S-Curve (solid vibrant blue)
    ax.plot(px_curve, py_scurve, color=PRIMARY_BLUE, lw=3, label="Neko S-Curve Blend (Zero Jerk)", zorder=6)

    # Current blending cursor
    if 3.0 <= t <= 11.5:
        tau_val = min(max((t - 3.0) / (8.5), 0.0), 1.0)
        curr_alpha = 10 * tau_val**3 - 15 * tau_val**4 + 6 * tau_val**5
        cx = 535 + tau_val * 380
        cy = 440 + curr_alpha * 160
        ax.plot([cx, cx], [430, 620], color=CYAN_ACCENT, lw=1.5, linestyle=':', zorder=7)
        ax.plot([cx], [cy], marker='o', markersize=9, color=CYAN_ACCENT, markeredgecolor=DARK_BLUE, markeredgewidth=2, zorder=8)
        ax.text(cx + 10, cy - 10, f"Blend: {curr_alpha*100:.1f}%", fontsize=9, fontweight='bold', color=DARK_BLUE, zorder=8)

    ax.legend(loc='upper left', bbox_to_anchor=(0.28, 0.59), fontsize=8.5, frameon=False)
    ax.text(535, 422, "t = 0 (Policy A)", fontsize=8, color='#64748B', zorder=5)
    ax.text(850, 422, "t = T_trans (Policy B)", fontsize=8, color='#64748B', zorder=5)

    # Kinematic Safety Governor
    p_gov = FancyBboxPatch((505, 190), 440, 195, boxstyle="round,pad=6,rounding_size=10",
                           facecolor='#F0FDF4', edgecolor='#86EFAC', lw=1.2, zorder=3)
    ax.add_patch(p_gov)
    ax.text(525, 355, "Kinematic Safety Governor & Phase Latch", fontsize=11, fontweight='bold', color=SUCCESS_TEAL, zorder=5)
    
    gov_rules = [
        ("Tilt Safety Angle", "Lock transition if roll/pitch > 22°", "PASS (4.2° < 22°)"),
        ("Gait Phase Latch", "Wait for bilateral foot contact confirmation", "LATCHED / SAFE"),
        ("Velocity Residual Gate", "Prevent high kinetic momentum collision", "|Δv| < 0.08 m/s OK")
    ]
    for i, (r_name, r_rule, r_status) in enumerate(gov_rules):
        y_pos = 325 - i * 42
        ax.text(525, y_pos + 6, r_name, fontsize=9, fontweight='bold', color=DARK_BLUE, zorder=5)
        ax.text(525, y_pos - 8, r_rule, fontsize=8, color='#475569', zorder=5)
        ax.text(820, y_pos + 6, f"✓ {r_status}", fontsize=8, color=SUCCESS_TEAL, fontweight='bold', ha='right', zorder=5)


    # PILLAR 3: MULTI-MODEL POLICY POOL (Center-Right: x=1000 to 1440)
    p3 = FancyBboxPatch((1000, 160), 440, 780, boxstyle="round,pad=10,rounding_size=16",
                        facecolor=CARD_BG, edgecolor=BORDER_COLOR, lw=2, zorder=2)
    ax.add_patch(p3)
    ax.text(1025, 905, "03 / POLICY EXPERT BANK", fontsize=12, fontweight='bold', color=PRIMARY_BLUE, zorder=5)
    ax.text(1025, 882, "Modular Policy Suite", fontsize=16, fontweight='bold', color=DARK_BLUE, zorder=5)

    policies = [
        ("Walking Locomotion", "Velocity flat/rough (4096 envs)", "Active Baseline", PRIMARY_BLUE if active_st in [1, 2] else '#CBD5E1'),
        ("Dynamic Somersault (Roulade)", "Acrobatic pitch tumble & recovery", "Swapping In", CYAN_ACCENT if active_st in [2, 3] else '#CBD5E1'),
        ("Interactive Fetch & Track", "Head-gaze lock & beak ball retrieve", "Standby / Warm", '#CBD5E1'),
        ("Sit-to-Stand / VelStand", "Posture flag twist[0] controlled", "Standby / Warm", '#CBD5E1'),
        ("Roller Skates / Slope", "Passive rolling + inverted friction", "Standby / Warm", '#CBD5E1'),
        ("Stairs Climbing Specialist", "High clearance & sole leveling", "Standby / Warm", '#CBD5E1')
    ]

    for i, (pol_name, pol_desc, pol_tag, border_col) in enumerate(policies):
        py = 780 - i * 98
        is_highlight = (border_col in [PRIMARY_BLUE, CYAN_ACCENT])
        card_fill = '#F0F9FF' if is_highlight else '#F8FAFC'
        card_p = FancyBboxPatch((1025, py), 390, 84, boxstyle="round,pad=5,rounding_size=8",
                               facecolor=card_fill, edgecolor=border_col, lw=2 if is_highlight else 1, zorder=3)
        ax.add_patch(card_p)
        ax.text(1045, py + 55, pol_name, fontsize=10.5, fontweight='bold', color=DARK_BLUE if is_highlight else '#475569', zorder=5)
        ax.text(1045, py + 34, pol_desc, fontsize=8.5, color='#64748B', zorder=5)
        ax.text(1045, py + 14, f"● ONNX Model · {pol_tag}", fontsize=8, color=border_col if is_highlight else '#94A3B8', fontweight='bold', zorder=5)


    # PILLAR 4: HARDWARE & BAM ACTUATOR (Right Column: x=1470 to 1850)
    p4 = FancyBboxPatch((1470, 160), 380, 780, boxstyle="round,pad=10,rounding_size=16",
                        facecolor=CARD_BG, edgecolor=BORDER_COLOR, lw=2, zorder=2)
    ax.add_patch(p4)
    ax.text(1495, 905, "04 / ACTUATOR TRANSFER", fontsize=12, fontweight='bold', color=PRIMARY_BLUE, zorder=5)
    ax.text(1495, 882, "BAM M6 & XL330", fontsize=16, fontweight='bold', color=DARK_BLUE, zorder=5)

    # Voltage & Motor Dynamics Box
    p4_sub1 = FancyBboxPatch((1495, 530), 330, 320, boxstyle="round,pad=6,rounding_size=10",
                             facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.2, zorder=3)
    ax.add_patch(p4_sub1)
    ax.text(1510, 815, "Voltage Control & Sag Compensation", fontsize=10.5, fontweight='bold', color=DARK_BLUE, zorder=5)
    
    act_specs = [
        ("Motor Model", "14x Dynamixel XL330"),
        ("Vin Dynamic Range", "6.5V - 8.2V (DR in sim)"),
        ("Voltage Sag Law", "V_drop = gain * Σ|τ|"),
        ("Force Limit", "Vin * kt / R (bounded)"),
        ("Backlash DR", "2.0° Backlash Twin Model")
    ]
    for i, (k_sp, v_sp) in enumerate(act_specs):
        y_pos = 775 - i * 48
        ax.text(1510, y_pos + 6, k_sp, fontsize=9, fontweight='bold', color='#334155', zorder=5)
        ax.text(1510, y_pos - 8, v_sp, fontsize=8.5, color=PRIMARY_BLUE, fontfamily='DejaVu Sans Mono', zorder=5)

    # 14 Actuator Channels Live Gauge
    p4_sub2 = FancyBboxPatch((1495, 190), 330, 315, boxstyle="round,pad=6,rounding_size=10",
                             facecolor=LIGHT_BLUE_BG, edgecolor='#BAE6FD', lw=1.2, zorder=3)
    ax.add_patch(p4_sub2)
    ax.text(1510, 475, "14 Joint Channels Output (50Hz)", fontsize=10.5, fontweight='bold', color=PRIMARY_BLUE, zorder=5)

    # Draw 14 miniature servo output bars
    leg_names = ["L-Yaw", "L-Roll", "L-Pitch", "L-Knee", "L-Ank", "Neck", "H-P", "H-Y", "H-R", "R-Yaw", "R-Roll", "R-Pitch", "R-Knee", "R-Ank"]
    for i in range(14):
        bx = 1515 + (i % 7) * 42
        by = 380 if i < 7 else 270
        # Simulated joint torque oscillation
        torque_val = 0.5 + 0.4 * math.sin(t * 12 + i * 0.8)
        bar_h = torque_val * 45
        # Bar frame
        ax.fill([bx, bx + 28, bx + 28, bx], [by, by, by + 50, by + 50], color='#E2E8F0', zorder=4)
        # Bar level
        bar_color = PRIMARY_BLUE if i < 5 else (CYAN_ACCENT if i < 9 else SUCCESS_TEAL)
        ax.fill([bx, bx + 28, bx + 28, bx], [by, by, by + bar_h, by + bar_h], color=bar_color, zorder=5)
        ax.text(bx + 14, by - 12, leg_names[i], fontsize=6.8, ha='center', color='#475569', zorder=5)

    # Bottom Footer & Legend
    ax.plot([70, 1850], [130, 130], color='#CBD5E1', lw=1, zorder=5)
    ax.text(80, 95, "NEKO CORE EMBODIED ORCHESTRATION SYSTEM", fontsize=11, fontweight='bold', color=DARK_BLUE, zorder=5)
    ax.text(80, 75, "Designed for Microduck Bipedal Robot · Proven Generalizable to Humanoid & Quadruped Platforms", fontsize=9.5, color='#64748B', zorder=5)
    
    # Legend right
    ax.text(1520, 85, "● 50 Hz Synchronous Loop", fontsize=9, color=PRIMARY_BLUE, fontweight='bold', zorder=5)
    ax.text(1700, 85, "● Sim-to-Real Verified", fontsize=9, color=SUCCESS_TEAL, fontweight='bold', zorder=5)

    # Dynamic Connecting Data Flow Pulses (Between Pillars)
    pulse_pos = (t * 2.5) % 1.0
    # From P1 to P2
    conn_y = 500
    ax.plot([450, 480], [conn_y, conn_y], color=CYAN_ACCENT, lw=2, linestyle=':', zorder=4)
    cx_p = 450 + pulse_pos * 30
    ax.plot([cx_p], [conn_y], marker='>', markersize=7, color=CYAN_ACCENT, zorder=6)

    # From P2 to P3
    ax.plot([970, 1000], [conn_y, conn_y], color=CYAN_ACCENT, lw=2, linestyle=':', zorder=4)
    cx_p2 = 970 + pulse_pos * 30
    ax.plot([cx_p2], [conn_y], marker='>', markersize=7, color=CYAN_ACCENT, zorder=6)

    # From P3 to P4
    ax.plot([1440, 1470], [conn_y, conn_y], color=CYAN_ACCENT, lw=2, linestyle=':', zorder=4)
    cx_p3 = 1440 + pulse_pos * 30
    ax.plot([cx_p3], [conn_y], marker='>', markersize=7, color=CYAN_ACCENT, zorder=6)

    fig.canvas.draw()
    # Extract RGB array
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3]
    return image

# Write MP4 Video
output_video_path = "neko_core_architecture_animation.mp4"
writer = iio.get_writer(output_video_path, fps=FPS, codec='libx264', quality=8)

for frame in range(TOTAL_FRAMES):
    img = draw_frame(frame)
    writer.append_data(img)
    if frame % 60 == 0:
        print(f"Rendered frame {frame}/{TOTAL_FRAMES} ({(frame/TOTAL_FRAMES)*100:.1f}%)")

writer.close()
plt.close(fig)
print(f"Video saved successfully to {output_video_path}!")
