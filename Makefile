# Makefile for Microduck Argon Sim-to-Real Pipeline
# Single-Model Argon Architect (Stage A: FreeCAD/CAD -> Stage B: Blender/OpenUSD -> Stage C: Newton/IsaacLab/Warp RL)

.PHONY: all audit cad usd rl test ablation smoke-stairs help

all: cad usd rl
	@echo "[Argon Pipeline] End-to-End Pipeline (Stage A -> Stage B -> Stage C) completed."

audit:
	@cat AUDIT_REPORT.md | head -n 45

cad:
	PYTHONPATH="" uv run python 01_freecad_cad/parametric_cad_generator.py

usd:
	PYTHONPATH="" uv run python 02_blender_omniverse_usd/blender_mesh_processor.py
	PYTHONPATH="" uv run python 02_blender_omniverse_usd/mjcf_to_urdf_usd_bridge.py

rl:
	PYTHONPATH="" uv run python 03_newton_isaaclab_rl/verify_rl_pipeline.py

pipeline:
	PYTHONPATH="" uv run python run_argon_pipeline.py --stage all

ablation:
	PYTHONPATH="" uv run python scripts/benchmark_neko_core_ablation.py

test:
	PYTHONPATH="" uv run --with pytest pytest tests/ -q

smoke-stairs:
	PYTHONPATH="" uv run train Mjlab-Stairs-Climb-Microduck --env.scene.num-envs 64 --agent.max_iterations 5

help:
	@echo "Available targets:"
	@echo "  make all          - Run Stage A (CAD), Stage B (URDF/OpenUSD), and Stage C (RL/Warp verification)"
	@echo "  make cad          - Stage A: Generate parametric .STEP, .STL, .OBJ models"
	@echo "  make usd          - Stage B: Normalize 54 robot meshes & export microduck.urdf + SimReady .usda"
	@echo "  make rl           - Stage C: Verify 61D RL contract, BAM M6, Neko Core & CAD-to-Physics bridge"
	@echo "  make ablation     - Run Neko Core 4-method quantitative ablation benchmark"
	@echo "  make test         - Run full pytest regression suite"
	@echo "  make smoke-stairs - Run 5-iteration GPU smoke test for High Stairs curriculum"

