"""Hierarchical Skill Coordination & Dynamic Action Blending.

Implements smooth manifold transition between policies without action discontinuity,
preventing BAM actuator current spikes and voltage drop (V_in < 6.0V).
Reference: Peng et al., ASE (SIGGRAPH 2021).
"""

from __future__ import annotations
import numpy as np


class SkillBlender:
    """First-order action blending filter for smooth skill transitions."""

    def __init__(self, n_actions: int = 14, transition_duration_s: float = 0.25, control_dt: float = 0.02):
        self.n_actions = n_actions
        self.transition_duration_s = transition_duration_s
        self.control_dt = control_dt
        self.current_action = np.zeros(n_actions, dtype=np.float32)
        self.source_action = np.zeros(n_actions, dtype=np.float32)
        self.transition_progress = 1.0  # 1.0 = fully transitioned

    def start_transition(self, from_action: np.ndarray):
        """Initiate transition from previous policy action."""
        self.source_action = from_action.copy()
        self.transition_progress = 0.0

    def step(self, target_action: np.ndarray) -> np.ndarray:
        """Blend action continuously towards target."""
        if self.transition_progress >= 1.0:
            self.current_action = target_action.copy()
            return self.current_action

        self.transition_progress = min(1.0, self.transition_progress + (self.control_dt / self.transition_duration_s))
        gamma = self.transition_progress
        self.current_action = (1.0 - gamma) * self.source_action + gamma * target_action
        return self.current_action
