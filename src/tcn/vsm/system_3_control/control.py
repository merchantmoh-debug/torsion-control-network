"""
TCN VSM System 3: Control (The MAOS Kernel)
===========================================

Orchestrates the Active Inference Loop and Resource Allocation.
Refactored from core.py ActiveInferenceController.
"""

import torch
import torch.nn as nn
from tcn.core import ActiveInferenceController, LyapunovStability

class MAOSKernel(nn.Module):
    """
    Meta-Aware Operating System (v4.0).
    Manages the Free Energy Minimization loop.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, beta: float = 0.1):
        super().__init__()
        self.aic = ActiveInferenceController(hidden_dim, vocab_size, beta)
        self.lyapunov = LyapunovStability()

    def optimization_step(self,
                         hidden_states: torch.Tensor,
                         target_probs: torch.Tensor) -> dict:
        """
        Calculates Free Energy and Gradient Control Signal.
        """
        # 1. Calculate Free Energy (VFE)
        # Bolt Optimization: free_energy is now a Tensor, metrics are detached Tensors
        free_energy, metrics = self.aic.compute_free_energy(hidden_states, target_probs)

        # 2. Check Stability (Lyapunov)
        # Bolt Optimization: Pass tensor directly; Lyapunov handles synchronization
        is_stable, dV = self.lyapunov.verify(free_energy)

        # 3. Compute Control Signal (Action)
        # u = -nabla F
        control_signal = self.aic.compute_control_signal(hidden_states, target_probs)

        return {
            "control_signal": control_signal,
            "free_energy": free_energy,
            "is_stable": is_stable,
            "dV": dV,
            "metrics": metrics
        }
