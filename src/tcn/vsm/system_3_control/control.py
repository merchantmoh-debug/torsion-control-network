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
        # Bolt Optimization: Use fused step
        # Disable redundant validation inside JIT loop (System 5 handles it)
        control_signal, free_energy, metrics = self.aic.compute_optimization_step(hidden_states, target_probs, validate=False)

        # 2. Check Stability (Lyapunov)
        # Bolt Optimization: Pass tensor directly to verify() to maintain graph continuity
        # LyapunovStability must handle tensors now.
        is_stable, dV = self.lyapunov.verify(free_energy)

        # Update metrics with the scalar values for logging if needed,
        # but keep tensors available if downstream needs them.
        # Here we just pass the metrics dict which has tensors.

        return {
            "control_signal": control_signal,
            "free_energy": free_energy,
            "is_stable": is_stable,
            "dV": dV,
            "metrics": metrics
        }
