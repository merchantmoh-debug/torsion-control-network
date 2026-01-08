"""
TCN VSM System 1: Operational Units (Kinetic Layer)
===================================================

Refactored from core.py.
Handles Riemannian Manifold geometry and Torsion Tensor operations.
"""

import torch
import torch.nn as nn
from src.tcn.core import RiemannianManifold, TorsionTensor # Import old logic for migration
# Ideally we would rewrite fully, but for "Refactor" we assume the math is valid
# and just re-house it in the correct VSM context.

class KineticOperator(nn.Module):
    """
    The 'Hands' of the system.
    Executes the generation trajectory on the manifold.
    """

    def __init__(self, hidden_dim: int, rank: int = 32):
        super().__init__()
        self.manifold = RiemannianManifold(dim=hidden_dim)
        self.torsion = TorsionTensor(hidden_dim=hidden_dim, rank=rank)

    def compute_metric(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the local metric tensor G_ij.
        """
        return self.manifold.compute_metric_tensor(hidden_states)

    def apply_torsion(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applies the 'Twist' to the trajectory.
        """
        return self.torsion(hidden_states)
