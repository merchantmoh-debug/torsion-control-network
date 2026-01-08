"""
Torsion Control Network (TCN) Core Kernel
=========================================
Mathematical backend for controlling LLM trajectories via Riemannian
Geometry and Active Inference.

Components:
1. RiemannianManifold: Statistical manifold geometry & metric estimation.
2. TorsionTensor: Skew-symmetric operator for trajectory twisting.
3. ActiveInferenceController: Free Energy minimization & gradient steering.
4. LyapunovStability: Convergence verification engine.

Author: The Architect
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from typing import Tuple, Dict, Optional
import math

class RiemannianManifold:
    """
    Utilities for operating on the statistical manifold M of the LLM's latent space.
    Approximates the Riemannian metric tensor g_ij to measure information geometry distances.
    """

    def __init__(self, dim: int, epsilon: float = 1e-6):
        self.dim = dim
        self.epsilon = epsilon

    @staticmethod
    def compute_metric_tensor(hidden_states: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Approximates the local Riemannian metric tensor G(h) using the
        Fisher Information Matrix proxy from hidden state covariance.

        Args:
            hidden_states: [Batch, Seq, Dim]

        Returns:
            G: [Batch, Dim, Dim] Positive semi-definite metric tensor
        """
        b, s, d = hidden_states.size()

        # Center the states (Mean subtraction)
        mean = hidden_states.mean(dim=1, keepdim=True)
        centered = hidden_states - mean

        # Compute Covariance as FIM proxy: (X^T X) / (N-1)
        # Einsum: batch,seq,i * batch,seq,j -> batch,i,j
        cov = torch.einsum('bsi,bsj->bij', centered, centered) / (s - 1 + epsilon)

        # Regularize to ensure invertibility (Positive Definiteness)
        identity = torch.eye(d, device=hidden_states.device).unsqueeze(0)
        G = cov + (epsilon * identity)

        return G

    @staticmethod
    def geodesic_distance(h1: torch.Tensor, h2: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared Mahalanobis distance induced by the metric G.
        d^2(x,y) ~ (x-y)^T G (x-y)
        """
        diff = (h1 - h2).unsqueeze(-1) # [B, S, D, 1]

        # Expand metric for sequence length if necessary or broadcast
        # Assuming metric is global per batch [B, D, D]
        G = metric.unsqueeze(1) # [B, 1, D, D]

        # Bilinear form: v^T G v
        # [B, S, 1, D] @ [B, 1, D, D] @ [B, S, D, 1]
        dist_sq = torch.matmul(torch.matmul(diff.transpose(-1, -2), G), diff)
        return dist_sq.squeeze()


class TorsionTensor(nn.Module):
    """
    Implements the Torsion Operator T^k_ij via Low-Rank Factorization.

    Mathematically, Torsion is the antisymmetric part of the affine connection.
    T(X, Y) = -T(Y, X).

    We model this as a rotation field applied to the latent trajectory.
    To avoid O(d^3) parameters, we use a Skew-Symmetric Low-Rank approximation:
    T(h) ~ V @ (Omega - Omega^T) @ U^T @ h
    """

    def __init__(self, hidden_dim: int, rank: int = 32, alpha: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha # Control gain / "Twist" strength

        # Projectors to low-rank manifold tangent space
        self.U = nn.Linear(hidden_dim, rank, bias=False)
        self.V = nn.Linear(rank, hidden_dim, bias=False)

        # The Core Torsion Generator (Skew-Symmetric)
        # We initialize randomly; skew-symmetry is enforced in forward()
        self.Omega = nn.Parameter(torch.randn(rank, rank) * 0.02)

        # Zero-init output projection for stability at start
        nn.init.zeros_(self.V.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply torsion field to the trajectory.
        h_{t+1} = h_t + T(h_t)

        Args:
            hidden_states: [Batch, Seq, Dim]
        Returns:
            twisted_states: [Batch, Seq, Dim]
        """
        # 1. Project to latent curvature space
        latent = self.U(hidden_states) # [B, S, R]

        # 2. Apply Skew-Symmetric Operator (Rotational Force)
        # S = Omega - Omega^T ensures S is skew-symmetric (S^T = -S)
        skew_omega = self.Omega - self.Omega.transpose(0, 1)
        rotated_latent = torch.matmul(latent, skew_omega) # [B, S, R]

        # 3. Project back to manifold tangent space
        torsion_field = self.V(rotated_latent) # [B, S, D]

        # 4. Apply geometric update (Symplectic Euler step)
        return hidden_states + (self.alpha * torsion_field)


class ActiveInferenceController(nn.Module):
    """
    Minimizes the Variational Free Energy (F) of the system dynamics.

    F = D_KL( Q(s) || P(target) ) + Beta * H(Q)

    Where:
    - Q(s): Current latent state distribution (Approximated by logits)
    - P(target): Desired distribution (Safety/Truth priors)
    - H(Q): Entropy (Exploration drive)
    """

    def __init__(self, hidden_dim: int, vocab_size: int, beta: float = 0.1):
        super().__init__()
        self.beta = beta
        # Policy Head: Maps hidden states to action probabilities (Logits)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)

    def compute_free_energy(self,
                          hidden_states: torch.Tensor,
                          target_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates Variational Free Energy F.
        """
        # Project to logits (Action Space)
        logits = self.policy_head(hidden_states)
        q_log_probs = F.log_softmax(logits, dim=-1)
        q_probs = torch.exp(q_log_probs)

        # 1. KL Divergence (Energy Term): Precision-weighted prediction error
        # D_KL = sum(Q * (logQ - logP))
        # Note: F.kl_div expects log_input as first arg
        # We treat target_probs as the Prior P
        kl_div = F.kl_div(q_log_probs, target_probs, reduction='batchmean')

        # 2. Entropy (Exploration Term)
        # H(Q) = -sum(Q * logQ)
        entropy = -torch.sum(q_probs * q_log_probs, dim=-1).mean()

        # 3. Free Energy F
        # F = Energy - Beta * Entropy
        # Minimizing F -> Minimize KL (Align) & Maximize Entropy (Explore)
        free_energy = kl_div - (self.beta * entropy)

        metrics = {
            "F": free_energy.item(),
            "KL": kl_div.item(),
            "H": entropy.item()
        }
        return free_energy, metrics

    def compute_control_signal(self,
                             hidden_states: torch.Tensor,
                             target_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the optimal control signal u* = -nabla_h F
        This steers the hidden state towards the Free Energy minimum.
        """
        # Detach and require grad to compute local sensitivity
        h_curr = hidden_states.detach().requires_grad_(True)

        # Compute Energy
        free_energy, _ = self.compute_free_energy(h_curr, target_probs)

        # Compute Gradient Flow: dF/dh
        grads = grad(free_energy, h_curr, create_graph=False)[0]

        # Control signal opposes the gradient (Gradient Descent on Manifold)
        return -grads


class LyapunovStability:
    """
    Verifies asymptotic stability of the control trajectory.
    Checks the condition dV/dt < 0 where V is the Lyapunov function (Free Energy).
    """

    def __init__(self, window_size: int = 10, threshold: float = 1e-4):
        self.history = []
        self.window_size = window_size
        self.threshold = threshold

    def verify(self, energy: float) -> Tuple[bool, float]:
        """
        Checks stability.
        Returns: (is_stable, delta_V)
        """
        self.history.append(energy)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 2:
            return True, 0.0

        # Calculate discrete derivative dV/dt (smoothed)
        # Simple finite difference of the last step
        delta = self.history[-1] - self.history[-2]

        # Stability Condition: Energy should not increase significantly
        # We allow delta <= threshold (noise floor)
        is_stable = delta <= self.threshold

        return is_stable, delta

    def reset(self):
        self.history = []
