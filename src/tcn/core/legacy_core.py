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
from typing import Tuple, Dict, Optional, Union, List, Any
import logging
import math

# Configure ARK Logger
logger = logging.getLogger("ARK.TCN")

# Bolt Optimization: JIT Compilation
# We try to import torch.compile. If not available (older torch), we use a dummy decorator.
try:
    from torch import compile as torch_compile
except ImportError:
    def torch_compile(func):
        return func

class RiemannianManifold:
    """
    Utilities for operating on the statistical manifold M of the LLM's latent space.
    Approximates the Riemannian metric tensor g_ij to measure information geometry distances.
    """

    def __init__(self, dim: int, epsilon: float = 1e-6):
        self.dim = dim
        self.epsilon = epsilon

    def __repr__(self):
        return f"<RiemannianManifold dim={self.dim} eps={self.epsilon}>"

    @staticmethod
    @torch_compile
    def compute_metric_tensor(hidden_states: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Approximates the local Riemannian metric tensor G(h) using the
        Fisher Information Matrix proxy from hidden state covariance.

        WARNING: This returns a dense [B, D, D] tensor. For high dimensions (D > 1024),
        this consumes O(D^2) memory. Consider using `compute_implicit_metric` for
        distance calculations.

        Args:
            hidden_states: [Batch, Seq, Dim]

        Returns:
            G: [Batch, Dim, Dim] Positive semi-definite metric tensor
        """
        # Sentinel: NaN Check
        if torch.isnan(hidden_states).any():
            raise ValueError("Input hidden_states contain NaNs.")

        b, s, d = hidden_states.size()

        # Sentinel: Sequence length check
        if s <= 1:
            logger.warning("Sequence length <= 1, covariance estimation will be unstable/degenerate.")

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
    def compute_implicit_metric(hidden_states: torch.Tensor, epsilon: float = 1e-6) -> Dict[str, torch.Tensor]:
        """
        Bolt Optimization: Computes components for implicit metric calculation.
        Avoids creating the [B, D, D] matrix.

        Returns a dictionary containing the centered states X.
        Distance ~ (1/(N-1)) ||X v||^2 + eps ||v||^2
        """
        if torch.isnan(hidden_states).any():
            raise ValueError("Input hidden_states contain NaNs.")

        mean = hidden_states.mean(dim=1, keepdim=True)
        centered = hidden_states - mean # [B, S, D]
        return {
            "centered": centered,
            "epsilon": torch.tensor(epsilon, device=hidden_states.device),
            "scale": torch.tensor(hidden_states.size(1) - 1 + epsilon, device=hidden_states.device)
        }

    @staticmethod
    @torch_compile
    def geodesic_distance(h1: torch.Tensor, h2: torch.Tensor, metric: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Computes the squared Mahalanobis distance induced by the metric G.
        d^2(x,y) ~ (x-y)^T G (x-y)

        Supports both dense G (Tensor) and implicit metric (Dict).
        """
        diff = h1 - h2 # [B, S, D]

        if isinstance(metric, dict) and "centered" in metric:
            # Bolt Fast Path: O(S^2 D) instead of O(S D^2)
            # v^T G v = v^T (1/k X^T X + eps I) v
            #         = 1/k (X v)^T (X v) + eps v^T v
            # But wait, X is [B, S_ref, D]. v is [B, S_tgt, D].
            # We treat X as the basis for the batch.
            # We want to compute this for each t in S_tgt.
            # (X v_t) is dot product of v_t with all ref vectors.

            X = metric["centered"] # [B, S_ref, D]
            scale = metric["scale"]
            eps = metric["epsilon"]

            # Term 1: 1/k || X v^T ||^2 ?
            # X: [B, S_ref, D]. diff: [B, S_tgt, D].
            # We want (diff @ X.T).
            # [B, S_tgt, D] @ [B, D, S_ref] -> [B, S_tgt, S_ref]

            # Use torch.matmul with transpose
            # X_T = X.transpose(1, 2) # [B, D, S_ref]
            proj = torch.matmul(diff, X.transpose(1, 2)) # [B, S_tgt, S_ref]

            # Squared norm of projections
            term1 = (proj ** 2).sum(dim=-1) / scale # [B, S_tgt]

            # Term 2: eps * ||v||^2
            term2 = eps * (diff ** 2).sum(dim=-1) # [B, S_tgt]

            return term1 + term2

        else:
            # Legacy Slow Path
            # Bolt: Memory safeguard
            if metric.dim() == 3 and metric.size(1) * metric.size(2) > 1e6:
                 logger.warning("Large metric tensor in legacy path. Consider using implicit metric for memory efficiency.")

            if metric.dim() == 3:
                G = metric.unsqueeze(1) # [B, 1, D, D]
            else:
                G = metric

            # diff is [B, S, D]. We need [B, S, D, 1] for bilinear form
            diff_unsqueezed = diff.unsqueeze(-1) # [B, S, D, 1]

            # v^T G v
            # [B, S, 1, D] @ [B, 1, D, D] @ [B, S, D, 1]
            # Warning: broadcasting G to [B, S, D, D] consumes huge memory!
            dist_sq = torch.matmul(torch.matmul(diff_unsqueezed.transpose(-1, -2), G), diff_unsqueezed)
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

    def __repr__(self):
        return f"TorsionTensor(dim={self.hidden_dim}, rank={self.rank}, alpha={self.alpha})"

    def validate_input(self, hidden_states: torch.Tensor):
        """Sentinel: Ensure input validity."""
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D input [Batch, Seq, Dim], got {hidden_states.shape}")
        if hidden_states.size(-1) != self.hidden_dim:
            raise ValueError(f"Dimension mismatch: expected {self.hidden_dim}, got {hidden_states.size(-1)}")
        if torch.isnan(hidden_states).any():
            raise ValueError("Input contains NaNs")

    @torch_compile
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply torsion field to the trajectory.
        h_{t+1} = h_t + T(h_t)

        Args:
            hidden_states: [Batch, Seq, Dim]
        Returns:
            twisted_states: [Batch, Seq, Dim]
        """
        self.validate_input(hidden_states)

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
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.beta = beta
        # Policy Head: Maps hidden states to action probabilities (Logits)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)

    def __repr__(self):
        return f"ActiveInferenceController(dim={self.hidden_dim}, vocab={self.vocab_size}, beta={self.beta})"

    def validate_inputs(self, hidden_states: torch.Tensor, target_probs: torch.Tensor):
        """Sentinel: Check inputs for optimization step."""
        if torch.isnan(hidden_states).any():
            raise ValueError("Hidden states contain NaNs")
        if torch.isnan(target_probs).any():
            raise ValueError("Target probs contain NaNs")
        if (target_probs < 0).any() or (target_probs > 1.0001).any():
            # Allow small float error
            logger.warning("Target probabilities out of [0,1] range (clamping)")
            target_probs.data.clamp_(0, 1)

        # Sentinel: Probability Sum Check
        prob_sum = target_probs.sum(dim=-1)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3):
            logger.warning("Target probabilities do not sum to 1. Normalizing...")
            target_probs.data.div_(prob_sum.unsqueeze(-1))

    @torch_compile
    def compute_free_energy(self,
                          hidden_states: torch.Tensor,
                          target_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculates Variational Free Energy F.

        Bolt Optimization:
        - Avoids .item() calls to prevent GPU synchronization.
        - Returns tensors in metrics.
        """
        # Sentinel: Ensure target_probs are valid (normalized)
        # We do this check here to ensure JIT compatibility (inline)
        prob_sum = target_probs.sum(dim=-1, keepdim=True)
        # Use simple epsilon check
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3):
             # In JIT, we might prefer functional operations over in-place if possible, but strictness matters
             target_probs = target_probs / prob_sum

        # Project to logits (Action Space)
        logits = self.policy_head(hidden_states)
        q_log_probs = F.log_softmax(logits, dim=-1)
        q_probs = torch.exp(q_log_probs)

        # 1. KL Divergence (Energy Term): Precision-weighted prediction error
        # D_KL = sum(Q * (logQ - logP))
        # Note: F.kl_div expects log_input as first arg
        # We treat target_probs as the Prior P
        # Verify target_probs shape
        if target_probs.shape != q_log_probs.shape:
             # Sentinel: Shape mismatch
             raise ValueError(f"Shape mismatch: logits {q_log_probs.shape} vs target {target_probs.shape}")

        kl_div = F.kl_div(q_log_probs, target_probs, reduction='batchmean')

        # 2. Entropy (Exploration Term)
        # H(Q) = -sum(Q * logQ)
        entropy = -torch.sum(q_probs * q_log_probs, dim=-1).mean()

        # 3. Free Energy F
        # F = Energy - Beta * Entropy
        # Minimizing F -> Minimize KL (Align) & Maximize Entropy (Explore)
        free_energy = kl_div - (self.beta * entropy)

        # Bolt Optimization: Return detached tensors instead of blocking scalar floats
        metrics = {
            "F": free_energy, # Returns tensor
            "KL": kl_div,     # Returns tensor
            "H": entropy      # Returns tensor
        }
        return free_energy, metrics

    def compute_optimization_step(self,
                                hidden_states: torch.Tensor,
                                target_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Bolt Optimization:
        Fused step that computes Free Energy AND Control Signal (Gradients)
        in a single forward pass, avoiding redundant computation.

        Returns:
            control_signal: u = -grad(F, h)
            free_energy: F
            metrics: dict of tensors
        """
        self.validate_inputs(hidden_states, target_probs)

        # Detach and require grad to compute local sensitivity
        # This is the point where we start the 'active inference' graph
        h_curr = hidden_states.detach().requires_grad_(True)

        # Single Forward Pass
        free_energy, metrics = self.compute_free_energy(h_curr, target_probs)

        # Compute Gradient Flow: dF/dh
        try:
            # create_graph=False because we don't need second derivatives of F w.r.t h
            grads = grad(free_energy, h_curr, create_graph=False)[0]
        except RuntimeError as e:
            logger.error(f"Gradient computation failed: {e}")
            grads = torch.zeros_like(h_curr)

        # Check for exploding gradients
        if torch.isnan(grads).any() or torch.isinf(grads).any():
             logger.warning("Control signal contains NaNs/Infs. Zeroing update.")
             grads = torch.zeros_like(h_curr)

        control_signal = -grads

        return control_signal, free_energy, metrics

    def compute_control_signal(self,
                             hidden_states: torch.Tensor,
                             target_probs: torch.Tensor) -> torch.Tensor:
        """
        Legacy wrapper for compute_optimization_step.
        Ideally use compute_optimization_step directly to save compute.
        """
        signal, _, _ = self.compute_optimization_step(hidden_states, target_probs)
        return signal


class LyapunovStability:
    """
    Verifies asymptotic stability of the control trajectory.
    Checks the condition dV/dt < 0 where V is the Lyapunov function (Free Energy).
    """

    def __init__(self, window_size: int = 10, threshold: float = 1e-4):
        self.history = []
        self.window_size = window_size
        self.threshold = threshold

    def __repr__(self):
        return f"LyapunovStability(window={self.window_size}, thresh={self.threshold})"

    def verify(self, energy: Union[float, torch.Tensor]) -> Tuple[bool, float]:
        """
        Checks stability.
        Returns: (is_stable, delta_V)
        """
        # Bolt Optimization: Handle Tensor input safely
        if isinstance(energy, torch.Tensor):
            val = energy.item()
        else:
            val = energy

        if math.isnan(val) or math.isinf(val):
             logger.critical("Lyapunov Energy is NaN/Inf!")
             return False, 0.0

        self.history.append(val)
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

        if not is_stable:
             logger.warning(f"Instability detected: dV={delta:.6f} > {self.threshold}")

        return is_stable, delta

    def reset(self):
        self.history = []
