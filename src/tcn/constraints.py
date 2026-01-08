"""
Torsion Control Network (TCN) Safety Kernel
===========================================
Explicit constraint enforcement module.
Acts as the 'Anvil' of stability, enforcing mathematical invariants
independent of the representation learning loop.

Components:
1. ConstraintViolationError: Explicit failure signal.
2. GeometricConstraints: Manifold topology enforcement.
3. StabilityContract: Lyapunov dynamics enforcement.

Author: The Architect
License: MIT
"""

import torch

class ConstraintViolationError(Exception):
    """
    Raised when a proposed state violates the mathematical invariants
    of the Torsion Control Network.

    This is a 'Truth-First' mechanism: The system refuses to process
    invalid states rather than hallucinating a result.
    """
    pass

class GeometricConstraints:
    """
    Enforces geometric topology constraints on the statistical manifold.
    """

    @staticmethod
    def verify_positive_definite(matrix: torch.Tensor, tol: float = 1e-5) -> None:
        """
        Verifies that the Metric Tensor G is positive definite.
        G must be symmetric and have strictly positive eigenvalues.

        Args:
            matrix: [..., Dim, Dim]
            tol: Numerical tolerance

        Raises:
            ConstraintViolationError if violated.
        """
        # 1. Symmetry Check
        if not torch.allclose(matrix, matrix.transpose(-1, -2), atol=tol):
            raise ConstraintViolationError("Geometric Invariant Failed: Metric tensor is not symmetric.")

        # 2. Positive Definiteness Check (Eigenvalues > 0)
        # Note: Cholesky is faster/more stable for checking PD than eigvalsh
        try:
            torch.linalg.cholesky(matrix)
        except RuntimeError:
            # Fallback to eigenvalue inspection for detailed error message if needed,
            # or just raise immediately.
            raise ConstraintViolationError("Geometric Invariant Failed: Metric tensor is not positive definite.")

    @staticmethod
    def verify_skew_symmetric(matrix: torch.Tensor, tol: float = 1e-5) -> None:
        """
        Verifies that the Torsion operator is skew-symmetric.
        A^T = -A
        """
        if not torch.allclose(matrix, -matrix.transpose(-1, -2), atol=tol):
             raise ConstraintViolationError("Geometric Invariant Failed: Torsion tensor is not skew-symmetric.")

class StabilityContract:
    """
    Enforces dynamic stability constraints based on Lyapunov theory.
    """

    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold

    def verify_decay(self, energy_prev: float, energy_curr: float) -> None:
        """
        Checks the Lyapunov condition: dV/dt <= 0 (within noise threshold).

        Args:
            energy_prev: Free Energy at t-1
            energy_curr: Free Energy at t

        Raises:
            ConstraintViolationError if energy spikes (instability).
        """
        delta = energy_curr - energy_prev

        # We allow small increases due to numerical noise or exploration (Qi),
        # but significant divergence triggers a halt.
        if delta > self.threshold:
            raise ConstraintViolationError(
                f"Stability Contract Violated: Free Energy divergence detected. "
                f"Delta: {delta:.6f} > Threshold: {self.threshold}"
            )
