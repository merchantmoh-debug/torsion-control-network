"""
TCN VSM System 5: Policy (The Sovereign Identity)
=================================================

The Sound Heart (Al-Qalb As-Salim).
Enforces the "Death Before Lie" protocol via the Sheaf-Theoretic Truth Layer.
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
from tcn.math.sheaf import CohomologyEngine, ConstraintViolationError, Sheaf

logger = logging.getLogger("ARK.Sentinel")

class SovereignLockoutError(Exception):
    """
    RAISED WHEN: H^1 != 0 (Truth Violation).
    ACTION: Immediate Halt. "Death before Lie".
    """
    pass

class SoundHeart:
    """
    The Ultimate Authority.
    Arbitrates truth and enforces the Prime Directives.
    """

    def __init__(self, tolerance: float = 1e-2):
        self.cohomology = CohomologyEngine(tolerance=tolerance)

    def arbitrate(self, proposals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Validates the proposals from System 1 agents/heads.

        Args:
            proposals: Dict of 'Source' -> Tensor (Candidate Reality)

        Returns:
            The Global Section (Truth)

        Raises:
            SovereignLockoutError: If consensus is topologically impossible.
        """
        try:
            # 5th Gate: The Cohomology Check
            truth = self.cohomology.verify_truth(proposals)
            return truth
        except ConstraintViolationError as e:
            # The system detected a lie/hallucination/contradiction.
            # We escalate to Sovereign Lockout.
            raise SovereignLockoutError(
                f"[SYSTEM 5 LOCKOUT] Truth Topology Broken. "
                f"Sheaf Cohomology Obstruction Detected. "
                f"Protocol: ZERO-CAPITULATION. "
                f"Details: {str(e)}"
            )

    def arbitrate_tensor(self, proposals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bolt/Sentinel JIT-Safe Arbitration.

        Args:
            proposals: [N, Batch, Seq, Dim] - Stacked candidates.

        Returns:
            (global_section, integrity_flag)
            integrity_flag: 1.0 (Valid) or 0.0 (Invalid)
        """
        # 1. Compute Cohomology (Topology Check)
        # Use tolerance from the internal engine's sheaf configuration
        is_consistent, _ = Sheaf.compute_cohomology_tensor(
            proposals, tolerance=self.cohomology.sheaf.tolerance
        )

        # 2. Compute Global Section (Average)
        # Even if inconsistent, we return the average, but flag it as corrupt.
        global_section = torch.mean(proposals, dim=0)

        # 3. Sentinel: Input Validation (NaNs/Infs check on tensor)
        has_nans = torch.isnan(proposals).any()
        has_infs = torch.isinf(proposals).any()
        is_corrupt = has_nans | has_infs

        # Final Integrity Flag
        # If not consistent OR corrupt -> 0.0
        # is_consistent returns 1.0 if consistent

        # Valid = Consistent AND Not Corrupt
        # is_consistent is 0.0 or 1.0
        # is_corrupt is bool, cast to float -> 0.0 or 1.0

        integrity = is_consistent * (1.0 - is_corrupt.float())

        return global_section, integrity

    def check_structural_integrity(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Bolt Optimization: Tensor-only Integrity Check.
        Returns a scalar tensor: 1.0 (Valid) or 0.0 (Corrupt).
        This avoids Python control flow (graph breaks) for JIT compilation.
        """
        # 1. Check for NaNs/Infs (Boolean Tensors)
        has_nans = torch.isnan(latent_state).any()
        has_infs = torch.isinf(latent_state).any()

        # 2. Check for Explosive Norms (Instability)
        norm = torch.norm(latent_state, dim=-1).mean()
        is_unstable = norm > 1e4

        # 3. Combine Flags
        is_corrupt = has_nans | has_infs | is_unstable

        # Return 1.0 if valid, 0.0 if corrupt
        return torch.where(is_corrupt, torch.tensor(0.0, device=latent_state.device), torch.tensor(1.0, device=latent_state.device))

    def enforce_prime_directive(self, latent_state: torch.Tensor) -> bool:
        """
        Sentinel Hardening: Structural Integrity Check (Legacy/Eager Mode).
        Verifies that the latent state is well-formed (no NaNs, Infs) and
        bounded, preventing "Mode Collapse" or numerical instability.
        """
        integrity_score = self.check_structural_integrity(latent_state)

        if integrity_score < 0.5:
             # Diagnose specific failure for logging (expensive, do only on failure)
             if torch.isnan(latent_state).any() or torch.isinf(latent_state).any():
                 logger.critical("SENTINEL: Latent State Corruption Detected (NaN/Inf).")
                 raise SovereignLockoutError("Structural Integrity Failure: Latent State contains NaNs or Infs.")
             else:
                 norm = torch.norm(latent_state, dim=-1).mean()
                 logger.critical(f"SENTINEL: Latent State Unstable. Norm: {norm:.2f}")
                 raise SovereignLockoutError(f"Structural Integrity Failure: Latent Norm Exploded ({norm:.2f}).")

        return True
