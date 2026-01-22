"""
TCN VSM System 5: Policy (The Sovereign Identity)
=================================================

The Sound Heart (Al-Qalb As-Salim).
Enforces the "Death Before Lie" protocol via the Sheaf-Theoretic Truth Layer.
"""

import torch
import logging
from typing import Dict, Any, Optional
from tcn.math.sheaf import CohomologyEngine, ConstraintViolationError

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

    def enforce_prime_directive(self, latent_state: torch.Tensor) -> bool:
        """
        Sentinel Hardening: Structural Integrity Check.
        Verifies that the latent state is well-formed (no NaNs, Infs) and
        bounded, preventing "Mode Collapse" or numerical instability.
        """
        # 1. Check for NaNs/Infs
        if torch.isnan(latent_state).any() or torch.isinf(latent_state).any():
             logger.critical("SENTINEL: Latent State Corruption Detected (NaN/Inf).")
             raise SovereignLockoutError("Structural Integrity Failure: Latent State contains NaNs or Infs.")

        # 2. Check for Explosive Norms (Instability)
        norm = torch.norm(latent_state, dim=-1).mean()
        if norm > 1e4: # Arbitrary high threshold for stability
             logger.critical(f"SENTINEL: Latent State Unstable. Norm: {norm:.2f}")
             raise SovereignLockoutError(f"Structural Integrity Failure: Latent Norm Exploded ({norm:.2f}).")

        return True
