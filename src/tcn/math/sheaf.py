"""
Sheaf Theory for Truth Consistency
==================================

Implements Sheaf Cohomology to detect structural contradictions (lies/hallucinations)
in the data stream.

If H^1(X, F) != 0, the data is inconsistent.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Bolt Optimization: JIT Compilation
try:
    from torch import compile as torch_compile
except ImportError:
    def torch_compile(func):
        return func

class ConstraintViolationError(Exception):
    pass

class Section:
    """
    A local section s over an open set U.
    Represents a specific observation or claim.
    """
    def __init__(self, data: torch.Tensor, region_id: str, confidence: float = 1.0):
        self.data = data
        self.region_id = region_id # The ID of the covering set (e.g. "Agent_A", "Head_3")
        self.confidence = confidence

class Sheaf:
    """
    Manages the collection of local sections and computes cohomology.
    """
    def __init__(self, tolerance: float = 1e-3):
        self.sections: Dict[str, Section] = {}
        self.overlaps: Dict[Tuple[str, str], float] = {} # Restriction map consistency metric
        self.tolerance = tolerance

    def add_section(self, section: Section):
        self.sections[section.region_id] = section

    def compute_restriction_loss(self, s_i: Section, s_j: Section) -> float:
        """
        Computes the distance between two sections on their overlap.
        Legacy method kept for single-pair debug.
        """
        if s_i.data.shape != s_j.data.shape:
            return float('inf')
        dist = torch.norm(s_i.data - s_j.data).item()
        return dist

    @torch_compile
    def compute_cohomology(self) -> Tuple[bool, float]:
        """
        Computes the Cech Cohomology check.
        Returns (is_consistent, max_inconsistency).

        Bolt Optimization:
        - Vectorized pairwise distance calculation using torch.cdist or pdist.
        - Avoids O(N^2) Python loops.
        """
        if len(self.sections) < 2:
            return True, 0.0 # Trivial consistency

        # 1. Stack all section data
        # Check shapes first
        sections_list = list(self.sections.values())
        first_shape = sections_list[0].data.shape

        # Sentinel: Dimension Mismatch Check
        for s in sections_list:
            if s.data.shape != first_shape:
                # Topologically invalid - sections must map to same fiber
                return False, float('inf')

        # Stack: [N, ...]
        try:
            stacked = torch.stack([s.data for s in sections_list])
        except RuntimeError:
            return False, float('inf')

        # Flatten for distance computation: [N, D_flat]
        N = stacked.size(0)
        flat = stacked.view(N, -1)

        # 2. Compute Pairwise Distances [N, N]
        # p=2 (Euclidean Norm) matches torch.norm(a-b)
        # cdist is efficient and JIT-friendly
        dists = torch.cdist(flat, flat, p=2)

        # 3. Find Max Divergence
        # The matrix is symmetric with 0 diagonal.
        max_inconsistency = dists.max().item()

        is_consistent = max_inconsistency <= self.tolerance
        return is_consistent, max_inconsistency

    @staticmethod
    def compute_cohomology_tensor(stacked_proposals: torch.Tensor, tolerance: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Cohomology on a pre-stacked tensor [N, ...].
        Returns (is_consistent (0.0/1.0), max_inconsistency).
        JIT-friendly: No object creation, no loops.
        """
        if stacked_proposals.size(0) < 2:
             return torch.tensor(1.0, device=stacked_proposals.device), torch.tensor(0.0, device=stacked_proposals.device)

        # Sentinel: Defense in Depth (Check for Corruption)
        # We perform this check in tensor-space to avoid graph breaks
        has_nans = torch.isnan(stacked_proposals).any()
        has_infs = torch.isinf(stacked_proposals).any()
        is_corrupt = has_nans | has_infs

        # Flatten for distance computation: [N, D_flat]
        N = stacked_proposals.size(0)
        flat = stacked_proposals.view(N, -1)

        # Pairwise Distances [N, N]
        dists = torch.cdist(flat, flat, p=2)

        # Max Divergence
        max_inconsistency = dists.max()

        # Consistency Check
        # Valid = (H^1 == 0) AND (No Corruption)
        is_consistent = torch.where(
            (max_inconsistency <= tolerance) & (~is_corrupt),
            torch.tensor(1.0, device=stacked_proposals.device),
            torch.tensor(0.0, device=stacked_proposals.device)
        )

        return is_consistent, max_inconsistency

class CohomologyEngine:
    """
    High-level interface for Truth Fusion.
    """
    def __init__(self, tolerance: float = 1e-2):
        self.sheaf = Sheaf(tolerance=tolerance)

    def verify_truth(self, proposals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inputs:
            proposals: Dict mapping 'AgentID' -> Tensor (Logits/State)

        Returns:
            The global section (consensus) if H^1 == 0.
            Raises ConstraintViolationError if H^1 != 0.
        """
        # Sentinel: Input Validation
        for agent_id, data in proposals.items():
            if torch.isnan(data).any():
                raise ConstraintViolationError(f"Corruption Detected: NaNs in signal from {agent_id}")
            if torch.isinf(data).any():
                raise ConstraintViolationError(f"Corruption Detected: Infs in signal from {agent_id}")

        # 1. Build Sheaf
        self.sheaf = Sheaf(tolerance=self.sheaf.tolerance) # Reset
        for agent_id, data in proposals.items():
            self.sheaf.add_section(Section(data, agent_id))

        # 2. Compute Cohomology
        is_consistent, error = self.sheaf.compute_cohomology()

        if not is_consistent:
            raise ConstraintViolationError(
                f"TRUTH COLLAPSE: H^1 != 0. Obstruction detected. Max Divergence: {error:.4f}"
            )

        # 3. Glue Global Section (Simple Average if consistent)
        stacked = torch.stack([s.data for s in self.sheaf.sections.values()])
        global_section = torch.mean(stacked, dim=0)

        return global_section
