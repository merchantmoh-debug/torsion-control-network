"""
Sovereign Entity (ARK v71)
==========================
The Unified Sovereign Organism.
Integrates Systems 1-5 into a coherent loop.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

# Bolt Optimization: Enable Tensor Cores for War Speed Matrix Multiplication
torch.set_float32_matmul_precision('high')

from tcn.vsm.system_1_ops.ops import KineticOperator
from tcn.vsm.system_2_coord.coord import Coordinator
from tcn.vsm.system_3_control.control import MAOSKernel
from tcn.vsm.system_4_intel.intel import FutureRadar
from tcn.vsm.system_5_policy.policy import SoundHeart, SovereignLockoutError

class SovereignEntity(nn.Module):
    """
    The Full TCN Sovereign Organism.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()

        # Initialize VSM Stack
        self.sys1_ops = KineticOperator(hidden_dim)
        self.sys2_coord = Coordinator()
        self.sys3_control = MAOSKernel(hidden_dim, vocab_size)
        self.sys4_intel = FutureRadar()
        self.sys5_policy = SoundHeart()

    @torch.compile # Bolt: Optimize the main feedback loop
    def generate_step(self,
                     hidden_states: torch.Tensor,
                     target_probs: torch.Tensor,
                     external_proposals: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """
        Executes one step of the Sovereign Loop.

        Args:
            hidden_states: [Batch, Seq, Dim] - Current mental state
            target_probs: [Batch, Seq, Vocab] - Desired/Prior distribution
            external_proposals: Optional multi-agent inputs for Truth Verification

        Returns:
            Updated hidden state and system metrics.
        """

        # Sentinel: Structural Integrity Check (Bolt Optimized)
        # We calculate integrity scores as tensors to maintain JIT graph compilation.
        integrity_in = self.sys5_policy.check_structural_integrity(hidden_states)

        integrity_target = torch.tensor(1.0, device=hidden_states.device)
        if target_probs is not None:
            # Simple NaN check for target_probs
            target_valid = ~torch.isnan(target_probs).any()
            integrity_target = torch.where(target_valid,
                                         torch.tensor(1.0, device=hidden_states.device),
                                         torch.tensor(0.0, device=hidden_states.device))

        # --- PHASE 1: INTELLIGENCE SCAN (System 4) ---
        # Scan for future collapse
        radar_scan = self.sys4_intel.scan_horizon(hidden_states)
        # radar_scan['status'] is now a tensor (0, 1, 2)
        # We allow flow to continue; mode logic is handled by tensor flags or outside

        # --- PHASE 2: TRUTH ARBITRATION (System 5) ---
        # If we have multiple inputs (or self-reflection), verify Cohomology
        integrity_truth = torch.tensor(1.0, device=hidden_states.device)
        integrity_error = torch.tensor(0.0, device=hidden_states.device)

        if external_proposals is not None and len(external_proposals) > 0:
            # Stack proposals
            # Bolt: Iterating dict keys is consistent in JIT if keys don't change
            proposals_list = list(external_proposals.values())
            stacked = torch.stack(proposals_list)

            # Arbitrate Tensor (No Exceptions)
            _, integrity_truth, integrity_error = self.sys5_policy.arbitrate_tensor(stacked)

        # --- PHASE 3: OPTIMIZATION (System 3) ---
        # Calculate Free Energy gradients
        control_packet = self.sys3_control.optimization_step(hidden_states, target_probs)
        raw_correction = control_packet['control_signal']

        # Calculate Norm for dashboard telemetry (Bolt/Sentinel requirement)
        # ARK Optimization: Return tensor to avoid graph break. Caller must sync if needed.
        control_norm = torch.norm(raw_correction)

        # --- PHASE 4: COORDINATION (System 2) ---
        # Dampen the signal
        smooth_correction = self.sys2_coord.dampen(raw_correction)

        # --- PHASE 5: KINETIC EXECUTION (System 1) ---
        # 1. Apply Torsion (Twist)
        twisted_state = self.sys1_ops.apply_torsion(hidden_states)

        # 2. Apply Control Gradient (Geodesic Steering)
        # x_new = x_twisted + correction
        final_state = twisted_state + smooth_correction

        # Sentinel: Final Integrity Verification
        integrity_out = self.sys5_policy.check_structural_integrity(final_state)

        # Aggregate Integrity (Logical AND via min or product)
        # If any score is 0.0, the result is 0.0
        system_integrity = torch.min(torch.stack([integrity_in, integrity_target, integrity_truth, integrity_out]))

        return {
            "state": final_state,
            "metrics": {
                "radar": radar_scan,
                "control": control_packet['metrics'],
                "control_norm": control_norm,
                "stable": control_packet['is_stable'],
                "integrity": system_integrity, # Bolt: Passed to outer loop for handling
                "truth_divergence": integrity_error # ARK: Palette Telemetry
            }
        }
