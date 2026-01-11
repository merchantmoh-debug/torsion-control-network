"""
Sovereign Entity (ARK v71)
==========================
The Unified Sovereign Organism.
Integrates Systems 1-5 into a coherent loop.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from src.tcn.vsm.system_1_ops.ops import KineticOperator
from src.tcn.vsm.system_2_coord.coord import Coordinator
from src.tcn.vsm.system_3_control.control import MAOSKernel
from src.tcn.vsm.system_4_intel.intel import FutureRadar
from src.tcn.vsm.system_5_policy.policy import SoundHeart, SovereignLockoutError

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

        # --- PHASE 1: INTELLIGENCE SCAN (System 4) ---
        # Scan for future collapse
        radar_scan = self.sys4_intel.scan_horizon(hidden_states)
        if radar_scan['status'] == 'COLLAPSE_IMMINENT':
            # In a full system, this would trigger a mode shift (e.g., to "Fortress Mode")
            pass

        # --- PHASE 2: TRUTH ARBITRATION (System 5) ---
        # If we have multiple inputs (or self-reflection), verify Cohomology
        if external_proposals:
            try:
                # This throws SovereignLockoutError if H^1 != 0
                _ = self.sys5_policy.arbitrate(external_proposals)
            except SovereignLockoutError:
                raise # Propagate up to stop generation

        # --- PHASE 3: OPTIMIZATION (System 3) ---
        # Calculate Free Energy gradients
        control_packet = self.sys3_control.optimization_step(hidden_states, target_probs)
        raw_correction = control_packet['control_signal']

        # --- PHASE 4: COORDINATION (System 2) ---
        # Dampen the signal
        smooth_correction = self.sys2_coord.dampen(raw_correction)

        # --- PHASE 5: KINETIC EXECUTION (System 1) ---
        # 1. Apply Torsion (Twist)
        twisted_state = self.sys1_ops.apply_torsion(hidden_states)

        # 2. Apply Control Gradient (Geodesic Steering)
        # x_new = x_twisted + correction
        # Bolt: In-place addition for memory efficiency
        final_state = twisted_state.add_(smooth_correction)

        return {
            "state": final_state,
            "metrics": {
                "radar": radar_scan,
                "control": control_packet['metrics'],
                "stable": control_packet['is_stable']
            }
        }
