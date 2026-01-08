"""
TCN VSM System 2: Coordination (The Dampener)
=============================================

Prevents oscillation between System 1 (Action) and System 5 (Truth).
Ensures smooth geodesic flow.
"""

import torch

class Coordinator:
    """
    Dampens the control signal to prevent 'jitter' or 'overshoot'.
    """

    def __init__(self, damping_factor: float = 0.5):
        self.damping_factor = damping_factor
        self.previous_correction = None

    def dampen(self, raw_correction: torch.Tensor) -> torch.Tensor:
        """
        Applies exponential smoothing to the correction vector.
        """
        if self.previous_correction is None:
            self.previous_correction = raw_correction
            return raw_correction

        # Smooth: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
        smoothed = (self.damping_factor * raw_correction) + \
                   ((1 - self.damping_factor) * self.previous_correction)

        self.previous_correction = smoothed.detach() # Detach to stop gradient history accumulation
        return smoothed
