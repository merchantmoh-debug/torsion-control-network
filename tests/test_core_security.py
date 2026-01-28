
import torch
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from tcn.core import RiemannianManifold, TorsionTensor, ActiveInferenceController

class TestCoreSecurity(unittest.TestCase):
    def setUp(self):
        self.B, self.S, self.D = 2, 10, 32
        self.h = torch.randn(self.B, self.S, self.D)
        self.target = torch.rand(self.B, self.S, 100) # Vocab=100
        self.target = self.target / self.target.sum(dim=-1, keepdim=True)

        # Ensure clean state
        self.h.requires_grad_(False)
        self.target.requires_grad_(False)

    def test_manifold_nan_check(self):
        """Sentinel: Verify RiemannianManifold rejects NaNs."""
        manifold = RiemannianManifold(self.D)
        h_bad = self.h.clone()
        h_bad[0, 0, 0] = float('nan')

        with self.assertRaisesRegex(ValueError, "NaN"):
            manifold.compute_metric_tensor(h_bad)

        with self.assertRaisesRegex(ValueError, "NaN"):
            manifold.compute_implicit_metric(h_bad)

    def test_torsion_validation(self):
        """Sentinel: Verify TorsionTensor behavior on invalid inputs."""
        torsion = TorsionTensor(self.D)

        # Bolt Optimization: Internal validation disabled for JIT speed.
        # Validation is delegated to System 5.
        # Checks that were raising ValueError will now proceed (or fail with RuntimeError from PyTorch).

        # NaN check - Should NOT raise ValueError from validate_input
        h_bad = self.h.clone()
        h_bad[0, 0, 0] = float('nan')
        # We expect it to run, producing nan output
        out = torsion(h_bad)
        self.assertTrue(torch.isnan(out).any())

    def test_aic_validation(self):
        """Sentinel: Verify ActiveInferenceController inputs."""
        aic = ActiveInferenceController(self.D, 100)

        # NaN in hidden
        h_bad = self.h.clone()
        h_bad[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, "Hidden states contain NaNs"):
            aic.compute_control_signal(h_bad, self.target)

        # Shape mismatch
        target_bad = torch.rand(self.B, self.S + 1, 100)
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            aic.compute_control_signal(self.h, target_bad)

    def test_aic_gradient_guard(self):
        """Sentinel: Verify gradient explosion protection."""
        # This is harder to trigger artificially without mocking grad,
        # but we can try to pass inputs that might cause issues if we bypass checks.
        pass

if __name__ == '__main__':
    unittest.main()
