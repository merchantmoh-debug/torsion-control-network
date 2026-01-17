
import torch
import pytest
import math
from tcn.core import ActiveInferenceController
from tcn.vsm.system_5_policy.policy import SoundHeart, SovereignLockoutError

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def hidden_states(device):
    # [Batch, Seq, Dim]
    return torch.randn(2, 5, 10, device=device)

class TestBoltOptimizations:
    def test_compute_optimization_step(self, hidden_states):
        """Verify fused step returns correct values and control signal."""
        vocab_size = 5
        controller = ActiveInferenceController(hidden_dim=10, vocab_size=vocab_size, beta=0.1)
        target_probs = torch.softmax(torch.randn(2, 5, vocab_size), dim=-1)

        # Bolt Optimization Step
        control_signal, free_energy, metrics = controller.compute_optimization_step(hidden_states, target_probs)

        # 1. Verify Shapes
        assert control_signal.shape == hidden_states.shape
        assert free_energy.ndim == 0 # scalar

        # 2. Verify Values match separate computation
        # Note: We need to clone hidden_states because optimization_step might detach
        h_clone = hidden_states.clone().detach()

        # Original methods
        expected_signal = controller.compute_control_signal(h_clone, target_probs)
        expected_energy, expected_metrics = controller.compute_free_energy(h_clone, target_probs)

        # Check agreement
        assert torch.allclose(control_signal, expected_signal, atol=1e-5)
        assert torch.allclose(free_energy, expected_energy, atol=1e-5)
        assert metrics["F"] == expected_metrics["F"]

class TestSentinelHardening:
    def test_structural_integrity_check(self, hidden_states):
        """Verify SoundHeart rejects malformed states."""
        policy = SoundHeart()

        # 1. Valid State
        assert policy.enforce_prime_directive(hidden_states) == True

        # 2. NaN State
        nan_state = hidden_states.clone()
        nan_state[0, 0, 0] = float('nan')

        with pytest.raises(SovereignLockoutError) as excinfo:
            policy.enforce_prime_directive(nan_state)
        assert "NaNs or Infs" in str(excinfo.value)

        # 3. Inf State
        inf_state = hidden_states.clone()
        inf_state[0, 0, 0] = float('inf')

        with pytest.raises(SovereignLockoutError) as excinfo:
            policy.enforce_prime_directive(inf_state)
        assert "NaNs or Infs" in str(excinfo.value)

        # 4. Exploding Norm
        exploded_state = hidden_states.clone() * 10000.0

        with pytest.raises(SovereignLockoutError) as excinfo:
            policy.enforce_prime_directive(exploded_state)
        assert "Latent Norm Exploded" in str(excinfo.value)
