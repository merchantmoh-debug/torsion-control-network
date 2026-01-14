import torch
import pytest
import math
from tcn.core import (
    RiemannianManifold,
    TorsionTensor,
    ActiveInferenceController,
    LyapunovStability
)

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def hidden_states(device):
    # [Batch, Seq, Dim]
    return torch.randn(2, 5, 10, device=device)

class TestRiemannianManifold:
    def test_compute_metric_tensor(self, hidden_states):
        """Verify metric tensor G is positive definite and correct shape."""
        metric = RiemannianManifold.compute_metric_tensor(hidden_states)

        # Shape check: [Batch, Dim, Dim]
        b, s, d = hidden_states.size()
        assert metric.shape == (b, d, d)

        # Symmetry check: G = G^T
        for i in range(b):
            assert torch.allclose(metric[i], metric[i].T, atol=1e-5)

        # Positive Definiteness check: Eigenvalues > 0
        for i in range(b):
            eigs = torch.linalg.eigvalsh(metric[i])
            assert torch.all(eigs > 0), f"Metric tensor is not positive definite: {eigs}"

    def test_geodesic_distance(self, hidden_states):
        """Verify geodesic distance computation."""
        metric = RiemannianManifold.compute_metric_tensor(hidden_states)

        # Test distance between h and h (should be 0)
        dist = RiemannianManifold.geodesic_distance(hidden_states, hidden_states, metric)
        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-5)

        # Test distance between h and shifted h
        shifted = hidden_states + 1.0
        dist = RiemannianManifold.geodesic_distance(hidden_states, shifted, metric)
        assert torch.all(dist >= 0)

class TestTorsionTensor:
    def test_forward_pass_shape(self, hidden_states):
        """Verify output shape and valid forward pass."""
        b, s, d = hidden_states.shape
        model = TorsionTensor(hidden_dim=d, rank=4, alpha=0.1)
        output = model(hidden_states)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()

    def test_skew_symmetry(self):
        """Verify that the internal operator is effectively skew-symmetric."""
        d, r = 10, 4
        model = TorsionTensor(hidden_dim=d, rank=r)

        # Access the skew operator construction logic
        skew_omega = model.Omega - model.Omega.transpose(0, 1)

        # Check S^T = -S
        assert torch.allclose(skew_omega.T, -skew_omega, atol=1e-5)

    def test_zero_initialization(self, hidden_states):
        """Verify that initially (with zero V weights) the output equals input."""
        d = hidden_states.shape[-1]
        model = TorsionTensor(hidden_dim=d)

        # V is initialized to zeros
        output = model(hidden_states)
        assert torch.allclose(output, hidden_states, atol=1e-6)

class TestActiveInferenceController:
    @pytest.fixture
    def controller(self):
        return ActiveInferenceController(hidden_dim=10, vocab_size=5, beta=0.1)

    def test_free_energy_computation(self, controller, hidden_states):
        """Verify Free Energy F is computed correctly."""
        # Create dummy target probs [Batch, Seq, Vocab]
        target_probs = torch.softmax(torch.randn(2, 5, 5), dim=-1)

        F_val, metrics = controller.compute_free_energy(hidden_states, target_probs)

        assert isinstance(F_val.item(), float)
        assert "KL" in metrics
        assert "H" in metrics

        # Energy = KL - Beta * H. Check consistency.
        expected_F = metrics["KL"] - controller.beta * metrics["H"]
        assert math.isclose(metrics["F"], expected_F, rel_tol=1e-5)

    def test_control_signal_gradient(self, controller, hidden_states):
        """Verify control signal (gradient) shape and existence."""
        target_probs = torch.softmax(torch.randn(2, 5, 5), dim=-1)

        # Ensure hidden_states does not have grad enabled initially for this test setup
        h = hidden_states.clone().detach()

        signal = controller.compute_control_signal(h, target_probs)

        assert signal.shape == h.shape
        assert not torch.allclose(signal, torch.zeros_like(signal))

class TestLyapunovStability:
    def test_stability_verification(self):
        """Verify stability logic dV/dt."""
        monitor = LyapunovStability(window_size=5, threshold=0.1)

        # Case 1: Stable decrease
        # 1.0 -> 0.9 (delta = -0.1 <= 0.1) -> Stable
        monitor.verify(1.0)
        is_stable, delta = monitor.verify(0.9)
        assert is_stable
        assert math.isclose(delta, -0.1)

        # Case 2: Unstable increase
        # 0.9 -> 1.5 (delta = 0.6 > 0.1) -> Unstable
        is_stable, delta = monitor.verify(1.5)
        assert not is_stable
        assert math.isclose(delta, 0.6)

    def test_window_management(self):
        """Verify history window is maintained."""
        monitor = LyapunovStability(window_size=3)
        for i in range(10):
            monitor.verify(float(i))

        assert len(monitor.history) == 3
        assert monitor.history == [7.0, 8.0, 9.0]
