import torch
import pytest
from tcn.core.legacy_core import ActiveInferenceController

def test_aic_optimization_logic():
    hidden_dim = 16
    vocab_size = 10
    aic = ActiveInferenceController(hidden_dim, vocab_size)

    h = torch.randn(2, 5, hidden_dim)
    t = torch.softmax(torch.randn(2, 5, vocab_size), dim=-1)

    # 1. Test basic computation
    fe, metrics = aic.compute_free_energy(h, t)
    assert isinstance(fe, torch.Tensor)
    assert "F" in metrics

    # 2. Test optimization step (calls compute_free_energy internally)
    control, fe_step, metrics_step = aic.compute_optimization_step(h, t)

    assert control.shape == h.shape
    assert torch.allclose(fe, fe_step)

    # 3. Test Implicit Flag
    # This should run without error and produce same result for valid input
    fe2, metrics2 = aic.compute_free_energy(h, t, validate=False)
    assert torch.allclose(fe, fe2)

    # And it should be slightly faster (though hard to measure in unit test)
