import pytest
import torch
from tcn.core.legacy_core import ActiveInferenceController

def test_mpemba_annealing_initialization():
    """Verify Mpemba mode initializes correctly."""
    controller = ActiveInferenceController(
        hidden_dim=32,
        vocab_size=100,
        beta=0.1,
        mpemba_mode=True,
        initial_beta=1.0,
        mpemba_decay=0.5
    )

    assert controller.mpemba_mode is True
    assert controller.initial_beta == 1.0
    assert controller.step_counter == 0

    # Check initial beta
    current_beta = controller.get_current_beta()
    assert torch.isclose(current_beta, torch.tensor(1.0))

def test_mpemba_annealing_step():
    """Verify beta decays over steps."""
    controller = ActiveInferenceController(
        hidden_dim=32,
        vocab_size=100,
        beta=0.1,
        mpemba_mode=True,
        initial_beta=1.0,
        mpemba_decay=0.5
    )

    # Create dummy inputs
    hidden = torch.randn(1, 10, 32)
    target = torch.softmax(torch.randn(1, 10, 100), dim=-1)

    # Step 0
    assert torch.isclose(controller.get_current_beta(), torch.tensor(1.0))

    # Run optimization step
    controller.compute_optimization_step(hidden, target)

    # Step 1: Beta should be 1.0 * 0.5 = 0.5
    assert controller.step_counter == 1
    assert torch.isclose(controller.get_current_beta(), torch.tensor(0.5))

    # Step 2: Beta should be 0.5 * 0.5 = 0.25
    controller.compute_optimization_step(hidden, target)
    assert controller.step_counter == 2
    assert torch.isclose(controller.get_current_beta(), torch.tensor(0.25))

def test_mpemba_floor():
    """Verify beta does not go below target beta."""
    controller = ActiveInferenceController(
        hidden_dim=32,
        vocab_size=100,
        beta=0.2, # Target floor
        mpemba_mode=True,
        initial_beta=1.0,
        mpemba_decay=0.1 # Fast decay
    )

    hidden = torch.randn(1, 10, 32)
    target = torch.softmax(torch.randn(1, 10, 100), dim=-1)

    # Step 0: 1.0
    assert torch.isclose(controller.get_current_beta(), torch.tensor(1.0))

    # Step 1: 0.1 (but floor is 0.2) -> should be 0.2
    controller.compute_optimization_step(hidden, target)

    current_beta = controller.get_current_beta()
    assert torch.isclose(current_beta, torch.tensor(0.2))

def test_mpemba_disabled():
    """Verify standard mode ignores annealing."""
    controller = ActiveInferenceController(
        hidden_dim=32,
        vocab_size=100,
        beta=0.1,
        mpemba_mode=False
    )

    hidden = torch.randn(1, 10, 32)
    target = torch.softmax(torch.randn(1, 10, 100), dim=-1)

    assert torch.isclose(controller.get_current_beta(), torch.tensor(0.1))

    controller.compute_optimization_step(hidden, target)

    # Counter shouldn't increment or matter
    assert controller.step_counter == 0
    assert torch.isclose(controller.get_current_beta(), torch.tensor(0.1))
