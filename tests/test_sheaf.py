
import torch
import pytest
import time
from src.tcn.math.sheaf import CohomologyEngine, Sheaf, Section, ConstraintViolationError

def test_sheaf_consistency():
    """Test that sheaf correctly identifies consistent sections."""
    engine = CohomologyEngine(tolerance=0.3)

    # Create consistent data
    base = torch.randn(10, 10)
    data1 = base + 0.01
    data2 = base - 0.01

    proposals = {
        "A": data1,
        "B": data2
    }

    # Should succeed
    truth = engine.verify_truth(proposals)
    assert truth is not None
    assert torch.allclose(truth, base, atol=0.1)

def test_sheaf_inconsistency():
    """Test that sheaf correctly identifies inconsistent sections (H^1 != 0)."""
    engine = CohomologyEngine(tolerance=0.1)

    # Create inconsistent data
    base = torch.randn(10, 10)
    data1 = base
    data2 = base + 1.0 # Large deviation

    proposals = {
        "A": data1,
        "B": data2
    }

    # Should fail
    with pytest.raises(ConstraintViolationError):
        engine.verify_truth(proposals)

def test_sheaf_performance():
    """Benchmark sheaf cohomology computation."""
    engine = CohomologyEngine(tolerance=0.1)

    # Large scale test
    N = 50 # Number of agents
    D = 1000 # Dimension

    base = torch.randn(D)
    proposals = {}
    for i in range(N):
        proposals[f"Agent_{i}"] = base + (torch.randn(D) * 0.001)

    start_time = time.time()
    engine.verify_truth(proposals)
    end_time = time.time()

    print(f"\nTime for {N} agents, dim {D}: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    test_sheaf_consistency()
    test_sheaf_inconsistency()
    test_sheaf_performance()
