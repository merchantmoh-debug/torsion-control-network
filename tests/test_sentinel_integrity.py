import torch
import pytest
from tcn.math.sheaf import CohomologyEngine, ConstraintViolationError

def test_verify_truth_nan_behavior():
    """
    Sentinel Test: Ensure NaNs result in a specific error.
    """
    engine = CohomologyEngine()

    # Create consistent inputs
    base = torch.randn(5, 5)
    proposals = {
        "A": base.clone(),
        "B": base.clone() # Consistent
    }

    # Inject NaN
    proposals["B"][0, 0] = float('nan')

    # Expectation: Should fail with specific corruption message.
    try:
        engine.verify_truth(proposals)
        pytest.fail("Should have raised an error for NaN input")
    except ConstraintViolationError as e:
        msg = str(e)
        print(f"\nCaught error: {msg}")
        # We want to distinguish between "Truth Collapse" (disagreement)
        # and "Input Corruption" (NaNs).
        # Current code raises "TRUTH COLLAPSE...".
        # We want "Corruption Detected".
        if "Corruption" not in msg and "NaN" not in msg:
             # Wait, "Max Divergence: nan" has "nan" in it.
             # But we want a clearer error.
             if "TRUTH COLLAPSE" in msg:
                 pytest.fail("Raised generic Truth Collapse instead of specific Corruption error.")
    except Exception as e:
        pytest.fail(f"Caught unexpected exception type: {type(e)}: {e}")

if __name__ == "__main__":
    test_verify_truth_nan_behavior()
