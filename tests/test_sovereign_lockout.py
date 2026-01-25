import torch
import pytest
from tcn.sovereign import SovereignEntity, SovereignLockoutError

def test_death_before_lie_lockout():
    """
    Verifies that the Sovereign Entity strictly locks out (raises Error)
    when presented with inconsistent truth inputs (H^1 != 0).
    """

    # 1. Initialize Sovereign
    hidden_dim = 16
    vocab_size = 100
    entity = SovereignEntity(hidden_dim, vocab_size)

    # 2. Create Dummy State
    # [Batch=1, Seq=5, Dim=16]
    hidden_state = torch.randn(1, 5, hidden_dim)

    # Target Probs (Dummy)
    target_probs = torch.randn(1, 5, vocab_size).softmax(dim=-1)

    # 3. Scenario A: Consistent Inputs (Truth)
    # Both "agents" agree roughly
    proposal_a = hidden_state.clone()
    proposal_b = hidden_state.clone() + 0.001 # Micro noise

    valid_inputs = {
        "Agent_Alpha": proposal_a,
        "Agent_Beta": proposal_b
    }

    try:
        result = entity.generate_step(hidden_state, target_probs, external_proposals=valid_inputs)
        assert result is not None
        print("\n[PASS] Consistent inputs processed successfully.")
    except SovereignLockoutError:
        pytest.fail("System locked out on valid inputs!")

    # 4. Scenario B: Inconsistent Inputs (The Lie)
    # Agents disagree fundamentally
    proposal_lie = hidden_state.clone() + 10.0 # Massive deviation

    invalid_inputs = {
        "Agent_Alpha": proposal_a,
        "Agent_Omega": proposal_lie
    }

    with pytest.raises(SovereignLockoutError) as excinfo:
        entity.generate_step(hidden_state, target_probs, external_proposals=invalid_inputs)

    print(f"\n[PASS] Lockout Triggered: {str(excinfo.value)}")
    assert "Truth Topology Broken" in str(excinfo.value)
    assert "ZERO-CAPITULATION" in str(excinfo.value)

if __name__ == "__main__":
    test_death_before_lie_lockout()
