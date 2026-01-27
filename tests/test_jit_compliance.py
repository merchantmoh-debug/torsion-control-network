import torch
import pytest
from tcn.vsm.system_5_policy.policy import SoundHeart
from tcn.sovereign import SovereignEntity

def test_policy_check_structural_integrity_compiles():
    """
    Bolt Check: Verifies that check_structural_integrity compiles without graph breaks.
    """
    policy = SoundHeart()
    latent = torch.randn(1, 10, 128)

    # Wrap in torch.compile
    compiled_check = torch.compile(policy.check_structural_integrity)

    # Run once to compile
    res = compiled_check(latent)
    assert res.item() == 1.0

    # Run again to verify
    latent_corrupt = latent.clone()
    latent_corrupt[0, 0, 0] = float('nan')
    res_corrupt = compiled_check(latent_corrupt)
    assert res_corrupt.item() == 0.0

def test_sovereign_generate_step_compiles():
    """
    Bolt Check: Verifies that SovereignEntity.generate_step compiles.
    This is the ultimate test of the refactoring.
    """
    entity = SovereignEntity(hidden_dim=32, vocab_size=100)

    # Inputs
    hidden = torch.randn(1, 10, 32)
    target = torch.randn(1, 10, 100).softmax(-1)

    # Compile
    # Note: We must compile the method bound to the instance,
    # or rely on the @torch.compile decorator already present in source.
    # Since it IS present in source, we just call it.

    # Warmup
    res1 = entity.generate_step(hidden, target)
    assert res1['metrics']['integrity'].item() == 1.0

    # Corrupt Input Check
    hidden_bad = hidden.clone()
    hidden_bad[0,0,0] = float('inf')
    res2 = entity.generate_step(hidden_bad, target)
    assert res2['metrics']['integrity'].item() == 0.0

if __name__ == "__main__":
    test_policy_check_structural_integrity_compiles()
    test_sovereign_generate_step_compiles()
    print("JIT Compliance Tests Passed")
