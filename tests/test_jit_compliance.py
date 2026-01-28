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

    # Proposals
    proposals = {
        "Head1": hidden.clone(),
        "Head2": hidden.clone() + 0.0001
    }

    # Explicitly compile with fullgraph=True to enforce NO graph breaks
    # This overrides the default @torch.compile on the method which might be partial
    opt_generate = torch.compile(entity.generate_step, fullgraph=True)

    # Warmup / Run
    # Note: We pass self explicitly if we compile the unbound method,
    # but torch.compile handles bound methods too if we access it from instance?
    # Actually, easiest is to just call the method if the decorator in source is sufficient.
    # But to test fullgraph=True specifically here, we wrap it.

    # We need to wrap a function that calls the method
    def forward_fn(h, t, p):
        return entity.generate_step(h, t, p)

    opt_fn = torch.compile(forward_fn, fullgraph=True)

    print("Running compiled function...")
    res1 = opt_fn(hidden, target, proposals)
    print("Metrics:", res1['metrics'])
    assert res1['metrics']['integrity'].item() == 1.0
    print("Success: Compiled with fullgraph=True")

    # Corrupt Input Check
    hidden_bad = hidden.clone()
    hidden_bad[0,0,0] = float('inf')
    res2 = opt_fn(hidden_bad, target, proposals)
    assert res2['metrics']['integrity'].item() == 0.0

if __name__ == "__main__":
    test_policy_check_structural_integrity_compiles()
    test_sovereign_generate_step_compiles()
    print("JIT Compliance Tests Passed")
