import torch
import pytest
from tcn.constraints import GeometricConstraints, StabilityContract, ConstraintViolationError

class TestGeometricConstraints:
    def test_verify_positive_definite_success(self):
        """Should pass for a valid PD matrix."""
        # Create a PD matrix: A = B @ B.T + epsilon * I
        B = torch.randn(3, 3)
        PD = B @ B.T + 1e-4 * torch.eye(3)
        GeometricConstraints.verify_positive_definite(PD)

    def test_verify_positive_definite_failure_symmetry(self):
        """Should fail if matrix is not symmetric."""
        non_sym = torch.randn(3, 3)
        # Ensure asymmetry
        non_sym[0, 1] = 1.0
        non_sym[1, 0] = 0.0

        with pytest.raises(ConstraintViolationError, match="not symmetric"):
            GeometricConstraints.verify_positive_definite(non_sym)

    def test_verify_positive_definite_failure_eigenvalues(self):
        """Should fail if matrix has negative eigenvalues."""
        # Diagonal matrix with a negative entry
        neg_eig = torch.diag(torch.tensor([1.0, -1.0, 1.0]))

        with pytest.raises(ConstraintViolationError, match="not positive definite"):
            GeometricConstraints.verify_positive_definite(neg_eig)

    def test_verify_skew_symmetric_success(self):
        """Should pass for a valid skew-symmetric matrix."""
        A = torch.randn(3, 3)
        skew = A - A.T
        GeometricConstraints.verify_skew_symmetric(skew)

    def test_verify_skew_symmetric_failure(self):
        """Should fail if matrix is not skew-symmetric."""
        sym = torch.randn(3, 3)
        sym = sym + sym.T  # Symmetric, not skew

        with pytest.raises(ConstraintViolationError, match="not skew-symmetric"):
            GeometricConstraints.verify_skew_symmetric(sym)

class TestStabilityContract:
    def test_verify_decay_success(self):
        """Should pass if energy decreases."""
        contract = StabilityContract(threshold=0.1)
        contract.verify_decay(energy_prev=1.0, energy_curr=0.9)

    def test_verify_decay_failure(self):
        """Should fail if energy increases beyond threshold."""
        contract = StabilityContract(threshold=0.1)

        with pytest.raises(ConstraintViolationError, match="Stability Contract Violated"):
            contract.verify_decay(energy_prev=1.0, energy_curr=1.2)
