"""
TCN Error Definitions
=====================
Centralized exception classes for the Torsion Control Network.
"""

class SovereignLockoutError(Exception):
    """
    RAISED WHEN: H^1 != 0 (Truth Violation) or Structural Integrity Failure.
    ACTION: Immediate Halt. "Death before Lie".
    Protocol: ZERO-CAPITULATION.
    """
    pass
