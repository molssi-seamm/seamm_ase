"""
Unit and regression test for the seamm_ase package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import seamm_ase


def test_seamm_ase_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "seamm_ase" in sys.modules
