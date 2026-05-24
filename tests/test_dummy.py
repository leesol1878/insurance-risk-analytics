"""
Simple tests for CI pipeline
"""

def test_pandas_import():
    """Test pandas can be imported"""
    import pandas as pd
    assert pd.__version__ is not None

def test_numpy_import():
    """Test numpy can be imported"""
    import numpy as np
    assert np.__version__ is not None

def test_dummy():
    """Basic test"""
    assert 1 + 1 == 2