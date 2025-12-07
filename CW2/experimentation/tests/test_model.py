import pytest
import pandas as pd
import numpy as np


def test_data_shape_logic():
    # Simulate a dummy dataframe similar to BGL
    data = {
        'IsAnomaly': [0, 1, 0, 0],
        'Feature1': [0.1, 0.2, 0.1, 0.5],
        'Feature2': [1.1, 1.2, 1.1, 1.5]
    }
    df = pd.DataFrame(data)
    
    # Test 1: Check target column exists
    assert 'IsAnomaly' in df.columns, "Target column missing"
    
    # Test 2: Check splitting logic (80/20 split)
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    assert train_size == 3
    assert test_size == 1

def test_requirements_file_exists():
    import os
    # Tests that the environment definition is present
    assert os.path.exists("requirements.txt") or os.path.exists("../requirements.txt")