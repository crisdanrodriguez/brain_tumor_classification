#!/usr/bin/env python3
"""
Basic test suite for Brain Tumor Classification project.
Tests imports and basic functionality of all modules.
"""

import sys
import os
import pandas as pd
import numpy as np

def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        import features_extraction
        print("✓ features_extraction imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import features_extraction: {e}")
        return False

    try:
        import stacking_sklearn
        print("✓ stacking_sklearn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import stacking_sklearn: {e}")
        return False

    try:
        import stacking_from_scratch
        print("✓ stacking_from_scratch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import stacking_from_scratch: {e}")
        return False

    return True

def test_data_loading():
    """Test that the dataset can be loaded."""
    try:
        dataset_path = 'data/brain_tumor_dataset.csv'
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset file not found: {dataset_path}")
            return False

        df = pd.read_csv(dataset_path, index_col=0)
        print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

        # Check for required columns
        expected_cols = ['image_name', 'label_name']  # These should be present
        for col in expected_cols:
            if col not in df.columns:
                print(f"✗ Missing expected column: {col}")
                return False

        return True
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

def test_models_instantiation():
    """Test that models can be instantiated."""
    try:
        from stacking_sklearn import get_models
        models = get_models()
        print(f"✓ Models instantiated: {list(models.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate models: {e}")
        return False

def run_tests():
    """Run all tests and report results."""
    print("Running basic tests for Brain Tumor Classification project...\n")

    tests = [
        ("Import Tests", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Instantiation", test_models_instantiation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        if test_func():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())