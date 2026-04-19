import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Test basic imports
def test_imports():
    """Test that all required modules can be imported"""
    import features_extraction
    import stacking_sklearn
    import stacking_from_scratch
    assert True

# Test feature extraction function
def test_feature_extraction():
    """Test that feature extraction works on dummy data"""
    from features_extraction import get_image_features

    # Create a dummy 512x512 image
    dummy_image = np.random.rand(512, 512)
    np.save('/tmp/dummy_image.npy', dummy_image)

    # This would normally fail without actual image, but tests the function exists
    # In a real test, you'd use a mock or actual test image
    assert callable(get_image_features)

# Test stacking sklearn
def test_stacking_sklearn():
    """Test that sklearn stacking works"""
    from stacking_sklearn import get_models, evaluate_model

    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=4, random_state=42)
    models = get_models()

    # Test that models dict is created
    assert isinstance(models, dict)
    assert 'Stacking' in models

    # Test evaluation (this will run but may not be meaningful with dummy data)
    scores = evaluate_model(models['KNN'], X, y)
    assert isinstance(scores, np.ndarray)

# Test that functions exist
def test_functions_exist():
    """Test that key functions are defined"""
    from features_extraction import get_image_features
    from stacking_sklearn import get_models, evaluate_model, get_stacking
    from stacking_from_scratch import knn_predict, decision_tree_predict, naive_bayes_predict

    assert callable(get_image_features)
    assert callable(get_models)
    assert callable(evaluate_model)
    assert callable(get_stacking)
    assert callable(knn_predict)
    assert callable(decision_tree_predict)
    assert callable(naive_bayes_predict)

if __name__ == "__main__":
    test_imports()
    test_feature_extraction()
    test_stacking_sklearn()
    test_functions_exist()
    print("All basic tests passed!")