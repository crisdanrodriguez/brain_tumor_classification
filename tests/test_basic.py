"""Basic smoke tests for the project package."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brain_tumor_classification.feature_extraction import get_image_features
from brain_tumor_classification.paths import DATASET_PATH, TESTING_DIR
from brain_tumor_classification.stacking_sklearn import get_models


class ProjectSmokeTests(unittest.TestCase):
    """Validate the repository's core assets and imports."""

    def test_dataset_can_be_loaded(self) -> None:
        self.assertTrue(DATASET_PATH.exists())
        dataset = pd.read_csv(DATASET_PATH, index_col=0)
        self.assertEqual(dataset.shape, (3206, 15))
        self.assertIn("image_name", dataset.columns)
        self.assertIn("label_name", dataset.columns)
        self.assertIn("label", dataset.columns)

    def test_models_can_be_instantiated(self) -> None:
        models = get_models()
        self.assertEqual(set(models.keys()), {"KNN", "DT", "NB", "Stacking"})

    def test_feature_extraction_returns_numeric_values(self) -> None:
        image_path = next((TESTING_DIR / "glioma_tumor").glob("*.jpg"))
        features = get_image_features(image_path)

        self.assertEqual(len(features), 12)
        for value in features:
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
