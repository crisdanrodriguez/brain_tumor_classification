"""Stacking implementation based on scikit-learn estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .paths import DATASET_PATH


def load_dataset(dataset_path=DATASET_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load the engineered feature dataset used for model training."""
    dataset = pd.read_csv(dataset_path, index_col=0)
    dataset = dataset.drop(["image_name", "label_name"], axis=1)

    x = dataset.iloc[:, :-1].to_numpy()
    y = dataset.iloc[:, -1].to_numpy()
    return x, y


def get_stacking(random_state: int = 42) -> StackingClassifier:
    """Create the stacking ensemble used in the original comparison."""
    level0 = [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("DT", DecisionTreeClassifier(max_depth=7, random_state=random_state)),
        ("NB", GaussianNB()),
    ]
    level1 = KNeighborsClassifier(n_neighbors=5)
    return StackingClassifier(estimators=level0, final_estimator=level1, cv=4)


def get_models(random_state: int = 42) -> dict[str, object]:
    """Return the baseline models plus the stacking ensemble."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DT": DecisionTreeClassifier(max_depth=7, random_state=random_state),
        "NB": GaussianNB(),
        "Stacking": get_stacking(random_state=random_state),
    }


def evaluate_model(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Evaluate a model using repeated stratified cross-validation."""
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=4,
        random_state=random_state,
    )
    return cross_val_score(
        model,
        x,
        y,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )


def main() -> None:
    x, y = load_dataset()

    for name, model in get_models().items():
        scores = evaluate_model(model, x, y)
        print(f">{name} {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")


if __name__ == "__main__":
    main()
