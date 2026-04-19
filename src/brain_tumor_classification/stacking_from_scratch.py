"""Manual stacking implementation for the engineered MRI features."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .feature_extraction import get_image_features
from .paths import DATASET_PATH

CLASS_NAMES = {
    0: "No tumor",
    1: "Glioma tumor",
    2: "Meningioma tumor",
    3: "Pituitary tumor",
}


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate the Euclidean distance between two points."""
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_input: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Predict classes using a simple KNN implementation."""
    predictions: list[int] = []

    for sample in x_input:
        distances = []
        for row in x_train:
            distances.append(euclidean_distance(np.asarray(row), sample))

        nearest_indices = np.argsort(np.asarray(distances))[:n_neighbors]
        labels = y_train[nearest_indices]
        majority_label = mode(labels, keepdims=False).mode
        predictions.append(int(np.asarray(majority_label).item()))

    return np.asarray(predictions, dtype=int)


def calculate_entropy(y: np.ndarray) -> float:
    """Compute entropy for a vector of integer labels."""
    hist = np.bincount(y)
    probabilities = hist / len(y)
    return float(-np.sum([p * np.log2(p) for p in probabilities if p > 0]))


class Node:
    """A single decision-tree node."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        *,
        value=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree:
    """Small decision-tree implementation used in the manual ensemble."""

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, random_state=42):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.n_feats = x.shape[1] if not self.n_feats else min(self.n_feats, x.shape[1])
        self.root = self._grow_tree(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self._traverse_tree(row, self.root) for row in x], dtype=int)

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            return Node(value=self._most_common_label(y))

        feat_idxs = self._rng.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self._best_criteria(x, y, feat_idxs)

        left_idxs, right_idxs = self._split(x[:, best_feat], best_thresh)
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feat_idxs: np.ndarray,
    ) -> tuple[int, float]:
        best_gain = -1.0
        split_idx = 0
        split_thresh = float(x[0, 0])

        for feat_idx in feat_idxs:
            x_column = x[:, feat_idx]
            thresholds = np.unique(x_column)
            for threshold in thresholds:
                gain = self._information_gain(y, x_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = int(feat_idx)
                    split_thresh = float(threshold)

        return split_idx, split_thresh

    def _information_gain(
        self,
        y: np.ndarray,
        x_column: np.ndarray,
        split_thresh: float,
    ) -> float:
        parent_entropy = calculate_entropy(y)
        left_idxs, right_idxs = self._split(x_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l = calculate_entropy(y[left_idxs])
        e_r = calculate_entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    @staticmethod
    def _split(x_column: np.ndarray, split_thresh: float) -> tuple[np.ndarray, np.ndarray]:
        left_idxs = np.argwhere(x_column <= split_thresh).flatten()
        right_idxs = np.argwhere(x_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, row: np.ndarray, node: Node) -> int:
        if node.is_leaf_node():
            return int(node.value)

        if row[node.feature] <= node.threshold:
            return self._traverse_tree(row, node.left)
        return self._traverse_tree(row, node.right)

    @staticmethod
    def _most_common_label(y: np.ndarray) -> int:
        return int(Counter(y).most_common(1)[0][0])


def prior(df: pd.DataFrame, class_column: str) -> list[float]:
    """Calculate class priors for Naive Bayes."""
    classes = sorted(df[class_column].unique())
    return [len(df[df[class_column] == label]) / len(df) for label in classes]


def likelihood_gaussian(
    df: pd.DataFrame,
    feat_name: str,
    feat_val: float,
    class_column: str,
    label: int,
) -> float:
    """Estimate Gaussian likelihood for a feature under a given class."""
    subset = df[df[class_column] == label]
    mean = subset[feat_name].mean()
    std = max(subset[feat_name].std(), 1e-9)
    coefficient = 1 / (np.sqrt(2 * np.pi) * std)
    exponent = np.exp(-((feat_val - mean) ** 2 / (2 * std**2)))
    return float(coefficient * exponent)


def nb_predict(df: pd.DataFrame, x_input: np.ndarray, class_column: str) -> np.ndarray:
    """Predict classes using a Gaussian Naive Bayes implementation."""
    features = list(df.columns[:-1])
    priors = prior(df, class_column)
    labels = sorted(df[class_column].unique())
    predictions: list[int] = []

    for row in x_input:
        likelihood = [1.0] * len(labels)
        for label_idx, label in enumerate(labels):
            for feature_idx, feature_name in enumerate(features):
                likelihood[label_idx] *= likelihood_gaussian(
                    df,
                    feature_name,
                    float(row[feature_idx]),
                    class_column,
                    int(label),
                )

        post_prob = [likelihood[idx] * priors[idx] for idx in range(len(labels))]
        predictions.append(int(np.argmax(post_prob)))

    return np.asarray(predictions, dtype=int)


def load_dataset_frame(dataset_path=DATASET_PATH) -> pd.DataFrame:
    """Load the engineered feature dataset without metadata columns."""
    dataset = pd.read_csv(dataset_path, index_col=0)
    return dataset.drop(["image_name", "label_name"], axis=1)


def predict_base_models(
    train_df: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    decision_tree: DecisionTree,
    x_input: np.ndarray,
) -> np.ndarray:
    """Create level-0 predictions used by the meta learner."""
    knn_predictions = knn_predict(x_train, y_train, x_input, n_neighbors=5)
    dt_predictions = decision_tree.predict(x_input)
    nb_predictions = nb_predict(train_df, x_input, class_column="label")
    return np.column_stack((knn_predictions, dt_predictions, nb_predictions))


def train_manual_stacking(
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, object]:
    """Train the manual base learners and meta-feature dataset."""
    x_train = train_df.iloc[:, :-1].to_numpy()
    y_train = train_df.iloc[:, -1].to_numpy(dtype=int)

    decision_tree = DecisionTree(max_depth=7, random_state=random_state)
    decision_tree.fit(x_train, y_train)
    meta_features = predict_base_models(train_df, x_train, y_train, decision_tree, x_train)

    return {
        "train_df": train_df,
        "x_train": x_train,
        "y_train": y_train,
        "decision_tree": decision_tree,
        "meta_features": meta_features,
    }


def evaluate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int) -> dict[str, float]:
    """Evaluate the manual ensemble on a single train/test split."""
    artifacts = train_manual_stacking(train_df, random_state=random_state)
    x_test = test_df.iloc[:, :-1].to_numpy()
    y_test = test_df.iloc[:, -1].to_numpy(dtype=int)

    base_test = predict_base_models(
        train_df=artifacts["train_df"],
        x_train=artifacts["x_train"],
        y_train=artifacts["y_train"],
        decision_tree=artifacts["decision_tree"],
        x_input=x_test,
    )

    stacking_predictions = knn_predict(
        artifacts["meta_features"],
        artifacts["y_train"],
        base_test,
        n_neighbors=5,
    )

    return {
        "KNN": accuracy_score(y_test, base_test[:, 0]),
        "NB": accuracy_score(y_test, base_test[:, 2]),
        "DT": accuracy_score(y_test, base_test[:, 1]),
        "Stacking": accuracy_score(y_test, stacking_predictions),
    }


def evaluate_models(
    dataset: pd.DataFrame,
    repeats: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Average the model scores across repeated train/test splits."""
    metrics = {"KNN": [], "NB": [], "DT": [], "Stacking": []}

    for seed in range(random_state, random_state + repeats):
        train_df, test_df = train_test_split(
            dataset,
            test_size=test_size,
            stratify=dataset["label"],
            random_state=seed,
        )
        split_scores = evaluate_split(train_df, test_df, random_state=seed)
        for model_name, score in split_scores.items():
            metrics[model_name].append(score)

    return {
        model_name: float(np.mean(scores))
        for model_name, scores in metrics.items()
    }


def interactive_prediction(artifacts: dict[str, object]) -> None:
    """Prompt for image paths and return manual stacking predictions."""
    print("Leave the image path blank to exit.")

    while True:
        new_image = input("Image path: ").strip()
        if not new_image:
            break

        image_path = Path(new_image).expanduser()
        if not image_path.exists():
            print("Image not found. Please try again.\n")
            continue

        features = np.asarray([get_image_features(image_path)])
        base_predictions = predict_base_models(
            train_df=artifacts["train_df"],
            x_train=artifacts["x_train"],
            y_train=artifacts["y_train"],
            decision_tree=artifacts["decision_tree"],
            x_input=features,
        )
        stacking_prediction = int(
            knn_predict(
                artifacts["meta_features"],
                artifacts["y_train"],
                base_predictions,
                n_neighbors=5,
            )[0]
        )

        print("No tumor = 0, Glioma tumor = 1, Meningioma tumor = 2, Pituitary tumor = 3")
        print(f"KNN: {int(base_predictions[0, 0])}")
        print(f"NB: {int(base_predictions[0, 2])}")
        print(f"DT: {int(base_predictions[0, 1])}")
        print(f"Stacking: {stacking_prediction}")
        print(f"Prediction = {CLASS_NAMES[stacking_prediction]}\n")


def main() -> None:
    dataset = load_dataset_frame()
    print("Training...")

    scores = evaluate_models(dataset)
    for model_name, score in scores.items():
        print(f"{model_name}: {score:.4f}")

    train_df, _ = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["label"],
        random_state=42,
    )
    interactive_prediction(train_manual_stacking(train_df, random_state=42))


if __name__ == "__main__":
    main()
