"""Feature extraction utilities for the MRI image dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from skimage import io
from skimage.feature import graycomatrix as greycomatrix
from skimage.feature import graycoprops as greycoprops
from skimage.transform import resize

from .paths import DATASET_PATH, TRAINING_DIR

CLASS_LABELS = {
    "no_tumor": 0,
    "glioma_tumor": 1,
    "meningioma_tumor": 2,
    "pituitary_tumor": 3,
}

FEATURE_COLUMNS = (
    "image_name",
    "mean",
    "variance",
    "std",
    "skewness",
    "kurtosis",
    "entropy",
    "contrast",
    "dissimilarity",
    "homogeneity",
    "asm",
    "energy",
    "correlation",
    "label_name",
    "label",
)


def get_image_features(image_path: str | Path) -> tuple[float, ...]:
    """Return first-order and texture features for a single MRI image."""
    image = io.imread(image_path, as_gray=True) * 255
    image = resize(image, (512, 512), anti_aliasing=True)
    image = image.astype(np.uint8)

    mean = float(np.mean(image))
    variance = float(np.var(image))
    std = float(np.std(image))

    image_1d = image.flatten()
    skewness = float(skew(image_1d))
    kurtos = float(kurtosis(image_1d))
    entro = float(entropy(image_1d))

    glcm = greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    contrast = float(greycoprops(glcm, "contrast")[0, 0])
    dissimilarity = float(greycoprops(glcm, "dissimilarity")[0, 0])
    homogeneity = float(greycoprops(glcm, "homogeneity")[0, 0])
    asm = float(greycoprops(glcm, "ASM")[0, 0])
    energy = float(greycoprops(glcm, "energy")[0, 0])
    correlation = float(greycoprops(glcm, "correlation")[0, 0])

    return (
        mean,
        variance,
        std,
        skewness,
        kurtos,
        entro,
        contrast,
        dissimilarity,
        homogeneity,
        asm,
        energy,
        correlation,
    )


def label_class(label_name: str) -> int:
    """Map a class name to its numeric label."""
    return CLASS_LABELS[label_name]


def build_feature_dataset(images_dir: str | Path = TRAINING_DIR) -> pd.DataFrame:
    """Build a feature table from the training image directories."""
    rows: list[tuple[object, ...]] = []
    images_root = Path(images_dir)

    for class_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            features = get_image_features(image_path)
            rows.append(
                (
                    image_path.name,
                    *features,
                    class_dir.name,
                    label_class(class_dir.name),
                )
            )

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def save_feature_dataset(
    output_path: str | Path = DATASET_PATH,
    images_dir: str | Path = TRAINING_DIR,
) -> Path:
    """Generate and store the extracted feature dataset as CSV."""
    dataset = build_feature_dataset(images_dir=images_dir)
    output = Path(output_path)
    dataset.to_csv(output)
    return output


def main() -> None:
    output_path = save_feature_dataset()
    print(f"Feature dataset saved to {output_path}")


if __name__ == "__main__":
    main()
