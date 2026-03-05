"""
Data loading and preprocessing utilities.

This module loads MNIST / Fashion-MNIST using Keras datasets
"""


import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

try:
    from keras.datasets import mnist, fashion_mnist
except Exception:  
    from tensorflow.keras.datasets import mnist, fashion_mnist


SUPPORTED_DATASETS = ("mnist", "fashion_mnist")


def one_hot(y, num_classes=10):
    """Convert integer class labels to one-hot vectors."""
    y = np.asarray(y, dtype=np.int64)
    return np.eye(num_classes, dtype=np.float64)[y]


def _load_raw_dataset(name):
    """Load raw train/test splits from Keras datasets."""
    if name == "mnist":
        return mnist.load_data()
    if name == "fashion_mnist":
        return fashion_mnist.load_data()
    raise ValueError(f"Dataset must be one of {SUPPORTED_DATASETS}, got '{name}'")


def load_dataset(name="mnist", val_ratio=0.1, seed=42):
    """
    Load dataset and return normalized flattened arrays with one-hot labels.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    (X_train_full, y_train_full), (X_test, y_test) = _load_raw_dataset(name)

    X_train_full = X_train_full.reshape(-1, 28 * 28).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float64) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_ratio,
        stratify=y_train_full,
        random_state=seed,
    )

    Y_train = one_hot(y_train)
    Y_val = one_hot(y_val)
    Y_test = one_hot(y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def show_samples(X, Y, num_samples=10, seed=None):
    """Display random sample images and labels from a flattened dataset."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(X.shape[0], num_samples, replace=False)
    else:
        indices = np.random.choice(X.shape[0], num_samples, replace=False)

    plt.figure(figsize=(num_samples, 2))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")
        plt.title(str(np.argmax(Y[idx])))
        plt.axis("off")
    plt.tight_layout()
    plt.show()
