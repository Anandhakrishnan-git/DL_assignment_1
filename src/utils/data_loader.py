"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import os
import struct
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------------------------------------
# Low-level IDX readers
# --------------------------------------------------

def load_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)

    return data.astype(np.float64) / 255.0


def load_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype(np.int64)


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def one_hot(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    return np.eye(num_classes, dtype=np.float64)[y]


def get_dataset_paths(name):
    base_path = os.path.join("data", name)

    paths = {
        "train_images": os.path.join(base_path, "train-images.idx3-ubyte"),
        "train_labels": os.path.join(base_path, "train-labels.idx1-ubyte"),
        "test_images":  os.path.join(base_path, "t10k-images.idx3-ubyte"),
        "test_labels":  os.path.join(base_path, "t10k-labels.idx1-ubyte"),
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key} not found at {path}")

    return paths


# --------------------------------------------------
# Main loader (train + val + test)
# --------------------------------------------------

def load_dataset(name="mnist", val_ratio=0.1, seed=42):
    """
    Load dataset with train/validation split.
    
    Returns:
        X_train, Y_train
        X_val,   Y_val
        X_test,  Y_test
    """

    if name not in ["mnist", "fashion_mnist"]:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    paths = get_dataset_paths(name)

    # Load raw data
    X = load_idx_images(paths["train_images"])
    y = load_idx_labels(paths["train_labels"])

    X_test = load_idx_images(paths["test_images"])
    y_test = load_idx_labels(paths["test_labels"])

    # Stratified train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_ratio,
        stratify=y,
        random_state=seed
    )

    # One-hot encode AFTER splitting
    Y_train = one_hot(y_train)
    Y_val   = one_hot(y_val)
    Y_test  = one_hot(y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# --------------------------------------------------
# Separate test-only loader (for inference.py)
# --------------------------------------------------

def load_test_dataset(name="mnist"):
    """Load only the test set (for inference script)."""
    if name not in ["mnist", "fashion_mnist"]:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    paths = get_dataset_paths(name)

    X_test = load_idx_images(paths["test_images"])
    y_test = load_idx_labels(paths["test_labels"])
    Y_test = one_hot(y_test)

    return X_test, Y_test



def show_samples(X, Y, num_samples=10):
    """
    Display random samples from dataset.
    X: (N, 784)
    Y: (N, 10) one-hot
    """

    indices = np.random.choice(X.shape[0], num_samples, replace=False)

    plt.figure(figsize=(num_samples, 2))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        plt.title(np.argmax(Y[idx]))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Sanity check (run this file directly)
# --------------------------------------------------

if __name__ == "__main__":
    Xtr, Ytr, Xval, Yval, Xte, Yte = load_dataset("mnist")

    print("Train:", Xtr.shape, Ytr.shape)
    print("Val:  ", Xval.shape, Yval.shape)
    print("Test: ", Xte.shape, Yte.shape)

    print("Range:", Xtr.min(), Xtr.max())
    print("One-hot check:", np.sum(Ytr[0]))

    show_samples(Xtr, Ytr, num_samples=10)