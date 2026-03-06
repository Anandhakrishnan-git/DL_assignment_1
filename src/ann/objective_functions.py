"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
from .activations import softmax
import numpy as np

def CE_loss(y_true, y_pred):
    """Cross-Entropy loss function."""
    y_pred  = softmax(y_pred)  # Use softmax to convert logits to probabilities
    
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def CE_loss_der(y_true, y_pred):
    """Derivative of Cross-Entropy loss function."""
    y_pred  = softmax(y_pred)  # Use softmax to convert logits to probabilities
    epsilon = 1e-15  # To prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (-y_true / y_pred) / y_true.shape[0]

def MSE(y_true, y_pred):
    """Mean Squared Error (MSE) loss function."""
    y_pred  = softmax(y_pred)  # Use softmax to convert logits to probabilities

    return np.mean((y_true - y_pred) ** 2)

def MSE_der(y_true, y_pred):
    """Derivative of Mean Squared Error (MSE) loss function."""
    y_pred  = softmax(y_pred)  # Use softmax to convert logits to probabilities
    return 2 * (y_pred - y_true) / y_true.size
