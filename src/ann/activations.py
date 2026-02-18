"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

def relu(z):
    """Rectified Linear Unit activation function."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU activation function."""
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of Sigmoid activation function."""
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    """Hyperbolic Tangent activation function."""
    return np.tanh(z)

def tanh_derivative(z):
    """Derivative of Tanh activation function."""
    t = tanh(z)
    return 1 - t**2

def softmax(z):
    """Softmax activation function."""
    pass

def softmax_derivative(z):
    """Derivative of Softmax activation function."""
    pass
