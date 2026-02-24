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
    t = np.tanh(z)
    return 1 - t**2

def softmax(z):
    """Softmax activation function."""
    z_shifted = z - np.max(z, axis=1, keepdims=True) # For numerical stability
    exp_z = np.exp(z_shifted) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_derivative(z):
    """Derivative of Softmax activation function."""
    s = softmax(z)
    n, c = s.shape
    jacobian = np.zeros((n, c, c), dtype=s.dtype)
    for i in range(n):
        si = s[i].reshape(-1, 1)
        jacobian[i] = np.diagflat(si) - (si @ si.T) 
    return jacobian
