"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros(output_size)

    def forward(self, X):
        """Compute the forward pass."""
        return X @ self.W + self.b

    def compute_gradients(self, X, d_out):
        """Compute gradients w.r.t. weights and biases."""
        dW = X.T @ d_out
        db = np.sum(d_out, axis=0)
        return dW, db
    