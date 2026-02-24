"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class NeuralLayer:
    def __init__(self, input_size, output_size, weight_init='xavier'):
        self.input_size = input_size
        self.output_size = output_size
        self.W = self._initialize_weights(weight_init)
        self.b = np.zeros(output_size)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
    
    def _initialize_weights(self, method):
        if method == 'xavier':
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        if method == 'he':
            std = np.sqrt(2 / self.input_size)
            return np.random.randn(self.input_size, self.output_size) * std

    def forward(self, X):
        """Compute the forward pass."""
        return X @ self.W + self.b

    def compute_gradients(self, X, d_out):
        """Compute gradients w.r.t. weights and biases."""
        dW = X.T @ d_out
        db = np.sum(d_out, axis=0)
        self.grad_W = dW
        self.grad_b = db
        return dW, db
    
