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
        self.b = np.zeros(output_size, dtype=np.float64)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
    
    def _initialize_weights(self, method):
        if method == 'xavier':
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size)).astype(np.float64)
        if method == 'zeros':
            return np.zeros((self.input_size, self.output_size), dtype=np.float64)
        if method == 'random':
            return (np.random.randn(self.input_size, self.output_size) * 0.01).astype(np.float64)
        raise ValueError(f"Unknown weight initialization method: {method}")

    def forward(self, X):
        """Compute the forward pass."""
        return X @ self.W + self.b

    def compute_gradients(self, a_prev, d_out):
        """Compute gradients w.r.t. weights and biases."""
        dW = a_prev.T @ d_out # shape (input_size, batch_size) @ (batch_size, output_size) -> (input_size, output_size)
        db = np.sum(d_out, axis=0) # shape (batch_size, output_size) -> (output_size,)
        self.grad_W = dW
        self.grad_b = db
        self.grad_norm = np.sqrt(np.sum(dW**2) + np.sum(db**2))
        return dW, db
    

    
