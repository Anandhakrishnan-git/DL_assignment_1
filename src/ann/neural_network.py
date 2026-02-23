"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import NeuralLayer
from .optimizers import SGD, Momentum, Adam, Nadam

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        pass

        self.layers = []  # List of NeuralLayer instances
        self.optimizer = None  # Optimizer instance (e.g., SGD, Adam)
        self.loss_function = None  # Loss function (e.g., cross-entropy)

        
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        pass

        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        #pass
        
        # Compute initial gradient from loss function
        grad = self.loss_function.gradient(y_true, y_pred)
        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            grad_w, grad_b = layer.compute_gradients(grad)
            # Update gradients for the optimizer
            self.optimizer.update([layer.W, layer.b], [grad_w, grad_b])
            # Update grad for next layer
            grad = grad @ layer.W.T  # Chain rule
        
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        pass

        self.optimizer.update_weights()

    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        pass
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Backward pass and weight update
                self.backward(y_batch, y_pred)
        

                
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        pass
        
        y_pred = self.forward(X)
        loss = self.loss_function.compute(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy
    
