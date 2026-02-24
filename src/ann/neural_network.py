"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import os
import argparse
import numpy as np

from .neural_layer import NeuralLayer
from .optimizers import SGD, Momentum, NAG
from .objective_functions import CE_loss, MSE
from .activations import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    softmax,
    softmax_derivative,
)


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
        self.input_size = 784       # number of pixels in a 28x28 image
        self.output_size = 10       # number of classes in the dataset (digits 0-9 for MNIST)
        
        requested_layers = int(getattr(cli_args, "num_layers", 2))
        hidden_sizes = list(getattr(cli_args, "hidden_size", [128, 64]))
        if len(hidden_sizes) < requested_layers:
            hidden_sizes += [hidden_sizes[-1]] * (requested_layers - len(hidden_sizes))
        if len(hidden_sizes) > requested_layers:
            hidden_sizes = hidden_sizes[:requested_layers]
        self.hidden_sizes = hidden_sizes

        self.activation_name = getattr(cli_args, "activation", "relu")
        self.weight_init = getattr(cli_args, "weight_init", "xavier")
        self.loss_name = getattr(cli_args, "loss", "cross_entropy")
        self.learning_rate = float(getattr(cli_args, "learning_rate", 0.001))
        optimizer_name = getattr(cli_args, "optimizer", "sgd")

        self.layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]        # full architecture of the network

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                NeuralLayer(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    weight_init=self.weight_init,
                )
            )

        self.optimizer = self._build_optimizer(optimizer_name, self.learning_rate)                      # initialize the optimizer
        self.activation, self.activation_derivative = self._build_activation(self.activation_name)      # initialize the activation function and its derivative
        self._a_cache = []  # cache for activations during forward pass, used in backward pass for grad compute
        self._z_cache = []  # cache for pre-activation values during forward pass, used in backward pass for grad compute

    def _build_optimizer(self, name, lr):
        if name == "sgd":
            return SGD(learning_rate=lr)
        if name == "momentum":
            return Momentum(learning_rate=lr)
        if name == "nag":
            return NAG(learning_rate=lr)


    def _build_activation(self, name):
        if name == "relu":
            return relu, relu_derivative
        if name == "sigmoid":
            return sigmoid, sigmoid_derivative
        if name == "tanh":
            return tanh, tanh_derivative

    def _compute_loss(self, y_true, y_pred):
        if self.loss_name == "cross_entropy":
            return CE_loss(y_true, y_pred)
        if self.loss_name == "mse":
            return MSE(y_true, y_pred)
        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def forward(self, X):
        """
        Forward propagation through all layers.

        Args:
            X: Input data

        Returns:
            Output probabilities
        """
        self._a_cache = [X]
        self._z_cache = []

        a = X
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            self._z_cache.append(z)
            if i == len(self.layers) - 1: # For the output layer, apply softmax to get probabilities
                a = softmax(z)
            else:
                a = self.activation(z)
            self._a_cache.append(a)
        return a

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        m = y_true.shape[0]

        if self.loss_name == "cross_entropy":
            delta = (y_pred - y_true) / m

        grad_w_list = []
        grad_b_list = []

        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]          
            a_prev = self._a_cache[layer_idx]
            dW, db = layer.compute_gradients(a_prev, delta)
            
            # Insert computed gradients at the beginning of the list to maintain correct order
            grad_w_list.insert(0, dW) 
            grad_b_list.insert(0, db)

            if layer_idx > 0:
                delta = (delta @ layer.W.T) * self.activation_derivative(self._z_cache[layer_idx - 1])
        
        return grad_w_list, grad_b_list
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for i, layer in enumerate(self.layers):
            self.optimizer.update([layer.W, layer.b], [layer.grad_W, layer.grad_b])
        

    def train(self, X_train, y_train, epochs, batch_size, X_val=None, y_val=None, verbose=True):
        """
        Train the network for specified epochs.
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)      # Shuffle the data at the beginning of each epoch
            X_epoch = X_train[indices]
            y_epoch = y_train[indices]

            # Process data in batches
            for start in range(0, num_samples, batch_size): 
                end = start + batch_size
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]
                y_pred = self.forward(X_batch) 
                self.backward(y_batch, y_pred) 
                self.update_weights() 

            train_loss, train_acc = self.evaluate(X_train, y_train)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                    )

        return history

    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred = self.forward(X)
        loss = self._compute_loss(y, y_pred)

        # compares the predicted class with the true class for each sample, and then takes the mean
        accuracy = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)))         
        return float(loss), accuracy

    def save_model(self, model_path):
        """Save model weights and metadata as a single .npy file."""
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Prepare model data
        model_data = {
            "hidden_sizes": self.hidden_sizes,
            "activation_name": self.activation_name,
            "weight_init": self.weight_init,
            "loss_name": self.loss_name,
            "learning_rate": self.learning_rate,
            "optimizer": type(self.optimizer).__name__.lower(),
            "weights": [layer.W for layer in self.layers],
            "biases": [layer.b for layer in self.layers],
        }
        np.save(model_path, model_data, allow_pickle=True)

