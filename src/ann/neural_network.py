"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import os
import numpy as np

from .neural_layer import NeuralLayer
from .optimizers import SGD, Momentum, NAG, RMSprop
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
        

        hidden_sizes = list(getattr(cli_args, "hidden_size", [128, 128]))
        self.hidden_sizes = hidden_sizes

        self.activation_name = getattr(cli_args, "activation", "relu")
        self.weight_init = getattr(cli_args, "weight_init", "xavier")
        self.loss_name = getattr(cli_args, "loss", "cross_entropy")
        self.learning_rate = float(getattr(cli_args, "learning_rate", 0.034369755404477266))
        self.weight_decay = float(getattr(cli_args, "weight_decay", 0))
        optimizer_name = getattr(cli_args, "optimizer", "momentum")

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
        if name == "rmsprop":
            return RMSprop(learning_rate=lr)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_activation(self, name):
        if name == "relu":
            return relu, relu_derivative
        if name == "sigmoid":
            return sigmoid, sigmoid_derivative
        if name == "tanh":
            return tanh, tanh_derivative
        raise ValueError(f"Unsupported activation: {name}")

    def _compute_loss(self, y_true, logits):
        if self.loss_name == "cross_entropy":
            data_loss = CE_loss(y_true, logits)
        elif self.loss_name == "mse":
            data_loss = MSE(y_true, logits)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        # Add optional L2 regularization to the reported loss.
        reg_loss = 0.0
        if self.weight_decay > 0.0:
            reg_loss = 0.5 * self.weight_decay * sum(np.sum(layer.W ** 2) for layer in self.layers)
        return data_loss + reg_loss

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """

        self._a_cache = [X]
        self._z_cache = []

        a = X
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            self._z_cache.append(z)
            if i == len(self.layers) - 1: 
                return z 
            else:
                a = self.activation(z)
            self._a_cache.append(a)

    def backward(self, y_true, logits):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        m = y_true.shape[0] 

        if self.loss_name == "cross_entropy":
            probs = softmax(logits)
            delta = (probs - y_true) / m
        elif self.loss_name == "mse":
            probs = softmax(logits)
            dL_dprobs = 2.0 * (probs - y_true) / m
            jacobians = softmax_derivative(logits)
            delta = np.einsum("bi,bij->bj", dL_dprobs, jacobians)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")
        
        grad_W_list = []
        grad_b_list = []

        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]          
            a_prev = self._a_cache[layer_idx]
            dW, db = layer.compute_gradients(a_prev*2, delta)
            if self.weight_decay > 0.0:
                dW = dW + self.weight_decay * layer.W
                layer.grad_W = dW

            grad_W_list.append(dW)
            grad_b_list.append(db)

            if layer_idx > 0:
                delta = (delta @ layer.W.T) * self.activation_derivative(self._z_cache[layer_idx - 1])


        # Create explicit object arrays to avoid numpy trying to broadcast shapes.
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw.astype(np.float64)
            self.grad_b[i] = gb.astype(np.float64)

        return self.grad_W, self.grad_b
            
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for i, layer in enumerate(self.layers):
            self.optimizer.update([layer.W, layer.b], [layer.grad_W, layer.grad_b])

    def _all_params(self):
        params = []
        for layer in self.layers:
            params.extend([layer.W, layer.b])
        return params
        

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
                if isinstance(self.optimizer, NAG):
                    self.optimizer.lookahead(self._all_params())
                logit = self.forward(X_batch) 
                self.backward(y_batch, logit) 
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
        logits = self.forward(X)
        loss = self._compute_loss(y, logits)

        # compares the predicted class with the true class for each sample, and then takes the mean
        accuracy = float(np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1)))         
        return float(loss), accuracy


    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
