"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSprop, Adam, Nadam
"""

import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """Update parameters based on gradients."""
        raise NotImplementedError("Must be implemented in subclass.")


class SGD(Optimizer):
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            key = id(param)
            if key not in self.velocity:
                self.velocity[key] = 0
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grad
            param -= self.velocity[key]


class NAG(Momentum):
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            key = id(param)
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param)
            v_prev = self.velocity[key].copy()
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grad
            param += -self.momentum * v_prev + (1 + self.momentum) * self.velocity[key]

