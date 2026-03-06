"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSprop
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
                self.velocity[key] = np.zeros_like(param)
            self.velocity[key] = self.momentum * self.velocity[key] + self.learning_rate * grad
            param -= self.velocity[key]


class NAG(Momentum):
    def lookahead(self, params):
        for param in params:
            key = id(param)
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param)
            param -= self.momentum * self.velocity[key]

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            key = id(param)
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param)
            v_prev = self.velocity[key].copy()
            self.velocity[key] = self.momentum * self.velocity[key] + self.learning_rate * grad
            # `param` is at lookahead position: theta_t - mu * v_t.
            # Move it to theta_t - v_t+1, where v_t+1 = mu * v_t - lr * grad(theta_t - mu * v_t).
            # theta_t+1 = theta_t - v_t+1 = (theta_t - mu*v_t) + mu*v_t - v_t+1 = lookahead/param + mu*v_t - v_t+1
            param += self.momentum * v_prev - self.velocity[key]

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.sq = {}

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            key = id(param)
            if key not in self.sq:
                self.sq[key] = np.zeros_like(param)
            self.sq[key] = self.beta * self.sq[key] + (1 - self.beta) * (grad ** 2)
            param -= self.learning_rate * grad / (np.sqrt(self.sq[key]) + self.epsilon)
