import numpy as np


class Softmax:
    """
    The softmax function, also known as softargmax or normalized exponential function,
    converts a vector of K real numbers into a probability distribution of K possible outcomes.
    This data is used to train a network under 'loss log' or 'cross-entropy'
    """
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
