import numpy as np


class ReLU:
    """
    A rectified linear unit (ReLU) is an activation function that introduces the property of
    nonlinearity to a deep learning model and solves the vanishing gradients issue.
    """
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
