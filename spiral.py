import matplotlib.pyplot as plt

import nnfs
import numpy as np
from nnfs.datasets import spiral_data

from activation import ReLU, Softmax
from layer import Dense
from loss import CategoricalCrossentropy


def plot():
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    plt.show()


nnfs.init()

X, y = spiral_data(samples=100, classes=3)
# plot()

