import numpy as np


def one_hot(Y):
    """

    :param Y:
    :return:
    """
    # create an array with shape Y samples, by number of choices in Y
    # e.g. (10000, 10)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # use numpy arange() function to iterate through Y rows 0-Y.size and set the Yth column to 1
    # aka one hot encode
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose the columns and rows to line up rows with the output
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
