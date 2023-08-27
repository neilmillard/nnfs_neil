import numpy as np


class Loss:
    """
    The parent loss class
    """
    def calculate(self, output, y):
        """
        Calculate the loss of a batch of output predictions (output) when compared to training answers (y)
        :param output: np array of outputs
        :param y: array of training answers
        :return:
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
