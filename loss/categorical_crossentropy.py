import numpy as np

from loss.loss import Loss


class CategoricalCrossentropy(Loss):
    """
    A loss class to calculate losses from Softmax outputs
    array of training answers in one hot encoded
    """
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            # Scaler class value
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # elif len(y_true.shape) == 2:
        else:
            # handle one hot encoded vectors
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
