from loss.loss import Loss


class Simple(Loss):
    """
    A loss (cost) class to calculate losses
    """
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        return error ** 2
