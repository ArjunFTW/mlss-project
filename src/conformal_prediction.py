import numpy as np


class ConformalPredictor:

    def __init__(
        self,
        alpha=0.05,
    ):
        self.alpha = alpha
        self.quantile = None

    def fit(
        self,
        y_true,
        y_pred,
    ):

        residuals = np.abs(
            y_true - y_pred
        )

        self.quantile = np.quantile(
            residuals,
            1 - self.alpha,
        )

    def predict(
        self,
        preds,
    ):

        lower = (
            preds - self.quantile
        )

        upper = (
            preds + self.quantile
        )

        return (
            preds,
            lower,
            upper,
        )

    def coverage(
        self,
        y_true,
        lower,
        upper,
    ):

        covered = (
            (y_true >= lower)
            &
            (y_true <= upper)
        )

        return covered.mean()