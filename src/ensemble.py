import numpy as np


class ForecastEnsemble:

    def __init__(
        self,
        lightgbm_weight=0.7,
        prophet_weight=0.3,
    ):
        self.lightgbm_weight = lightgbm_weight
        self.prophet_weight = prophet_weight

    def predict(
        self,
        lightgbm_preds,
        prophet_preds,
    ):

        return (
            self.lightgbm_weight
            * np.array(lightgbm_preds)
            + self.prophet_weight
            * np.array(prophet_preds)
        )