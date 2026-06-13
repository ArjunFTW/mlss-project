from src.ensemble import ForecastEnsemble
from src.conformal_prediction import ConformalPredictor

import numpy as np

y_true = np.array([10, 12, 15, 18])

preds = np.array([11, 13, 14, 17])

cp = ConformalPredictor()

cp.fit(y_true, preds)

preds, lower, upper = cp.predict(preds)

print("Coverage:", cp.coverage(
    y_true,
    lower,
    upper,
))

ensemble = ForecastEnsemble()

print(
    ensemble.predict(
        [10, 20],
        [12, 18],
    )
)