import joblib
import pandas as pd

from lightgbm import LGBMRegressor

from data_loader import M5DataLoader
from feature_engineering import FeatureEngineer
from evaluate import calculate_metrics


FEATURES = [
    "sell_price",
    "month",
    "week",
    "day",
    "dayofweek",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
]

TARGET = "sales"


def main():

    loader = M5DataLoader(
        "data/raw"
    )

    sales = loader.load_sales()
    calendar = loader.load_calendar()
    prices = loader.load_prices()

    fe = FeatureEngineer()

    df = fe.run(
        sales,
        calendar,
        prices,
    )

    split_date = (
        df["date"]
        .sort_values()
        .iloc[int(len(df) * 0.8)]
    )

    train_df = df[
        df["date"] < split_date
    ]

    valid_df = df[
        df["date"] >= split_date
    ]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    preds = model.predict(
        X_valid
    )

    metrics = calculate_metrics(
        y_valid,
        preds,
    )

    print("\nValidation Metrics")
    print(metrics)

    joblib.dump(
        model,
        "models/lightgbm.pkl",
    )

    print(
        "\nSaved model -> models/lightgbm.pkl"
    )


if __name__ == "__main__":
    main()