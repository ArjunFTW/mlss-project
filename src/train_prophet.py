import pandas as pd
import joblib

from prophet import Prophet

from data_loader import M5DataLoader
from feature_engineering import FeatureEngineer


MODEL_PATH = "models/prophet_model.pkl"


def prepare_prophet_data(df):
    prophet_df = (
        df.groupby("date")["sales"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "date": "ds",
                "sales": "y",
            }
        )
    )

    return prophet_df


def main():

    loader = M5DataLoader("data/raw")

    sales = loader.load_sales()
    calendar = loader.load_calendar()
    prices = loader.load_prices()

    fe = FeatureEngineer()

    df = fe.run(
        sales,
        calendar,
        prices,
    )

    prophet_df = prepare_prophet_data(df)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    model.fit(prophet_df)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"Saved Prophet model to {MODEL_PATH}")


if __name__ == "__main__":
    main()