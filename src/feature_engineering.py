import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self):
        pass

    def melt_sales(self, sales_df):
        """
        Convert M5 wide format into long format.
        """

        id_cols = [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ]

        day_cols = [c for c in sales_df.columns if c.startswith("d_")]

        long_df = sales_df.melt(
            id_vars=id_cols,
            value_vars=day_cols,
            var_name="d",
            value_name="sales",
        )

        return long_df

    def merge_calendar(self, sales_long, calendar_df):

        sales_long = sales_long.merge(
            calendar_df,
            on="d",
            how="left",
        )

        return sales_long

    def merge_prices(self, sales_long, prices_df):

        sales_long = sales_long.merge(
            prices_df,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        return sales_long

    def create_lag_features(self, df):

        group_cols = ["store_id", "item_id"]

        for lag in [1, 7, 14, 28]:

            df[f"lag_{lag}"] = (
                df.groupby(group_cols)["sales"]
                .shift(lag)
            )

        return df

    def create_rolling_features(self, df):

        group_cols = ["store_id", "item_id"]

        for window in [7, 14, 28]:

            df[f"rolling_mean_{window}"] = (
                df.groupby(group_cols)["sales"]
                .transform(
                    lambda x:
                    x.shift(1)
                     .rolling(window)
                     .mean()
                )
            )

            df[f"rolling_std_{window}"] = (
                df.groupby(group_cols)["sales"]
                .transform(
                    lambda x:
                    x.shift(1)
                     .rolling(window)
                     .std()
                )
            )

        return df

    def create_date_features(self, df):

        df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["quarter"] = df["date"].dt.quarter

        df["is_weekend"] = (
            df["dayofweek"] >= 5
        ).astype(int)

        return df

    def run(
        self,
        sales_df,
        calendar_df,
        prices_df,
    ):

        df = self.melt_sales(sales_df)

        df = self.merge_calendar(
            df,
            calendar_df,
        )

        df = self.merge_prices(
            df,
            prices_df,
        )

        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_date_features(df)

        df = df.sort_values(
            ["store_id", "item_id", "date"]
        )

        df = df.dropna()

        return df