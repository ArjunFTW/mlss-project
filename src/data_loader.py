import pandas as pd
from pathlib import Path


class M5DataLoader:
    """
    Loads the M5 Forecasting Accuracy dataset.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_sales(self):
        return pd.read_csv(
            self.data_dir / "sales_train_validation.csv"
        )

    def load_calendar(self):
        return pd.read_csv(
            self.data_dir / "calendar.csv"
        )

    def load_prices(self):
        return pd.read_csv(
            self.data_dir / "sell_prices.csv"
        )

    def load_all(self):
        sales = self.load_sales()
        calendar = self.load_calendar()
        prices = self.load_prices()

        return {
            "sales": sales,
            "calendar": calendar,
            "prices": prices,
        }


if __name__ == "__main__":
    loader = M5DataLoader("data/raw")

    datasets = loader.load_all()

    print("Sales Shape:", datasets["sales"].shape)
    print("Calendar Shape:", datasets["calendar"].shape)
    print("Prices Shape:", datasets["prices"].shape)