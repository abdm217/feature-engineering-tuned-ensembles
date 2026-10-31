
import pandas as pd


def engineer_features(data):
    """
    Create domain-based features for house price prediction.

    Features:
    1. TotalSF: Basement + first floor + second floor area.
    2. HouseAge: Age of the house at the time of sale.
    3. TotalBathrooms: Full and half bathrooms combined.
    4. HasRemodeled: Whether the house was remodeled after construction.
    5. TotalPorchSF: Combined porch and deck area.
    """

    data = data.copy()

    data["TotalSF"] = (
        data["TotalBsmtSF"].fillna(0)
        + data["1stFlrSF"].fillna(0)
        + data["2ndFlrSF"].fillna(0)
    )

    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]

    data["TotalBathrooms"] = (
        data["FullBath"].fillna(0)
        + 0.5 * data["HalfBath"].fillna(0)
        + data["BsmtFullBath"].fillna(0)
        + 0.5 * data["BsmtHalfBath"].fillna(0)
    )

    data["HasRemodeled"] = (
        data["YearRemodAdd"] > data["YearBuilt"]
    ).astype(int)

    data["TotalPorchSF"] = (
        data["OpenPorchSF"].fillna(0)
        + data["EnclosedPorch"].fillna(0)
        + data["3SsnPorch"].fillna(0)
        + data["ScreenPorch"].fillna(0)
        + data["WoodDeckSF"].fillna(0)
    )

    return data
