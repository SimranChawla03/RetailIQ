import numpy as np
import pandas as pd


def calculate_restocking(df):

    df = df.copy()

    demand = (
        df.groupby(["Brand", "Category"])["Units"]
        .transform("mean")
    )

    df["Average_Demand"] = demand

    df["Required_Stock"] = (
        df["Average_Demand"] * df["Lead_Time_Days"]
    )

    # High-performance vectorized calculation
    conditions = [
        df["Stock_On_Hand"] <= df["Reorder_Level"],
        df["Stock_On_Hand"] < df["Required_Stock"]
    ]
    choices = ["Restock Now", "Restock Soon"]
    df["Restock_Status"] = np.select(conditions, choices, default="Stock Sufficient")

    diff = df["Required_Stock"] - df["Stock_On_Hand"]
    df["Suggested_Quantity"] = np.maximum(0, np.ceil(diff)).astype(int)

    return df