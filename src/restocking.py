import pandas as pd
import math


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

    def get_status(row):

        if row["Stock_On_Hand"] <= row["Reorder_Level"]:
            return "Restock Now"

        elif row["Stock_On_Hand"] < row["Required_Stock"]:
            return "Restock Soon"

        else:
            return "Stock Sufficient"

    df["Restock_Status"] = df.apply(get_status, axis=1)

    df["Suggested_Quantity"] = (
        df["Required_Stock"] - df["Stock_On_Hand"]
    ).apply(lambda x: max(0, math.ceil(x)))

    return df