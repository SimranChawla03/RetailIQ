import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


def calculate_restocking(df):
    df = df.copy()

    # Calculate demand by brand and category if available
    if "Brand" in df.columns and "Category" in df.columns and "Units" in df.columns:
        demand = (
            df.groupby(["Brand", "Category"])["Units"]
            .transform("mean")
        )
    elif "Units" in df.columns:
        demand = df["Units"].transform("mean")
    else:
        demand = pd.Series(1.0, index=df.index)

    df["Average_Demand"] = demand.fillna(1.0)

    lead_time = (
        pd.to_numeric(df["Lead_Time_Days"], errors="coerce").fillna(5.0)
        if "Lead_Time_Days" in df.columns
        else pd.Series(5.0, index=df.index)
    )

    df["Required_Stock"] = df["Average_Demand"] * lead_time

    stock_on_hand = (
        pd.to_numeric(df["Stock_On_Hand"], errors="coerce")
        if "Stock_On_Hand" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    reorder_level = (
        pd.to_numeric(df["Reorder_Level"], errors="coerce")
        if "Reorder_Level" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    conditions = [
        stock_on_hand.notna() & reorder_level.notna() & (stock_on_hand <= reorder_level),
        stock_on_hand.notna() & (stock_on_hand < df["Required_Stock"])
    ]
    choices = ["Restock Now", "Restock Soon"]
    df["Restock_Status"] = np.select(conditions, choices, default="Stock Sufficient")

    diff = df["Required_Stock"] - stock_on_hand.fillna(0)
    df["Suggested_Quantity"] = np.maximum(0, np.ceil(diff.fillna(0))).astype(int)

    return df