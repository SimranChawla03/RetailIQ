import pandas as pd


def get_associations(product, rules_df, min_lift=1.0):

    matches = rules_df[
        rules_df["Product"].astype(str).str.lower() == product.lower()
    ].copy()

    matches = matches[matches["Lift"] >= min_lift]

    matches = matches.sort_values(
        by=["Lift", "Confidence"],
        ascending=False
    )

    return matches[
        ["Product", "Recommended_Product", "Support", "Confidence", "Lift"]
    ]