import pandas as pd


def predict_clusters(df, models):

    features = [
        "Units",
        "Revenue",
        "Selling_Price",
        "Margin",
        "Stock_On_Hand",
        "Reorder_Level",
        "Lead_Time_Days"
    ]

    X = df[features]

    X_scaled = models["clustering_scaler"].transform(X)

    df = df.copy()

    df["KMeans_Cluster"] = models["kmeans"].predict(X_scaled)

    return df