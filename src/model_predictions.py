import pandas as pd


def predict_revenue(df, model):
    features = [
        "Units",
        "Cost_Price",
        "Selling_Price",
        "Month",
        "Day_of_Week",
        "Customer_Age",
        "Loyalty_Flag"
    ]

    X = df[features]

    df = df.copy()
    df["Predicted_Revenue"] = model.predict(X)

    return df


def predict_inventory(df, models):
    features = [
        "Units",
        "Selling_Price",
        "Revenue",
        "Margin",
        "Lead_Time_Days",
        "Customer_Age",
        "Loyalty_Flag",
        "Month",
        "Day_of_Week"
    ]

    X = df[features]

    X_scaled = models["classification_scaler"].transform(X)

    X_scaled = pd.DataFrame(
    X_scaled,
    columns=features,
    index=X.index
)
    df = df.copy()

    df["KNN_Prediction"] = models["knn"].predict(X_scaled.values)
    df["Decision_Tree_Prediction"] = models["decision_tree"].predict(X_scaled)
    df["SVM_Prediction"] = models["svm"].predict(X_scaled.values)
    df["Random_Forest_Prediction"] = models["random_forest"].predict(X_scaled)
    return df