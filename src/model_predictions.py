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


def predict_inventory(df, models, primary_only=True):
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

    if "classification_scaler" not in models:
        return df

    X = df[features]

    X_scaled = models["classification_scaler"].transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=features,
        index=X.index
    )

    df = df.copy()

    if primary_only:
        # Fast path: predict using the primary champion model
        for column_name, model_key in [
            ("Random_Forest_Prediction", "random_forest"),
            ("Decision_Tree_Prediction", "decision_tree"),
            ("KNN_Prediction", "knn"),
        ]:
            if model_key in models:
                df[column_name] = models[model_key].predict(X_scaled)
                break
        return df

    classifiers = {
        "KNN_Prediction": "knn",
        "Decision_Tree_Prediction": "decision_tree",
        "SVM_Prediction": "svm",
        "Random_Forest_Prediction": "random_forest",
    }

    for column_name, model_key in classifiers.items():
        if model_key in models:
            df[column_name] = models[model_key].predict(X_scaled)

    return df