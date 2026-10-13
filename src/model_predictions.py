import numpy as np
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

    df = df.copy()
    X = pd.DataFrame(index=df.index)
    for col in features:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            X[col] = np.nan

    # Safe imputation for inference
    if "Selling_Price" in X.columns:
        X["Selling_Price"] = X["Selling_Price"].fillna(70.0)
    if "Cost_Price" in X.columns:
        X["Cost_Price"] = X["Cost_Price"].fillna(X["Selling_Price"] * 0.75)
    if "Units" in X.columns:
        X["Units"] = X["Units"].fillna(1.0)
    if "Month" in X.columns:
        X["Month"] = X["Month"].fillna(1)
    if "Day_of_Week" in X.columns:
        X["Day_of_Week"] = X["Day_of_Week"].fillna(0)
    if "Customer_Age" in X.columns:
        X["Customer_Age"] = X["Customer_Age"].fillna(35.0)
    if "Loyalty_Flag" in X.columns:
        X["Loyalty_Flag"] = X["Loyalty_Flag"].fillna(0)

    df["Predicted_Revenue"] = model.predict(X[features])
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

    df = df.copy()
    X = pd.DataFrame(index=df.index)
    for col in features:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            X[col] = np.nan

    # Safe imputation for inference
    if "Selling_Price" in X.columns:
        X["Selling_Price"] = X["Selling_Price"].fillna(50.0)
    if "Units" in X.columns:
        X["Units"] = X["Units"].fillna(1.0)
    if "Revenue" in X.columns:
        X["Revenue"] = X["Revenue"].fillna(X["Units"] * X["Selling_Price"])
    if "Margin" in X.columns:
        X["Margin"] = X["Margin"].fillna(X["Revenue"] * 0.25)
    if "Lead_Time_Days" in X.columns:
        X["Lead_Time_Days"] = X["Lead_Time_Days"].fillna(5.0)
    if "Customer_Age" in X.columns:
        X["Customer_Age"] = X["Customer_Age"].fillna(35.0)
    if "Loyalty_Flag" in X.columns:
        X["Loyalty_Flag"] = X["Loyalty_Flag"].fillna(0.0)
    if "Month" in X.columns:
        X["Month"] = X["Month"].fillna(1.0)
    if "Day_of_Week" in X.columns:
        X["Day_of_Week"] = X["Day_of_Week"].fillna(0.0)

    X_scaled = models["classification_scaler"].transform(X[features])

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=features,
        index=X.index
    )

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


def forecast_daily_sales(daily_df, model, horizon_days=7):
    """
    Generate rolling forward forecast for future days using the trained regression model.
    """
    if daily_df.empty or model is None:
        return pd.DataFrame()

    df = daily_df.copy().sort_values("Date").reset_index(drop=True)
    forecasts = []
    
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    current_history = df.copy()

    for step in range(1, horizon_days + 1):
        target_date = last_date + pd.Timedelta(days=step)
        day_of_week = target_date.dayofweek
        month = target_date.month
        day_of_month = target_date.day
        is_weekend = int(day_of_week in [5, 6])

        # Lag features from current history
        rev_hist = current_history["Revenue"].values
        units_hist = current_history["Units"].values
        n = len(rev_hist)

        lag_1_rev = rev_hist[-1] if n >= 1 else 0
        lag_2_rev = rev_hist[-2] if n >= 2 else lag_1_rev
        lag_7_rev = rev_hist[-7] if n >= 7 else lag_1_rev
        lag_14_rev = rev_hist[-14] if n >= 14 else lag_7_rev

        lag_1_units = units_hist[-1] if n >= 1 else 0
        lag_2_units = units_hist[-2] if n >= 2 else lag_1_units
        lag_7_units = units_hist[-7] if n >= 7 else lag_1_units
        lag_14_units = units_hist[-14] if n >= 14 else lag_7_units

        roll_7_rev = rev_hist[-7:].mean() if n >= 7 else rev_hist.mean()
        roll_14_rev = rev_hist[-14:].mean() if n >= 14 else rev_hist.mean()
        roll_7_std = rev_hist[-7:].std() if n >= 7 else 0.0

        features = pd.DataFrame([{
            "Lag_1_Revenue": lag_1_rev,
            "Lag_2_Revenue": lag_2_rev,
            "Lag_7_Revenue": lag_7_rev,
            "Lag_14_Revenue": lag_14_rev,
            "Lag_1_Units": lag_1_units,
            "Lag_2_Units": lag_2_units,
            "Lag_7_Units": lag_7_units,
            "Lag_14_Units": lag_14_units,
            "Roll_7_Revenue_Mean": roll_7_rev,
            "Roll_14_Revenue_Mean": roll_14_rev,
            "Roll_7_Revenue_Std": roll_7_std,
            "Day_of_Week": day_of_week,
            "Month": month,
            "Day_of_Month": day_of_month,
            "Is_Weekend": is_weekend
        }])

        pred_rev = float(model.predict(features)[0])
        # Approximate units using recent average revenue-per-unit
        avg_rev_per_unit = (rev_hist[-7:].sum() / max(1, units_hist[-7:].sum())) if n >= 7 else 130.0
        pred_units = max(1, int(round(pred_rev / max(1.0, avg_rev_per_unit))))

        forecast_row = {
            "Date": target_date.strftime("%Y-%m-%d"),
            "Forecasted_Revenue": round(max(0.0, pred_rev), 2),
            "Forecasted_Units": pred_units,
            "Day_Name": target_date.strftime("%A")
        }
        forecasts.append(forecast_row)

        # Append to temporary history for rolling multi-step lags
        new_row = pd.DataFrame([{
            "Date": target_date.date(),
            "Revenue": max(0.0, pred_rev),
            "Units": pred_units
        }])
        current_history = pd.concat([current_history, new_row], ignore_index=True)

    return pd.DataFrame(forecasts)
