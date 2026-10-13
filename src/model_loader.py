import os
import joblib


def load_models(base_dir=None):
    if base_dir is None:
        # Resolve project root from this file location (src/ -> project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    models = {}

    model_files = {
        # Regression models
        "regression": os.path.join(base_dir, "models/regression/linear_regression.pkl"),
        # Classification models (Tuned)
        "knn": os.path.join(base_dir, "models/classification/tuned/knn.pkl"),
        "decision_tree": os.path.join(base_dir, "models/classification/tuned/decision_tree.pkl"),
        "svm": os.path.join(base_dir, "models/classification/tuned/svm.pkl"),
        "random_forest": os.path.join(base_dir, "models/classification/tuned/random_forest.pkl"),
        "classification_scaler": os.path.join(base_dir, "models/classification/tuned/scaler.pkl"),
        # Upgraded models
        "forecasting_regressor": os.path.join(base_dir, "models/forecasting/forecast_regressor.pkl"),
        "forecasting_scaler": os.path.join(base_dir, "models/forecasting/forecast_scaler.pkl"),
        "demand_trend_classifier": os.path.join(base_dir, "models/classification/upgraded/demand_trend_classifier.pkl"),
        "demand_trend_scaler": os.path.join(base_dir, "models/classification/upgraded/demand_trend_scaler.pkl"),
        "inventory_decision_classifier": os.path.join(base_dir, "models/classification/upgraded/inventory_decision_classifier.pkl"),
        "inventory_decision_scaler": os.path.join(base_dir, "models/classification/upgraded/inventory_decision_scaler.pkl"),
    }

    for name, path in model_files.items():
        try:
            if os.path.exists(path):
                models[name] = joblib.load(path)
        except Exception:
            pass

    return models