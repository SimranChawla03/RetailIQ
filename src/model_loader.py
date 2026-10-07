import joblib


def load_models():

    models = {}

    models["regression"] = joblib.load(
        "models/regression/linear_regression.pkl"
    )

    models["kmeans"] = joblib.load(
        "models/clustering/kmeans.pkl"
    )

    models["clustering_scaler"] = joblib.load(
        "models/clustering/scaler.pkl"
    )

    models["knn"] = joblib.load(
        "models/classification/tuned/knn.pkl"
    )

    models["decision_tree"] = joblib.load(
        "models/classification/tuned/decision_tree.pkl"
    )

    models["svm"] = joblib.load(
        "models/classification/tuned/svm.pkl"
    )

    models["random_forest"] = joblib.load(
        "models/classification/tuned/random_forest.pkl"
    )

    models["classification_scaler"] = joblib.load(
        "models/classification/tuned/scaler.pkl"
    )

    models["apriori"] = joblib.load(
        "models/association/apriori_rules.pkl"
    )

    return models