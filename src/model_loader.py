import joblib


def load_models():

    models = {}

    model_files = {

        "regression":
            "models/regression/linear_regression.pkl",

        "kmeans":
            "models/clustering/kmeans.pkl",

        "clustering_scaler":
            "models/clustering/scaler.pkl",

        "knn":
            "models/classification/tuned/knn.pkl",

        "decision_tree":
            "models/classification/tuned/decision_tree.pkl",

        "svm":
            "models/classification/tuned/svm.pkl",

        "random_forest":
            "models/classification/tuned/random_forest.pkl",

        "classification_scaler":
            "models/classification/tuned/scaler.pkl",

        "apriori":
            "models/association/apriori_rules.pkl",

    }

    for name, path in model_files.items():

        try:
            models[name] = joblib.load(path)

        except FileNotFoundError:
            # Model file missing — skip it.
            # The rest of the application will check
            # for the key before using the model.
            pass

    return models