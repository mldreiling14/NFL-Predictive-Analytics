import joblib


def load_model(model_path="models/win_probability_model.joblib"):
    return joblib.load(model_path)