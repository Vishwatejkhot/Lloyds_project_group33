import os
import joblib
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

_BEST = {"growth": "XGBoost", "risk": "XGBoost", "lending": "XGBoost"}


@st.cache_resource(show_spinner="Loading models…")
def load_all_models() -> dict:
    """Load sample-trained models for all three labels. Cached for the session."""
    result = {}
    for label in ["growth", "risk", "lending"]:
        path = os.path.join(MODELS_DIR, f"{label}_model_data.joblib")
        if os.path.exists(path):
            result[label] = joblib.load(path)
    return result


def best_model(models: dict, label: str):
    """Return (model, scaler, feature_list, model_name) for the best model of a label."""
    data = models[label]
    name = _BEST.get(label, "XGBoost")
    if name not in data["models"]:
        name = list(data["models"].keys())[0]
    return data["models"][name], data["scaler"], data["features"], name
