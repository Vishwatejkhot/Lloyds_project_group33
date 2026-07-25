import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


@st.cache_resource(show_spinner=False)
def _get_explainer(_model):
    return shap.TreeExplainer(_model)


def compute_shap(model, X_scaled: np.ndarray):
    """Return (shap_values array, expected_value) for a single scaled row."""
    explainer = _get_explainer(model)
    sv = explainer.shap_values(X_scaled)
    ev = explainer.expected_value
    # XGBoost binary may return list [neg, pos]; take positive class
    if isinstance(sv, list):
        sv = sv[1]
        ev = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
    return sv, float(ev)


def top_features(shap_values: np.ndarray, feature_names: list, n: int = 8) -> list:
    """Return [(feature_name, shap_value), ...] sorted by |shap| descending."""
    row = shap_values[0] if shap_values.ndim > 1 else shap_values
    pairs = sorted(zip(feature_names, row.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:n]


def waterfall_figure(shap_values: np.ndarray, expected_value: float,
                     feature_names: list, display_values: np.ndarray,
                     max_display: int = 10) -> plt.Figure:
    """Return a matplotlib Figure with a SHAP waterfall plot."""
    row = shap_values[0] if shap_values.ndim > 1 else shap_values
    explanation = shap.Explanation(
        values=row,
        base_values=expected_value,
        data=display_values,
        feature_names=feature_names,
    )
    fig, _ = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    return fig
