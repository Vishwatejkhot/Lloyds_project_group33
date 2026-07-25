"""
Company Predictor — browse dataset, predict new company, SHAP waterfall, AI explanation, full report.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.model_loader import load_all_models, best_model
from utils.shap_utils import compute_shap, top_features, waterfall_figure
from agents.report_agent import quick_explain, generate_full_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "feature_engineered_dataset.csv")

st.set_page_config(page_title="Predictor · Lloyds SME", page_icon="🔮", layout="wide")
st.title("🔮 Company Predictor")
st.caption("Browse 50k sampled UK SMEs · Predict new company · SHAP explainability · AI report")


# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading company dataset…")
def load_data():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH, low_memory=False, nrows=500_000)
    df = df.sample(n=min(50_000, len(df)), random_state=42).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Scoring companies…")
def score_sample(_df, _models):
    rows = []
    for label in ["growth", "risk", "lending"]:
        model, scaler, features, _ = best_model(_models, label)
        X = _df[features].fillna(0)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        for i, p in enumerate(probs):
            rows.append({"idx": i, "label": label, "prob": round(float(p), 4)})
    wide = pd.DataFrame(rows).pivot(index="idx", columns="label", values="prob").reset_index()
    wide.columns.name = None
    return wide


def prob_color(p):
    if p >= 0.7: return "🟢"
    if p >= 0.4: return "🟡"
    return "🔴"


def get_top_input_features(models, label, n=12):
    """Return top N feature names by model importance for the input form."""
    model, _, features, _ = best_model(models, label)
    try:
        importances = model.feature_importances_
        pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return [f for f, _ in pairs[:n]]
    except Exception:
        return features[:n]


def build_input_row(models, label, user_vals):
    """Build a single-row DataFrame matching the model's feature list, filling unknowns with 0."""
    _, _, features, _ = best_model(models, label)
    row = {f: user_vals.get(f, 0.0) for f in features}
    return pd.DataFrame([row])


def show_prediction_block(models, label, user_vals, sector):
    """Score + SHAP + AI explain for one label from user input."""
    model, scaler, features, mname = best_model(models, label)
    X_df    = build_input_row(models, label, user_vals)
    X_orig  = X_df.values
    X_scaled = scaler.transform(X_orig)
    prob    = float(model.predict_proba(X_scaled)[0, 1])
    sv, ev  = compute_shap(model, X_scaled)
    top     = top_features(sv, features)
    return prob, sv, ev, features, top, mname, X_orig[0]


# ── Load models once ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    models = load_all_models()
    if not models:
        st.error("No models found in models/ folder.")
        st.stop()


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_browse, tab_new = st.tabs(["📋 Browse Dataset", "🆕 Predict New Company"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Browse dataset
# ══════════════════════════════════════════════════════════════════════════
with tab_browse:
    df = load_data()
    if df is None:
        st.warning("feature_engineered_dataset.csv not found locally. Ask Vishwatej for the file.")
        st.stop()

    # ── Sidebar filters ────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sectors = ["All"] + sorted(df["bcb_sector"].dropna().unique().tolist())
        sel_sector = st.selectbox("BCB Sector", sectors)
    with col_f2:
        sort_by = st.selectbox("Sort by", ["growth", "risk", "lending"])
    with col_f3:
        n_show = st.slider("Companies to show", 10, 200, 50)

    # ── Score & filter ─────────────────────────────────────────────────────
    scores = score_sample(df, models)
    df_scored = df.copy()
    df_scored["growth_prob"]  = scores["growth"].values
    df_scored["risk_prob"]    = scores["risk"].values
    df_scored["lending_prob"] = scores["lending"].values

    if sel_sector != "All":
        df_scored = df_scored[df_scored["bcb_sector"] == sel_sector]

    df_display = (
        df_scored
        .sort_values(f"{sort_by}_prob", ascending=False)
        .head(n_show)
        .reset_index(drop=True)
    )

    # ── Company table ──────────────────────────────────────────────────────
    st.subheader(f"Top {n_show} companies by {sort_by} probability")
    show_cols = [c for c in ["bcb_sector","sic_code_1","company_size_score",
                              "has_mortgage_history","num_mortgages_total",
                              "growth_prob","risk_prob","lending_prob"] if c in df_display.columns]
    st.dataframe(
        df_display[show_cols].style.background_gradient(
            subset=["growth_prob","risk_prob","lending_prob"], cmap="RdYlGn"
        ),
        width="stretch", height=300,
    )

    # ── Select a company ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Select a company for detailed analysis")
    row_idx = st.number_input("Row index (from table above)", min_value=0,
                               max_value=len(df_display) - 1, value=0, step=1)
    company = df_display.iloc[int(row_idx)]
    sector  = str(company.get("bcb_sector", "Unknown"))

    c1, c2, c3 = st.columns(3)
    g, r, l = company["growth_prob"], company["risk_prob"], company["lending_prob"]
    c1.metric(f"{prob_color(g)} Growth Opportunity", f"{g:.1%}")
    c2.metric(f"{prob_color(r)} Risk Signal",        f"{r:.1%}")
    c3.metric(f"{prob_color(l)} Lending Need",       f"{l:.1%}")
    st.caption(f"Sector: **{sector}** · SIC: {company.get('sic_code_1','?')} · "
               f"Size score: {company.get('company_size_score','?')}")

    # ── SHAP ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("SHAP Explainability")
    shap_label = st.selectbox("Explain label", ["growth","risk","lending"], key="browse_shap")

    with st.spinner("Computing SHAP values…"):
        model, scaler, features, mname = best_model(models, shap_label)
        X_orig   = company[features].fillna(0).values.reshape(1, -1)
        X_scaled = scaler.transform(X_orig)
        sv, ev   = compute_shap(model, X_scaled)
        top_feat = top_features(sv, features)

    col_shap, col_info = st.columns([3, 1])
    with col_shap:
        fig = waterfall_figure(sv, ev, features, X_orig[0])
        st.pyplot(fig, width="stretch")
        plt.close("all")
    with col_info:
        st.markdown(f"**Model:** {mname}")
        st.markdown("**Top drivers:**")
        for fname, fval in top_feat[:6]:
            arrow = "↑" if fval > 0 else "↓"
            st.markdown(f"- {arrow} `{fname}` ({fval:+.3f})")

    # ── AI Explanation ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("AI Explanation")
    if st.button("⚡ Quick Explain (Groq)", key="browse_explain"):
        with st.spinner("Generating explanation via Groq…"):
            prob_val = float(company[f"{shap_label}_prob"])
            try:
                text, model_used = quick_explain(shap_label, prob_val, top_feat)
                st.success(text)
                st.markdown(f'<span class="llm-tag">via {model_used}</span>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"LLM error: {e}")

    # ── Full Report ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Full BI Report")
    if st.button("📄 Generate Full Report (OpenAI)", key="browse_report"):
        with st.spinner("Generating report…"):
            preds    = {"growth": float(g), "risk": float(r), "lending": float(l)}
            shap_all = {}
            for lbl in ["growth","risk","lending"]:
                m2, s2, f2, _ = best_model(models, lbl)
                X2 = company[f2].fillna(0).values.reshape(1, -1)
                sv2, ev2 = compute_shap(m2, s2.transform(X2))
                shap_all[lbl] = top_features(sv2, f2, n=5)
            try:
                report, model_used = generate_full_report(preds, shap_all, sector)
                st.markdown(report)
                st.markdown(f'<span class="llm-tag">via {model_used}</span>', unsafe_allow_html=True)
                st.session_state["last_report_context"] = {
                    "sector": sector, "growth": float(g),
                    "risk": float(r), "lending": float(l),
                    "top_features": {k: [(f, v) for f, v in vals] for k, vals in shap_all.items()},
                }
            except Exception as e:
                st.error(f"LLM error: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Predict new company
# ══════════════════════════════════════════════════════════════════════════
with tab_new:
    st.subheader("Enter company details to get predictions")
    st.caption(
        "Fill in the fields below. Top features by model importance are shown. "
        "Any feature not shown defaults to 0."
    )

    # Collect top features across all 3 labels (union, no duplicates)
    all_top = []
    seen = set()
    for lbl in ["growth", "risk", "lending"]:
        for f in get_top_input_features(models, lbl, n=10):
            if f not in seen:
                all_top.append(f)
                seen.add(f)

    # ── Input form ────────────────────────────────────────────────────────
    with st.form("new_company_form"):
        st.markdown("**Company Features**")

        user_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(all_top):
            col = cols[i % 3]
            # Heuristic: binary features get a selectbox, others get a number input
            is_binary = any(kw in feat.lower() for kw in
                            ["has_", "is_", "flag", "overdue", "active", "dissolved",
                             "accounts_late", "liquidation", "dormant"])
            if is_binary:
                user_vals[feat] = col.selectbox(feat, [0, 1], key=f"inp_{feat}")
            else:
                user_vals[feat] = col.number_input(
                    feat, value=0.0, step=0.1, format="%.2f", key=f"inp_{feat}"
                )

        new_sector = st.text_input("Sector (for report, optional)", value="Unknown")
        submitted = st.form_submit_button("🔮 Predict", type="primary")

    # ── Results ───────────────────────────────────────────────────────────
    if submitted:
        results = {}
        shap_all = {}

        with st.spinner("Running predictions…"):
            for lbl in ["growth", "risk", "lending"]:
                prob, sv, ev, feats, top, mname, X_orig_row = show_prediction_block(
                    models, lbl, user_vals, new_sector
                )
                results[lbl]  = {"prob": prob, "sv": sv, "ev": ev,
                                  "feats": feats, "top": top, "mname": mname,
                                  "X_orig": X_orig_row}
                shap_all[lbl] = top

        # Metric row
        g = results["growth"]["prob"]
        r = results["risk"]["prob"]
        l = results["lending"]["prob"]

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{prob_color(g)} Growth Opportunity", f"{g:.1%}")
        c2.metric(f"{prob_color(r)} Risk Signal",        f"{r:.1%}")
        c3.metric(f"{prob_color(l)} Lending Need",       f"{l:.1%}")

        # SHAP waterfall for each label
        st.markdown("---")
        st.subheader("SHAP Explanations")
        for lbl in ["growth", "risk", "lending"]:
            with st.expander(f"{lbl.capitalize()} — SHAP waterfall", expanded=(lbl == "growth")):
                col_w, col_t = st.columns([3, 1])
                with col_w:
                    fig = waterfall_figure(
                        results[lbl]["sv"], results[lbl]["ev"],
                        results[lbl]["feats"], results[lbl]["X_orig"]
                    )
                    st.pyplot(fig, width="stretch")
                    plt.close("all")
                with col_t:
                    st.markdown(f"**Model:** {results[lbl]['mname']}")
                    st.markdown("**Top drivers:**")
                    for fname, fval in results[lbl]["top"][:6]:
                        arrow = "↑" if fval > 0 else "↓"
                        st.markdown(f"- {arrow} `{fname}` ({fval:+.3f})")

        # AI buttons
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚡ Quick Explain (Groq)", key="new_explain"):
                with st.spinner("Generating explanation…"):
                    try:
                        text, model_used = quick_explain("growth", g, shap_all["growth"])
                        st.success(text)
                        st.markdown(f'<span class="llm-tag">via {model_used}</span>',
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
        with col_b:
            if st.button("📄 Full BI Report (OpenAI)", key="new_report"):
                with st.spinner("Generating report…"):
                    try:
                        report, model_used = generate_full_report(
                            {"growth": g, "risk": r, "lending": l},
                            {k: v["top"][:5] for k, v in results.items()},
                            new_sector
                        )
                        st.markdown(report)
                        st.markdown(f'<span class="llm-tag">via {model_used}</span>',
                                    unsafe_allow_html=True)
                        st.session_state["last_report_context"] = {
                            "sector": new_sector, "growth": g,
                            "risk": r, "lending": l,
                            "top_features": {k: [(f, v) for f, v in vals["top"]]
                                              for k, vals in results.items()},
                        }
                    except Exception as e:
                        st.error(f"LLM error: {e}")
