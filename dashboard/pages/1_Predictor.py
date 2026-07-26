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
from utils.pdf_generator import pdf_download_button
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


@st.cache_data(show_spinner=False)
def get_feature_medians():
    """Median of every numeric column — used as default values in the new-company form."""
    if not os.path.exists(DATASET_PATH):
        return {}
    df = pd.read_csv(DATASET_PATH, low_memory=False, nrows=100_000)
    return df.select_dtypes(include="number").median().to_dict()


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
    model, _, features, _ = best_model(models, label)
    try:
        importances = model.feature_importances_
        pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return [f for f, _ in pairs[:n]]
    except Exception:
        return features[:n]


def predict_prob(models, label, user_vals, medians: dict):
    """Run predict_proba only — lightweight, no SHAP. Unseen features get dataset median."""
    model, scaler, features, mname = best_model(models, label)
    row = {f: user_vals.get(f, medians.get(f, 0.0)) for f in features}
    X = pd.DataFrame([row]).values
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    return prob, model, scaler, features, mname, X[0]


def render_shap_block(model, scaler, features, mname, X_orig):
    """Compute and render SHAP waterfall + top features. Called lazily."""
    X_scaled = scaler.transform(X_orig.reshape(1, -1))
    try:
        sv, ev = compute_shap(model, X_scaled)
        top = top_features(sv, features)
        col_w, col_t = st.columns([3, 1])
        with col_w:
            fig = waterfall_figure(sv, ev, features, X_orig)
            st.pyplot(fig, width="stretch")
            plt.close("all")
        with col_t:
            st.markdown(f"**Model:** {mname}")
            st.markdown("**Top drivers:**")
            for fname, fval in top[:6]:
                arrow = "↑" if fval > 0 else "↓"
                st.markdown(f"- {arrow} `{fname}` ({fval:+.3f})")
        return top
    except Exception as e:
        st.error(f"SHAP error: {e}")
        return []


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

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sectors = ["All"] + sorted(df["bcb_sector"].dropna().unique().tolist())
        sel_sector = st.selectbox("BCB Sector", sectors)
    with col_f2:
        sort_by = st.selectbox("Sort by", ["growth", "risk", "lending"])
    with col_f3:
        n_show = st.slider("Companies to show", 10, 200, 50)

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

    st.markdown("---")
    st.subheader("Full BI Report")
    if st.button("📄 Generate Full Report (OpenAI)", key="browse_report"):
        with st.spinner("Generating report…"):
            preds    = {"growth": float(g), "risk": float(r), "lending": float(l)}
            shap_all = {}
            for lbl in ["growth","risk","lending"]:
                m2, s2, f2, _ = best_model(models, lbl)
                X2 = company[f2].fillna(0).values.reshape(1, -1)
                sv2, _ = compute_shap(m2, s2.transform(X2))
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
                st.session_state["browse_report_text"] = report
                st.session_state["browse_report_meta"] = {
                    "sector": sector, "g": float(g), "r": float(r), "l": float(l),
                    "shap_all": shap_all,
                }
            except Exception as e:
                st.error(f"LLM error: {e}")

    if "browse_report_text" in st.session_state:
        m = st.session_state["browse_report_meta"]
        pdf_download_button(
            report_text=st.session_state["browse_report_text"],
            sector=m["sector"], growth=m["g"], risk=m["r"], lending=m["l"],
            top_features=m["shap_all"],
            key="browse_pdf_dl",
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Predict new company (predictions fast on submit; SHAP lazy per label)
# ══════════════════════════════════════════════════════════════════════════
with tab_new:
    st.subheader("Enter company details to get predictions")
    st.caption(
        "Fill in the fields below. Top features by model importance are shown. "
        "Any feature not shown defaults to 0. SHAP is computed on demand per label."
    )

    # Collect top features across all 3 labels (union, no duplicates)
    all_top = []
    seen = set()
    for lbl in ["growth", "risk", "lending"]:
        for f in get_top_input_features(models, lbl, n=10):
            if f not in seen:
                all_top.append(f)
                seen.add(f)

    medians = get_feature_medians()

    # ── Input form ────────────────────────────────────────────────────────
    with st.form("new_company_form"):
        user_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(all_top):
            col = cols[i % 3]
            is_binary = any(kw in feat.lower() for kw in
                            ["has_", "is_", "flag", "overdue", "active", "dissolved",
                             "accounts_late", "liquidation", "dormant"])
            default_val = float(medians.get(feat, 0.0))
            if is_binary:
                default_idx = min(int(round(default_val)), 1)
                user_vals[feat] = col.selectbox(feat, [0, 1], index=default_idx, key=f"inp_{feat}")
            else:
                user_vals[feat] = col.number_input(
                    feat, value=round(default_val, 3), step=0.1, format="%.3f", key=f"inp_{feat}"
                )
        new_sector = st.text_input("Sector (for report, optional)", value="Unknown")
        submitted = st.form_submit_button("🔮 Predict", type="primary")

    # On submit: run predict_proba only (fast) — store in session_state
    if submitted:
        with st.spinner("Running predictions…"):
            preds_out = {}
            try:
                for lbl in ["growth", "risk", "lending"]:
                    prob, mdl, scl, feats, mn, X_row = predict_prob(models, lbl, user_vals, medians)
                    preds_out[lbl] = {"prob": prob, "model": mdl, "scaler": scl,
                                      "features": feats, "mname": mn, "X_orig": X_row}
                st.session_state["new_preds"]    = preds_out
                st.session_state["new_sector"]   = new_sector
                st.session_state["new_user_vals"] = user_vals
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ── Results — shown from session_state so they persist across reruns ──
    if "new_preds" in st.session_state:
        preds_out  = st.session_state["new_preds"]
        new_sector = st.session_state["new_sector"]

        g = preds_out["growth"]["prob"]
        r = preds_out["risk"]["prob"]
        l = preds_out["lending"]["prob"]

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{prob_color(g)} Growth Opportunity", f"{g:.2%}")
        c2.metric(f"{prob_color(r)} Risk Signal",        f"{r:.2%}")
        c3.metric(f"{prob_color(l)} Lending Need",       f"{l:.2%}")

        # ── SHAP — lazy per label ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("SHAP Explanations (click a label to compute)")

        shap_results = {}
        for lbl in ["growth", "risk", "lending"]:
            p = preds_out[lbl]
            with st.expander(f"{lbl.capitalize()} — {p['prob']:.1%} probability"):
                if st.button(f"Compute SHAP for {lbl}", key=f"shap_btn_{lbl}"):
                    with st.spinner(f"Computing SHAP for {lbl}…"):
                        top = render_shap_block(
                            p["model"], p["scaler"], p["features"], p["mname"], p["X_orig"]
                        )
                        shap_results[lbl] = top
                        st.session_state[f"shap_top_{lbl}"] = top
                elif f"shap_top_{lbl}" in st.session_state:
                    top = render_shap_block(
                        p["model"], p["scaler"], p["features"], p["mname"], p["X_orig"]
                    )
                    shap_results[lbl] = top

        # ── AI buttons ─────────────────────────────────────────────────────
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚡ Quick Explain (Groq)", key="new_explain"):
                top_g = st.session_state.get("shap_top_growth", [])
                with st.spinner("Generating explanation…"):
                    try:
                        text, model_used = quick_explain("growth", g, top_g)
                        st.success(text)
                        st.markdown(f'<span class="llm-tag">via {model_used}</span>',
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
        with col_b:
            if st.button("📄 Full BI Report (OpenAI)", key="new_report"):
                shap_all = {lbl: st.session_state.get(f"shap_top_{lbl}", [])
                            for lbl in ["growth", "risk", "lending"]}
                with st.spinner("Generating report…"):
                    try:
                        report, model_used = generate_full_report(
                            {"growth": g, "risk": r, "lending": l},
                            shap_all, new_sector
                        )
                        st.markdown(report)
                        st.markdown(f'<span class="llm-tag">via {model_used}</span>',
                                    unsafe_allow_html=True)
                        st.session_state["last_report_context"] = {
                            "sector": new_sector, "growth": g, "risk": r, "lending": l,
                            "top_features": {k: [(f, v) for f, v in vals]
                                             for k, vals in shap_all.items()},
                        }
                        st.session_state["new_report_text"] = report
                        st.session_state["new_report_meta"] = {
                            "sector": new_sector, "g": g, "r": r, "l": l,
                            "shap_all": shap_all,
                        }
                    except Exception as e:
                        st.error(f"LLM error: {e}")

        if "new_report_text" in st.session_state:
            m = st.session_state["new_report_meta"]
            pdf_download_button(
                report_text=st.session_state["new_report_text"],
                sector=m["sector"], growth=m["g"], risk=m["r"], lending=m["l"],
                top_features=m["shap_all"],
                key="new_pdf_dl",
            )
