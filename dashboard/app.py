import streamlit as st

st.set_page_config(
    page_title="Lloyds Prospect Dashboard",
    layout="wide"
)

st.title("Lloyds Project Group 33 Dashboard")

st.write(
    "This dashboard will display exploratory analysis, model predictions, "
    "ranked prospects, and explainable AI results."
)

st.subheader("Planned Dashboard Sections")

st.markdown("""
- Overview
- Exploratory Data Analysis
- Feature Engineering Summary
- Model Performance
- SHAP Explainability
- Ranked Prospect List
- Company Details
""")

st.info("Dashboard setup completed. Data integration will be added once model outputs are available.")
