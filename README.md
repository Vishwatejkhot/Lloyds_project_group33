# Lloyds SME Intelligence Platform
**Group 33 · Dissertation Project · Companies House ML + Generative AI**

> See [SETUP.md](SETUP.md) for full setup and run instructions.

---

## What This Project Does

Using machine learning trained on **3,145,434 UK companies** from Companies House, this platform identifies SME prospects for Lloyds Banking Group across three signals:

| Signal | Best Model | F1 Score | AUC |
|--------|-----------|----------|-----|
| 🌱 Growth Opportunity | XGBoost | 0.9341 | — |
| ⚠️ Risk Signal (14:1 imbalance) | XGBoost | 0.9973 | 1.0000 |
| 💷 Lending Need Proxy | XGBoost | 1.0000 | 1.0000 |

Statistical validation includes McNemar's test, 1000-iteration Bootstrap CI, and Bayesian model comparison.
SHAP explainability (Pratik) identifies key features driving each prediction.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Vishwatejkhot/Lloyds_project_group33.git
cd Lloyds_project_group33/lloyds_project

# 2. Set up environment (Windows — creates .venv, installs all packages)
setup.bat

# 3. Add API keys to dashboard/.env
#    OPENAI_API_KEY=sk-...
#    GROQ_API_KEY=gsk_...

# 4. Run the dashboard
.venv\Scripts\activate
streamlit run dashboard\app.py
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | Platform overview, model summary, navigation guide |
| **Predictor** | Browse 50k SMEs or enter a new company for predictions + SHAP + AI report |
| **Performance** | ROC/PR curves, confusion matrices, McNemar's / Bootstrap / Bayesian tests, SHAP plots |
| **AI Agent** | ReAct agentic chat with tool calling — powered by Groq + OpenAI routing |

### Predict New Company
The **Predictor** page has a **"Predict New Company"** tab where you can enter any company's features manually. The form auto-generates input fields for the most important features (by XGBoost importance). Fill in what you know — anything left blank defaults to 0. Hit **Predict** to get:
- Growth / Risk / Lending probability scores
- SHAP waterfall plots for all 3 labels
- AI quick explanation (Groq) and full BI report (OpenAI)

---

## LLM Gateway

The dashboard uses **LiteLLM** to route between two AI providers automatically:

| Task | Primary | Fallback |
|------|---------|---------|
| Quick Explain | Groq `openai/gpt-oss-120b` | OpenAI GPT-4o-mini |
| Full Report | OpenAI GPT-4o-mini | Groq `openai/gpt-oss-120b` |
| AI Agent chat | Groq `openai/gpt-oss-120b` | OpenAI GPT-4o-mini |

Works with just one API key — falls back automatically if one provider fails.

---

## Data Pipeline (original scripts)

```bash
python companies_house/fetch_bulk.py     # Step 1: Download Companies House bulk data
python companies_house/filter_sectors.py # Step 2: Filter for BCB target sectors
python gdelt/fetch_gdelt.py              # Step 3: Fetch GDELT media data
python news/fetch_opencorporates.py      # Step 4: OpenCorporates data
python utils/merge_all.py               # Step 5: Merge into one dataset
```

Output lands in `/output/` as CSV files.

---

## Data Sources

| Source | Auth | What You Get |
|--------|------|-------------|
| Companies House Bulk Download | None | All UK companies, SIC codes, filing history |
| GDELT | None | Global news events, sentiment |
| OpenCorporates | None (free tier) | Cross-border company info |

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already done for `main`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch `main`, and set **Main file path** to `dashboard/app.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   OPENAI_API_KEY = "sk-..."
   GROQ_API_KEY = "gsk_..."
   ```
5. Deploy. The Predictor page's "Browse Dataset" tab uses the committed
   `output/feature_engineered_sample.csv` (50k-row sample) since the full
   3.1M-row dataset is too large for GitHub — see `.gitignore`.

---

## Project Structure

```
lloyds_project/
├── dashboard/          # Streamlit dashboard
│   ├── app.py
│   ├── pages/          # Predictor, Performance, AI Agent
│   ├── agents/         # LLM gateway + ReAct agent
│   └── utils/          # Model loader + SHAP utils
├── models/             # Trained .joblib model files
├── output/             # Evaluation outputs, CSVs, plots
├── requirements.txt    # All Python dependencies
├── setup.bat           # One-click Windows setup
└── SETUP.md            # Detailed setup guide
```
