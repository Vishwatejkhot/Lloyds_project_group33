# Lloyds SME Intelligence Dashboard — Setup Guide
**Group 33** · Companies House ML · XGBoost + SHAP + Generative AI

---

## Prerequisites

- **Python 3.10 or higher** — download from [python.org](https://www.python.org/downloads/)
- **Git** — to clone the repository
- **API Keys** — one or both of the following:
  - [Groq API key](https://console.groq.com) (free tier available) — used for quick explanations
  - [OpenAI API key](https://platform.openai.com/api-keys) — used for full report generation

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Vishwatejkhot/Lloyds_project_group33.git
cd Lloyds_project_group33/lloyds_project
```

---

## Step 2 — Run the Setup Script (Windows)

Double-click `setup.bat` **or** run it from the terminal:

```
setup.bat
```

This will automatically:
- Create a Python virtual environment (`.venv`)
- Install all required packages from `requirements.txt`
- Copy `.env.example` → `dashboard/.env` (if not already present)

> **Takes ~2–5 minutes** on first run due to package downloads.

---

## Step 3 — Add Your API Keys

Open `dashboard/.env` in any text editor and fill in your keys:

```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

> The dashboard still runs if only one key is present — the LLM gateway automatically routes to whichever provider is configured.

---

## Step 4 — Required Data Files

The following large files are **not in Git** (gitignored). Place them in the `output/` folder:

| File | Description | Where to get it |
|------|-------------|-----------------|
| `feature_engineered_dataset.csv` | Full feature-engineered dataset (3.1M rows) | Ask Vishwatej / Bhavika |
| `predictions.csv` | Model predictions (~227 MB) | Ask Vishwatej |

The `models/` folder (`.joblib` files) is also required — ask Vishwatej if missing.

---

## Step 5 — Run the Dashboard

```bash
# Activate the virtual environment
.venv\Scripts\activate

# Launch the dashboard
streamlit run dashboard\app.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## Dashboard Pages

| Page | What it does |
|------|-------------|
| **Home** | Overview of the platform and model performance summary |
| **Predictor** | Browse 50k SMEs, view ML scores, SHAP waterfall, AI explanation |
| **Performance** | ROC/PR curves, confusion matrices, McNemar's test, Bootstrap CI, Bayesian comparison |
| **AI Agent** | Chat with a ReAct agent powered by Groq + OpenAI routing |

---

## Troubleshooting

**`ModuleNotFoundError`** — packages not installed. Re-run `setup.bat`.

**`feature_engineered_dataset.csv not found`** — the Predictor page won't load without this file. See Step 4.

**LLM errors / API key errors** — check `dashboard/.env` has valid keys. The dashboard still works for ML predictions and SHAP without any API keys.

**Port already in use** — run on a different port:
```bash
streamlit run dashboard\app.py --server.port 8502
```

**Wrong Python version** — check with `python --version`. Must be 3.10+.

---

## Project Structure

```
lloyds_project/
├── dashboard/
│   ├── app.py                  # Home page
│   ├── pages/
│   │   ├── 1_Predictor.py      # ML predictions + SHAP + AI report
│   │   ├── 2_Performance.py    # Model evaluation metrics
│   │   └── 3_AI_Agent.py       # Agentic AI chat interface
│   ├── agents/
│   │   ├── gateway.py          # LLM routing (Groq / OpenAI)
│   │   └── report_agent.py     # ReAct agent with tool calling
│   ├── utils/
│   │   ├── model_loader.py     # Load .joblib models
│   │   └── shap_utils.py       # SHAP computation + waterfall plots
│   └── .env                    # Your API keys (not in Git)
├── output/                     # Evaluation outputs + CSVs + plots
├── models/                     # Trained .joblib model files
├── requirements.txt            # All Python dependencies
└── setup.bat                   # One-click environment setup (Windows)
```

---

## LLM Gateway — How It Works

The dashboard uses **LiteLLM** to route between two AI providers automatically:

| Task | Primary | Fallback |
|------|---------|---------|
| Quick Explain | Groq (`openai/gpt-oss-120b`) | OpenAI GPT-4o-mini |
| Full Report | OpenAI GPT-4o-mini | Groq (`openai/gpt-oss-120b`) |
| AI Agent chat | Groq (`openai/gpt-oss-120b`) | OpenAI GPT-4o-mini |

If one provider fails or has no key, it automatically falls back to the other.
