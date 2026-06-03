# Lloyds BCB Data Collector
## Dissertation Project — No API Keys Required

This toolkit collects data for all 8 Lloyds BCB target sectors using
**free, no-auth** data sources only.

---

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
# Step 1: Download Companies House bulk data (no key needed)
python companies_house/fetch_bulk.py

# Step 2: Filter for BCB target sectors
python companies_house/filter_sectors.py

# Step 3: Fetch GDELT media data (no key needed)
python gdelt/fetch_gdelt.py

# Step 4: Fetch OpenCorporates data (no key needed)
python news/fetch_opencorporates.py

# Step 5: Merge everything into one dataset
python utils/merge_all.py
```

## Output
All final data lands in `/output/` as CSV files ready for modelling.

---

## Data Sources Used (All Free, No API Key)
| Source | Auth Needed | What You Get |
|---|---|---|
| Companies House Bulk Download | None | All UK companies, financials, SIC codes |
| GDELT | None | Global news events, sentiment, tone |
| OpenCorporates | None (free tier) | Cross-border company info |
| Companies House Filing API | None | Director changes, filing history |
