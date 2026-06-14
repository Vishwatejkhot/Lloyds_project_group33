import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

INPUTS = {
    "base":       os.path.join(OUTPUT_DIR, "companies_bcb_filtered.csv"),
    "filings":    os.path.join(OUTPUT_DIR, "filings_and_directors.csv"),
    "gdelt_co":   os.path.join(OUTPUT_DIR, "gdelt_sentiment.csv"),
    "gdelt_sec":  os.path.join(OUTPUT_DIR, "gdelt_sector_trends.csv"),
    "oc":         os.path.join(OUTPUT_DIR, "opencorporates_enriched.csv"),
}

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_model_dataset.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "final_model_dataset_summary.txt")


def load_if_exists(path: str, name: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        print(f"  Loaded {name}: {df.shape}")
        return df
    else:
        print(f"  WARNING: {name} not found at {path} — skipping.")
        return None


def create_opportunity_labels(df: pd.DataFrame) -> pd.DataFrame:
    avg_tone = df.get("avg_tone", pd.Series([np.nan] * len(df), index=df.index))
    has_sentiment = avg_tone.notna()

    # GDELT covers only a small sampled subset of companies (small private UK
    # firms rarely appear in global news), so avg_tone is null for the vast
    # majority of rows. Build labels from structured signals that are populated
    # for the whole population, and use sentiment only as a bonus override
    # where it happens to be available.
    df["label_growth_opportunity"] = (
        (df.get("company_age_years", pd.Series([np.nan] * len(df), index=df.index)) < 10) &
        (df.get("is_active", pd.Series([0] * len(df), index=df.index)) == 1) &
        (df.get("is_dormant", pd.Series([1] * len(df), index=df.index)) == 0) &
        (df.get("accounts_overdue_days", pd.Series([999] * len(df), index=df.index)) == 0)
    ).astype(int)
    df.loc[has_sentiment & (avg_tone > 0), "label_growth_opportunity"] = 1

    df["label_risk_signal"] = (
        (df.get("accounts_ever_late", pd.Series([0] * len(df), index=df.index)).fillna(0) == 1) |
        (df.get("conf_stmt_overdue_days", pd.Series([0] * len(df), index=df.index)).fillna(0) > 180) |
        (df.get("has_late_filing_history", pd.Series([0] * len(df), index=df.index)).fillna(0) == 1) |
        (df.get("director_turnover_signal", pd.Series([0] * len(df), index=df.index)).fillna(0) == 1)
    ).astype(int)
    df.loc[has_sentiment & (avg_tone < -3), "label_risk_signal"] = 1

    df["label_lending_need_proxy"] = (
        (df.get("has_active_mortgages", pd.Series([0] * len(df), index=df.index)).fillna(0) == 1) &
        (df.get("is_active", pd.Series([0] * len(df), index=df.index)) == 1) &
        (df.get("label_risk_signal", pd.Series([0] * len(df), index=df.index)) == 0)
    ).astype(int)

    return df


def print_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("FINAL MODEL DATASET SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total companies: {len(df):,}")
    lines.append(f"Total features:  {len(df.columns):,}")
    lines.append("")

    lines.append("--- Sector Distribution ---")
    if "bcb_sector" in df.columns:
        for sector, count in df["bcb_sector"].value_counts().items():
            lines.append(f"  {sector}: {count:,} ({count/len(df)*100:.1f}%)")

    lines.append("")
    lines.append("--- Proxy Label Distribution ---")
    for label in ["label_growth_opportunity", "label_risk_signal", "label_lending_need_proxy"]:
        if label in df.columns:
            pos = df[label].sum()
            lines.append(f"  {label}: {pos:,} positive ({pos/len(df)*100:.1f}%)")

    lines.append("")
    lines.append("--- Data Sources Merged ---")
    lines.append("  [1] Companies House bulk data (structured financials, SIC, status)")
    lines.append("  [2] Companies House filings API (late filings, director changes)")
    lines.append("  [3] GDELT company-level sentiment (article tone, volume)")
    lines.append("  [4] GDELT sector trends (sector-wide sentiment trends)")
    lines.append("  [5] OpenCorporates (cross-border, subsidiaries)")

    lines.append("")
    lines.append("--- Feature Groups ---")
    lines.append("  COMPANY STRUCTURE: age, type, status, dormancy")
    lines.append("  FILING BEHAVIOUR:  overdue days, late rate, history")
    lines.append("  DIRECTORS:         turnover, appointments, resignations")
    lines.append("  MORTGAGES:         outstanding, total, satisfied")
    lines.append("  MEDIA SENTIMENT:   avg tone, % negative, % positive, trend")
    lines.append("  SECTOR CONTEXT:    sector-level sentiment & volume trends")
    lines.append("  INTERNATIONAL:     has parent, has subsidiaries, OC type")

    lines.append("")
    lines.append("--- Missing Data by Column ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    for col in df.columns:
        if missing_pct[col] > 20:
            lines.append(f"  WARNING: {col} — {missing_pct[col]}% missing")

    text = "\n".join(lines)
    print(text)
    return text


def main():
    print("Loading data sources...")
    base     = load_if_exists(INPUTS["base"],      "Companies House (base)")
    filings  = load_if_exists(INPUTS["filings"],   "Filing history & directors")
    gdelt_co = load_if_exists(INPUTS["gdelt_co"],  "GDELT company sentiment")
    gdelt_sec= load_if_exists(INPUTS["gdelt_sec"], "GDELT sector trends")
    oc       = load_if_exists(INPUTS["oc"],        "OpenCorporates")

    if base is None:
        print("ERROR: Base Companies House data is required. Run filter_sectors.py first.")
        return

    base["CompanyNumber"] = base["CompanyNumber"].astype(str).str.zfill(8)
    df = base.copy()

    if filings is not None:
        filings["CompanyNumber"] = filings["CompanyNumber"].astype(str).str.zfill(8)
        drop_cols = [c for c in filings.columns if c in df.columns and c != "CompanyNumber"]
        filings = filings.drop(columns=drop_cols)
        df = df.merge(filings, on="CompanyNumber", how="left")
        print(f"After merging filings: {df.shape}")

    if gdelt_co is not None:
        gdelt_co["CompanyNumber"] = gdelt_co["CompanyNumber"].astype(str).str.zfill(8)
        gdelt_cols = [c for c in gdelt_co.columns if c not in df.columns or c == "CompanyNumber"]
        df = df.merge(gdelt_co[gdelt_cols], on="CompanyNumber", how="left")
        print(f"After merging GDELT company sentiment: {df.shape}")

    if gdelt_sec is not None:
        gdelt_sec = gdelt_sec.rename(columns={
            "sector": "bcb_sector",
            "avg_tone": "sector_avg_tone",
            "article_count": "sector_article_count",
            "volume_trend": "sector_volume_trend",
            "tone_trend": "sector_tone_trend",
            "pct_negative": "sector_pct_negative",
        })
        sector_cols = ["bcb_sector", "sector_avg_tone", "sector_article_count",
                       "sector_volume_trend", "sector_tone_trend", "sector_pct_negative"]
        sector_cols = [c for c in sector_cols if c in gdelt_sec.columns]
        df = df.merge(gdelt_sec[sector_cols], on="bcb_sector", how="left")
        print(f"After merging GDELT sector trends: {df.shape}")

    if oc is not None:
        oc["CompanyNumber"] = oc["CompanyNumber"].astype(str).str.zfill(8)
        oc_cols = [c for c in oc.columns if c not in df.columns or c == "CompanyNumber"]
        df = df.merge(oc[oc_cols], on="CompanyNumber", how="left")
        print(f"After merging OpenCorporates: {df.shape}")

    df = create_opportunity_labels(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFinal dataset saved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")

    summary_text = print_summary(df)
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary_text)
    print(f"\nSummary saved: {SUMMARY_PATH}")
    print("\n✓ Data collection complete! Your dataset is ready for modelling.")
    print(f"  Load it with: df = pd.read_csv('{OUTPUT_PATH}')")


if __name__ == "__main__":
    main()
