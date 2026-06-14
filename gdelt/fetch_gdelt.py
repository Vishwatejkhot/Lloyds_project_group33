import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "companies_bcb_filtered.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "gdelt_sentiment.csv")
SECTOR_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "gdelt_sector_trends.csv")

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_EVENT_API = "https://api.gdeltproject.org/api/v2/geo/geo"

DELAY = 1.5
SAMPLE_PER_SECTOR = 50

SECTOR_QUERIES = {
    "Manufacturing": "UK manufacturing sector production output",
    "Public_Sector_Education_Charities": "UK public sector education budget funding",
    "Healthcare": "UK healthcare NHS funding investment",
    "Technology_Legal_Professional": "UK technology startup investment fintech",
    "Agriculture": "UK agriculture farming food supply",
    "Real_Estate": "UK real estate property market commercial",
    "Wholesale_Retail": "UK retail wholesale consumer spending",
    "Fast_Growth_Emerging": "UK startup scale-up growth emerging technology",
}


def query_gdelt_doc(query: str, timespan: str = "1y", max_records: int = 100) -> dict:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "timespan": timespan,
        "sort": "DateDesc",
        "format": "json",
    }
    try:
        r = requests.get(GDELT_DOC_API, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            return {}
    except Exception as e:
        return {}


def query_gdelt_timeline(query: str, timespan: str = "1y") -> dict:
    params = {
        "query": query,
        "mode": "timelinetone",
        "timespan": timespan,
        "format": "json",
    }
    try:
        r = requests.get(GDELT_DOC_API, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception as e:
        return {}


def parse_articles(articles_data: dict) -> dict:
    articles = articles_data.get("articles", [])
    if not articles:
        return {
            "article_count": 0,
            "avg_tone": np.nan,
            "avg_positive_score": np.nan,
            "avg_negative_score": np.nan,
            "tone_std": np.nan,
            "pct_negative": np.nan,
            "pct_positive": np.nan,
        }

    tones = []
    pos_scores = []
    neg_scores = []

    for art in articles:
        tone = art.get("tone", None)
        if tone is not None:
            try:
                tone = float(tone)
                tones.append(tone)
                pos_scores.append(max(0, tone))
                neg_scores.append(max(0, -tone))
            except ValueError:
                pass

    if not tones:
        return {"article_count": len(articles), "avg_tone": np.nan, "avg_positive_score": np.nan,
                "avg_negative_score": np.nan, "tone_std": np.nan, "pct_negative": np.nan, "pct_positive": np.nan}

    tones_arr = np.array(tones)
    return {
        "article_count": len(articles),
        "avg_tone": round(np.mean(tones_arr), 4),
        "avg_positive_score": round(np.mean(pos_scores), 4),
        "avg_negative_score": round(np.mean(neg_scores), 4),
        "tone_std": round(np.std(tones_arr), 4),
        "pct_negative": round(np.mean(tones_arr < -2) * 100, 2),
        "pct_positive": round(np.mean(tones_arr > 2) * 100, 2),
    }


def parse_timeline(timeline_data: dict) -> dict:
    timeline = timeline_data.get("timeline", [])
    if not timeline or len(timeline) < 2:
        return {
            "timeline_points": 0,
            "volume_trend": np.nan,
            "tone_trend": np.nan,
            "recent_avg_tone": np.nan,
            "early_avg_tone": np.nan,
        }

    try:
        series = timeline[0].get("data", []) if isinstance(timeline[0], dict) else timeline

        values = []
        for point in series:
            val = point.get("value", point.get("Value", None))
            if val is not None:
                try:
                    values.append(float(val))
                except ValueError:
                    pass

        if len(values) < 4:
            return {"timeline_points": len(values), "volume_trend": np.nan,
                    "tone_trend": np.nan, "recent_avg_tone": np.nan, "early_avg_tone": np.nan}

        mid = len(values) // 2
        early = np.mean(values[:mid])
        recent = np.mean(values[mid:])
        trend = recent - early

        return {
            "timeline_points": len(values),
            "volume_trend": round(trend, 4),
            "tone_trend": round(trend, 4),
            "recent_avg_tone": round(recent, 4),
            "early_avg_tone": round(early, 4),
        }
    except Exception:
        return {"timeline_points": 0, "volume_trend": np.nan,
                "tone_trend": np.nan, "recent_avg_tone": np.nan, "early_avg_tone": np.nan}


def fetch_sector_trends():
    print("\n=== Fetching GDELT Sector-Level Trends ===")
    records = []

    for sector, query in tqdm(SECTOR_QUERIES.items(), desc="Sectors"):
        articles_data = query_gdelt_doc(query, timespan="1y", max_records=250)
        article_features = parse_articles(articles_data)

        timeline_data = query_gdelt_timeline(query, timespan="1y")
        timeline_features = parse_timeline(timeline_data)

        record = {
            "sector": sector,
            "gdelt_query": query,
            **article_features,
            **timeline_features,
        }
        records.append(record)
        print(f"  {sector}: {article_features['article_count']} articles, "
              f"avg tone: {article_features['avg_tone']}")

        time.sleep(DELAY)

    df = pd.DataFrame(records)
    df.to_csv(SECTOR_OUTPUT_PATH, index=False)
    print(f"\nSector trends saved: {SECTOR_OUTPUT_PATH}")
    return df


def fetch_company_level_sentiment():
    if not os.path.exists(INPUT_PATH):
        print(f"WARNING: {INPUT_PATH} not found. Skipping company-level GDELT.")
        print("Run companies_house/filter_sectors.py first.")
        return None

    print("\n=== Fetching GDELT Company-Level Sentiment ===")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    sample = (
        df[df["is_active"] == 1]
        .groupby("bcb_sector", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLE_PER_SECTOR), random_state=42))
    )
    print(f"Processing {len(sample):,} companies across all BCB sectors.")

    name_col = "CompanyName" if "CompanyName" in sample.columns else " CompanyName"
    sample["clean_name"] = (
        sample[name_col]
        .astype(str)
        .str.replace(r"\bLTD\b|\bLIMITED\b|\bPLC\b|\bLLP\b", "", regex=True)
        .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        .str.strip()
        .str.lower()
    )

    results = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Company GDELT"):
        name = row["clean_name"]
        if not name or len(name) < 4:
            continue

        query = f'"{name}" UK'
        articles_data = query_gdelt_doc(query, timespan="1y", max_records=50)
        features = parse_articles(articles_data)

        record = {
            "CompanyNumber": row.get("CompanyNumber", ""),
            "CompanyName": row.get(name_col, ""),
            "bcb_sector": row.get("bcb_sector", ""),
            "gdelt_query": query,
            **features,
        }
        results.append(record)
        time.sleep(DELAY)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCompany-level GDELT saved: {OUTPUT_PATH}")
    print(f"Shape: {out_df.shape}")
    return out_df


def main():
    sector_df = fetch_sector_trends()
    company_df = fetch_company_level_sentiment()

    print("\n=== GDELT Collection Complete ===")
    print(f"Sector trends:    {SECTOR_OUTPUT_PATH}")
    print(f"Company sentiment: {OUTPUT_PATH}")
    print("\nNext step: Run `python news/fetch_opencorporates.py`")


if __name__ == "__main__":
    main()
