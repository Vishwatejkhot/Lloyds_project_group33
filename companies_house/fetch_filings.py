import os
import time
import requests
import pandas as pd
from tqdm import tqdm

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "companies_bcb_filtered.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "filings_and_directors.csv")

API_KEY = ""
BASE_URL = "https://api.company-information.service.gov.uk"
DELAY_SECONDS = 0.6
SAMPLE_PER_SECTOR = 200


def get_headers():
    if API_KEY:
        return {}
    return {}


def get_auth():
    if API_KEY:
        return (API_KEY, "")
    return None


def fetch_company_profile(company_number: str) -> dict:
    url = f"{BASE_URL}/company/{company_number}"
    try:
        r = requests.get(url, auth=get_auth(), timeout=10)
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception as e:
        return {}


def fetch_filing_history(company_number: str, max_items: int = 20) -> list:
    url = f"{BASE_URL}/company/{company_number}/filing-history"
    params = {"items_per_page": max_items, "category": "accounts"}
    try:
        r = requests.get(url, auth=get_auth(), params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("items", [])
        return []
    except Exception:
        return []


def fetch_officers(company_number: str) -> list:
    url = f"{BASE_URL}/company/{company_number}/officers"
    try:
        r = requests.get(url, auth=get_auth(), timeout=10)
        if r.status_code == 200:
            return r.json().get("items", [])
        return []
    except Exception:
        return []


def count_director_changes(officers: list) -> dict:
    from datetime import datetime, timedelta
    one_year_ago = datetime.now() - timedelta(days=365)
    two_years_ago = datetime.now() - timedelta(days=730)

    appointments_1yr = 0
    resignations_1yr = 0
    appointments_2yr = 0
    total_directors = 0

    for officer in officers:
        role = officer.get("officer_role", "")
        if "director" not in role.lower():
            continue
        total_directors += 1

        appointed_str = officer.get("appointed_on", "")
        resigned_str = officer.get("resigned_on", "")

        try:
            if appointed_str:
                appointed = datetime.strptime(appointed_str, "%Y-%m-%d")
                if appointed > one_year_ago:
                    appointments_1yr += 1
                if appointed > two_years_ago:
                    appointments_2yr += 1
            if resigned_str:
                resigned = datetime.strptime(resigned_str, "%Y-%m-%d")
                if resigned > one_year_ago:
                    resignations_1yr += 1
        except ValueError:
            pass

    return {
        "total_directors": total_directors,
        "director_appointments_1yr": appointments_1yr,
        "director_resignations_1yr": resignations_1yr,
        "director_appointments_2yr": appointments_2yr,
        "director_turnover_signal": int(resignations_1yr >= 2 or appointments_1yr >= 3),
    }


def count_late_filings(filings: list) -> dict:
    late_count = 0
    total_filings = len(filings)
    for filing in filings:
        if filing.get("action_date") and filing.get("date"):
            try:
                action = pd.to_datetime(filing["action_date"])
                filed = pd.to_datetime(filing["date"])
                days_after_period_end = (filed - action).days
                if days_after_period_end > 275:
                    late_count += 1
            except Exception:
                pass
    return {
        "total_account_filings": total_filings,
        "late_filings_count": late_count,
        "late_filing_rate": round(late_count / total_filings, 3) if total_filings > 0 else 0,
        "has_late_filing_history": int(late_count > 0),
    }


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Run `python companies_house/filter_sectors.py` first.")
        return

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Loaded {len(df):,} companies from filtered dataset.")

    sampled = (
        df[df["is_active"] == 1]
        .groupby("bcb_sector", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLE_PER_SECTOR), random_state=42))
    )
    print(f"Sampling {len(sampled):,} companies for filing/director enrichment.")

    results = []

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Fetching CH data"):
        company_number = str(row.get("CompanyNumber", "")).zfill(8)
        if not company_number or company_number == "00000000":
            continue

        filings = fetch_filing_history(company_number)
        officers = fetch_officers(company_number)

        filing_stats = count_late_filings(filings)
        director_stats = count_director_changes(officers)

        record = {
            "CompanyNumber": company_number,
            "CompanyName": row.get("CompanyName", ""),
            "bcb_sector": row.get("bcb_sector", ""),
            **filing_stats,
            **director_stats,
        }
        results.append(record)

        time.sleep(DELAY_SECONDS)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved enriched filings data: {OUTPUT_PATH}")
    print(f"Shape: {out_df.shape}")
    print("\nNext step: Run `python gdelt/fetch_gdelt.py`")


if __name__ == "__main__":
    main()
