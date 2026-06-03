import os
import time
import requests
import pandas as pd
from tqdm import tqdm

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "companies_bcb_filtered.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "opencorporates_enriched.csv")

BASE_URL = "https://api.opencorporates.com/v0.4"
DELAY = 1.0
SAMPLE_SIZE = 300


def search_company(company_name: str, jurisdiction: str = "gb") -> list:
    url = f"{BASE_URL}/companies/search"
    params = {
        "q": company_name,
        "jurisdiction_code": jurisdiction,
        "per_page": 5,
        "inactive": "false",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return data.get("results", {}).get("companies", [])
        elif r.status_code == 429:
            print("Rate limited by OpenCorporates. Waiting 60 seconds...")
            time.sleep(60)
            return []
        return []
    except Exception as e:
        return []


def get_company_detail(company_number: str, jurisdiction: str = "gb") -> dict:
    url = f"{BASE_URL}/companies/{jurisdiction}/{company_number}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json().get("results", {}).get("company", {})
        return {}
    except Exception:
        return {}


def extract_oc_features(oc_data: dict) -> dict:
    if not oc_data:
        return {
            "oc_found": 0,
            "oc_registered_address_country": "",
            "oc_company_type": "",
            "oc_source_url": "",
            "oc_number_of_employees": "",
            "oc_has_parent": 0,
            "oc_has_subsidiaries": 0,
            "oc_industry": "",
        }

    company = oc_data if "name" in oc_data else oc_data.get("company", {})

    return {
        "oc_found": 1,
        "oc_registered_address_country": company.get("registered_address_in_full", ""),
        "oc_company_type": company.get("company_type", ""),
        "oc_source_url": company.get("opencorporates_url", ""),
        "oc_number_of_employees": company.get("number_of_employees", ""),
        "oc_has_parent": int(bool(company.get("ultimate_beneficial_owners") or
                                   company.get("controlling_entity"))),
        "oc_has_subsidiaries": int(bool(company.get("subsidiaries"))),
        "oc_industry": company.get("industry_codes", [{}])[0].get("description", "")
                        if company.get("industry_codes") else "",
    }


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Run companies_house/filter_sectors.py first.")
        return

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Loaded {len(df):,} companies.")

    sample = (
        df[df["is_active"] == 1]
        .groupby("bcb_sector", group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), SAMPLE_SIZE // df["bcb_sector"].nunique()),
            random_state=42
        ))
    )
    print(f"Querying OpenCorporates for {len(sample):,} companies...")

    name_col = "CompanyName" if "CompanyName" in sample.columns else " CompanyName"
    results = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="OpenCorporates"):
        company_name = str(row.get(name_col, "")).strip()
        company_number = str(row.get("CompanyNumber", "")).zfill(8)

        oc_data = get_company_detail(company_number)

        if not oc_data:
            matches = search_company(company_name)
            if matches:
                oc_data = matches[0].get("company", {})

        features = extract_oc_features(oc_data)

        record = {
            "CompanyNumber": company_number,
            "CompanyName": company_name,
            "bcb_sector": row.get("bcb_sector", ""),
            **features,
        }
        results.append(record)
        time.sleep(DELAY)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {out_df.shape}")
    found_rate = out_df["oc_found"].mean() * 100
    print(f"Match rate: {found_rate:.1f}%")
    print("\nNext step: Run `python utils/merge_all.py`")


if __name__ == "__main__":
    main()
