import os
import glob
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.sic_codes import get_sector, BCB_SECTORS

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "raw", "companies_house_bulk")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "companies_bcb_filtered.csv")

KEEP_COLS = [
    " CompanyName",
    "CompanyNumber",
    "RegAddress.AddressLine1",
    "RegAddress.PostTown",
    "RegAddress.Country",
    "CompanyCategory",
    "CompanyStatus",
    "CountryOfOrigin",
    "DissolutionDate",
    "IncorporationDate",
    "Accounts.AccountRefDay",
    "Accounts.AccountRefMonth",
    "Accounts.NextDueDate",
    "Accounts.LastMadeUpDate",
    "Accounts.AccountCategory",
    "Returns.NextDueDate",
    "Returns.LastMadeUpDate",
    "Mortgages.NumMortCharges",
    "Mortgages.NumMortOutstanding",
    "Mortgages.NumMortPartSatisfied",
    "Mortgages.NumMortSatisfied",
    "SICCode.SicText_1",
    "SICCode.SicText_2",
    "SICCode.SicText_3",
    "SICCode.SicText_4",
    "LimitedPartnerships.NumGenPartners",
    "LimitedPartnerships.NumLimPartners",
    "URI",
    "PreviousName_1.CONDATE",
    "PreviousName_1.CompanyName",
    "ConfStmtNextDueDate",
    "ConfStmtLastMadeUpDate",
]


def parse_sic_number(sic_text: str) -> str:
    if pd.isna(sic_text):
        return ""
    return str(sic_text).strip().split(" ")[0].strip()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")

    today = pd.Timestamp.today()

    df["incorporation_date"] = pd.to_datetime(df["IncorporationDate"], errors="coerce", dayfirst=True)
    df["company_age_years"] = (today - df["incorporation_date"]).dt.days / 365.25

    df["accounts_next_due"] = pd.to_datetime(df["Accounts.NextDueDate"], errors="coerce", dayfirst=True)
    df["accounts_last_made_up"] = pd.to_datetime(df["Accounts.LastMadeUpDate"], errors="coerce", dayfirst=True)
    df["accounts_overdue_days"] = (today - df["accounts_next_due"]).dt.days.clip(lower=0)
    df["accounts_ever_late"] = (df["accounts_overdue_days"] > 0).astype(int)

    df["conf_stmt_next_due"] = pd.to_datetime(df["ConfStmtNextDueDate"], errors="coerce", dayfirst=True)
    df["conf_stmt_overdue_days"] = (today - df["conf_stmt_next_due"]).dt.days.clip(lower=0)

    df["num_mortgages_outstanding"] = pd.to_numeric(df["Mortgages.NumMortOutstanding"], errors="coerce").fillna(0)
    df["num_mortgages_total"] = pd.to_numeric(df["Mortgages.NumMortCharges"], errors="coerce").fillna(0)
    df["has_active_mortgages"] = (df["num_mortgages_outstanding"] > 0).astype(int)

    df["has_changed_name"] = df["PreviousName_1.CompanyName"].notna().astype(int)

    df["is_active"] = (df["CompanyStatus"].str.lower() == "active").astype(int)
    df["is_dissolved"] = (df["CompanyStatus"].str.lower() == "dissolved").astype(int)

    df["account_category"] = df["Accounts.AccountCategory"].fillna("UNKNOWN").str.upper()
    df["is_dormant"] = (df["account_category"] == "DORMANT").astype(int)
    df["is_micro"] = (df["account_category"].isin(["MICRO-ENTITY", "MICRO ENTITY"])).astype(int)
    df["is_small"] = (df["account_category"] == "SMALL").astype(int)
    df["is_full_accounts"] = (df["account_category"] == "FULL").astype(int)

    df["fast_growth_proxy"] = (
        (df["company_age_years"] < 7) &
        (df["is_active"] == 1) &
        (df["is_dormant"] == 0) &
        (df["accounts_overdue_days"] == 0) &
        (df["is_full_accounts"] == 1)
    ).astype(int)

    return df


def main():
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {RAW_DIR}")
        print("Run `python companies_house/fetch_bulk.py` first.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process.")

    all_chunks = []
    target_sectors = set(BCB_SECTORS.keys()) - {"Fast_Growth_Emerging"}

    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")

        chunk_iter = pd.read_csv(
            csv_file,
            chunksize=50_000,
            encoding="latin-1",
            low_memory=False,
            on_bad_lines="skip"
        )

        for chunk in tqdm(chunk_iter, desc="Filtering chunks"):
            chunk.columns = [c.strip() for c in chunk.columns]

            sic_col = "SICCode.SicText_1"
            if sic_col not in chunk.columns:
                continue

            chunk["sic_code_1"] = chunk[sic_col].apply(parse_sic_number)
            chunk["bcb_sector"] = chunk["sic_code_1"].apply(get_sector)

            filtered = chunk[chunk["bcb_sector"].isin(target_sectors)].copy()

            if len(filtered) > 0:
                all_chunks.append(filtered)

    if not all_chunks:
        print("No matching companies found. Check SIC code mapping.")
        return

    print("\nCombining all filtered chunks...")
    df = pd.concat(all_chunks, ignore_index=True)
    print(f"Total BCB companies found: {len(df):,}")

    print("\n--- Sector Breakdown ---")
    for sector, count in df["bcb_sector"].value_counts().items():
        print(f"  {sector}: {count:,}")

    df = engineer_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nNext step: Run `python gdelt/fetch_gdelt.py`")


if __name__ == "__main__":
    main()
