import subprocess
import sys
import os

STEPS = [
    ("Step 1/6 — Install dependencies",
     [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"]),

    ("Step 2/6 — Download Companies House bulk data (~700MB, may take 10 mins)",
     [sys.executable, "companies_house/fetch_bulk.py"]),

    ("Step 3/6 — Filter for BCB target sectors",
     [sys.executable, "companies_house/filter_sectors.py"]),

    ("Step 4/6 — Fetch filing history & director changes (CH API, no key needed)",
     [sys.executable, "companies_house/fetch_filings.py"]),

    ("Step 5/6 — Fetch GDELT news sentiment (no key needed)",
     [sys.executable, "gdelt/fetch_gdelt.py"]),

    ("Step 6/6 — Fetch OpenCorporates enrichment (no key needed)",
     [sys.executable, "news/fetch_opencorporates.py"]),

    ("Merging all data sources into final dataset",
     [sys.executable, "utils/merge_all.py"]),
]

def run_step(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"\nERROR in: {label}")
        print("You can skip this step and continue with the next one.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            sys.exit(1)

if __name__ == "__main__":
    print("Lloyds BCB Dissertation — Data Collection Pipeline")
    print("All sources are FREE and require NO API keys.\n")

    for label, cmd in STEPS:
        run_step(label, cmd)

    print("\n" + "="*60)
    print("  DATA COLLECTION COMPLETE")
    print("="*60)
    print("\nYour final dataset is at: output/final_model_dataset.csv")
    print("Summary report is at:     output/final_model_dataset_summary.txt")
    print("\nTo start modelling:")
    print("  import pandas as pd")
    print("  df = pd.read_csv('output/final_model_dataset.csv')")
