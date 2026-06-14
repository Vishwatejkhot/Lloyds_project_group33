import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def get_bulk_url():
    now = datetime.now()
    for month_offset in range(0, 3):
        month = now.month - month_offset
        year = now.year
        if month <= 0:
            month += 12
            year -= 1
        url = f"https://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-{year}-{month:02d}-01.zip"
        print(f"Trying URL: {url}")
        r = requests.head(url, timeout=10)
        if r.status_code == 200:
            return url
    raise RuntimeError("Could not find Companies House bulk file. Check: https://download.companieshouse.gov.uk/en_output.html")


def download_with_progress(url, dest_path):
    print(f"\nDownloading Companies House bulk data...")
    print(f"Source: {url}")
    print("This is ~700MB and may take 5-15 minutes depending on your connection.\n")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"\nSaved to: {dest_path}")
    return dest_path


def extract_zip(zip_path, extract_to):
    print(f"\nExtracting ZIP file...")
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        print(f"Files inside ZIP: {names}")
        z.extractall(extract_to)
    print(f"Extracted to: {extract_to}")
    return [os.path.join(extract_to, n) for n in names if n.endswith(".csv")]


def load_and_preview(csv_files):
    df = pd.read_csv(csv_files[0], nrows=5, encoding="latin-1")
    print("\n--- Companies House Bulk Data: Column Names ---")
    for col in df.columns:
        print(f"  {col}")
    print(f"\nTotal columns: {len(df.columns)}")
    return df


def main():
    zip_path = os.path.join(RAW_DIR, "companies_house_bulk.zip")
    extract_path = os.path.join(RAW_DIR, "companies_house_bulk")
    os.makedirs(extract_path, exist_ok=True)

    if os.path.exists(zip_path):
        print(f"ZIP already exists at {zip_path}, skipping download.")
        print("Delete the file to re-download.")
    else:
        url = get_bulk_url()
        download_with_progress(url, zip_path)

    csv_files = extract_zip(zip_path, extract_path)

    if csv_files:
        load_and_preview(csv_files)
        print(f"\nNext step: Run `python companies_house/filter_sectors.py`")
    else:
        print("WARNING: No CSV files found in ZIP. Check the extract folder manually.")
        print(f"Path: {extract_path}")


if __name__ == "__main__":
    main()
