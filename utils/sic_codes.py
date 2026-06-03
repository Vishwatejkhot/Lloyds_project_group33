BCB_SECTORS = {
    "Manufacturing": list(range(10000, 34000)),

    "Public_Sector_Education_Charities": (
        list(range(84000, 86000)) +
        list(range(85000, 86000)) +
        list(range(88000, 89000)) +
        [99000]
    ),

    "Healthcare": (
        list(range(86000, 88000)) +
        list(range(87000, 88000))
    ),

    "Technology_Legal_Professional": (
        list(range(62000, 64000)) +
        list(range(69000, 75000)) +
        list(range(58000, 64000))
    ),

    "Agriculture": (
        list(range(1000, 4000))
    ),

    "Real_Estate": (
        list(range(68000, 69000))
    ),

    "Wholesale_Retail": (
        list(range(45000, 48000))
    ),

    "Fast_Growth_Emerging": []
}

SIC_TO_SECTOR = {}
for sector, codes in BCB_SECTORS.items():
    for code in codes:
        SIC_TO_SECTOR[str(code)] = sector

def get_sector(sic_code: str) -> str:
    if not sic_code:
        return "Unknown"
    try:
        code_int = int(str(sic_code).strip())
        return SIC_TO_SECTOR.get(str(code_int), "Other/Excluded")
    except ValueError:
        return "Unknown"


if __name__ == "__main__":
    tests = ["62012", "86100", "10110", "68100", "01110"]
    for t in tests:
        print(f"SIC {t} -> {get_sector(t)}")
