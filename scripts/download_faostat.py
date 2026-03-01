import hashlib
from io import BytesIO
from pathlib import Path
import sys
import urllib.request

import pandas as pd

# Download FAOSTAT cereals production for Morocco (area_code=504, cereals total=1717, element=5510)
URL = (
    "https://fenixservices.fao.org/api/faostat/QC"
    "?area_code=504&item_code=1717&element_code=5510&year=1990:2023&format=csv"
)
TIMEOUT_SEC = 90
MAX_RETRIES = 3
RAW_DIR = Path(__file__).resolve().parents[1] / "question_1_food_security" / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RAW_DIR / "morocco_food_security_faostat_cereals_prod_1990_2023.csv"


def main() -> int:
    data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                data = resp.read()
            break
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed: {exc}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                print("[ERROR] Download failed after retries", file=sys.stderr)
                return 1

    sha256 = hashlib.sha256(data).hexdigest()

    try:
        df = pd.read_csv(BytesIO(data))
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Could not parse CSV: {exc}", file=sys.stderr)
        return 1

    # Align units to 1000 MT for comparison with processed data
    if "Value" not in df.columns or "Year" not in df.columns:
        print("[ERROR] Unexpected FAO columns: missing Year/Value", file=sys.stderr)
        return 1

    df = df[["Year", "Value"]].rename(columns={"Value": "cereal_prod_t"})
    df["cereal_prod_1000MT"] = df["cereal_prod_t"] / 1000.0

    # Save raw download for traceability
    df.to_csv(OUTPUT_CSV, index=False)

    # Log simple stats
    years = (int(df["Year"].min()), int(df["Year"].max()))
    print(f"[OK] Downloaded FAOSTAT cereals production {years[0]}-{years[1]}")
    print(f"[OK] Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"[OK] SHA256: {sha256}")
    print(f"[OK] Saved to: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
