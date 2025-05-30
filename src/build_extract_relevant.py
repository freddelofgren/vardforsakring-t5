# src/build_extract_relevant.py

import json
from pathlib import Path

# 1) Paths
BASE_DIR      = Path(__file__).parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
DATASET_PATH  = BASE_DIR / "data" / "dataset.jsonl"
OUTPUT_PATH   = BASE_DIR / "data" / "extract_relevant.jsonl"

def find_raw_file(company_name: str) -> Path:
    """
    Försöker matcha ett försäkringsbolag (t.ex. "Folksam Bas") 
    mot en txt-fil i data/raw/ (t.ex. Folksam.txt).
    """
    key = company_name.split()[0].lower()  # "Folksam Bas" -> "folksam"
    for txt in RAW_DIR.glob("*.txt"):
        if txt.stem.lower().startswith(key):
            return txt
    raise FileNotFoundError(f"Ingen råtext hittades för '{company_name}' (letade efter prefix '{key}')")

def build_extract_relevant():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        raise RuntimeError(f"Råtexts-mappen saknas: {RAW_DIR}")

    with open(DATASET_PATH, encoding="utf-8") as ds_f, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:

        for line in ds_f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            försäkring = entry.get("försäkring")
            if not försäkring:
                continue  # hoppa om ingen "försäkring"-nyckel

            # 2) Hitta rätt txt-fil och läs in hela råtexten
            raw_path = find_raw_file(försäkring)
            raw_text = raw_path.read_text(encoding="utf-8")

            # 3) Target är hela JSON-objektet som sträng
            target_str = json.dumps(entry, ensure_ascii=False)

            # 4) Skriv ut i extract_relevant.jsonl
            record = {
                "raw":    raw_text,
                "target": target_str
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Skapade {OUTPUT_PATH} med {sum(1 for _ in open(OUTPUT_PATH, encoding='utf-8'))} rader.")

if __name__ == "__main__":
    build_extract_relevant()
