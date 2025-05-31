# src/build_all_tasks.py

import json
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
EXTRACT_PATH  = BASE_DIR / "data" / "extract_relevant.jsonl"
COMPARE_PATH  = BASE_DIR / "data" / "dataset.jsonl"
FAQ_PATH      = BASE_DIR / "data" / "dataset_faq.jsonl"
OUTPUT_PATH   = BASE_DIR / "data" / "all_tasks.jsonl"

def build_all_tasks():
    # Skapa data-mappen om den inte finns
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        # --- 1) EXTRACT-uppgift ---
        if EXTRACT_PATH.exists():
            for line in open(EXTRACT_PATH, encoding="utf-8"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw_text = obj["raw"].replace("\n", " ")
                tgt      = obj["target"]
                inp      = f"extract: {raw_text}"
                record   = {"input": inp, "target": tgt}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print(f"⚠️  Missade {EXTRACT_PATH} (kör build_extract_relevant.py först).")

        # --- 2) COMPARE-uppgift ---
        if COMPARE_PATH.exists():
            for line in open(COMPARE_PATH, encoding="utf-8"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "input" in obj and "output" in obj:
                    inp = obj["input"].replace("\n", " ")
                    tgt = obj["output"]
                    record = {"input": f"compare: {inp}", "target": tgt}
                else:
                    # Om dataset.jsonl är andra nycklar: serialisera hela objektet som sträng
                    json_str = json.dumps(obj, ensure_ascii=False)
                    record = {"input": f"compare: {json_str}", "target": json_str}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print(f"⚠️  Missade {COMPARE_PATH} (lägg till dataset.jsonl).")

        # --- 3) FAQ-uppgift ---
        if FAQ_PATH.exists():
            for line in open(FAQ_PATH, encoding="utf-8"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                q   = obj["input"]
                a   = obj["output"]
                record = {"input": f"faq: {q}", "target": a}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print(f"⚠️  Missade {FAQ_PATH} (kör preprocess.py först).")

    # Räkna antalet rader
    num = sum(1 for _ in open(OUTPUT_PATH, encoding="utf-8"))
    print(f"✅ Skapade {OUTPUT_PATH} med {num} rader.")

if __name__ == "__main__":
    build_all_tasks()
