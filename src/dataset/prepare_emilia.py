# prepare_emilia.py

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_DIR    = PROJECT_ROOT / "data" / "emilia" / "KO"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "emilia"/ "emilia.jsonl"


if OUTPUT_JSONL.exists():
    OUTPUT_JSONL.unlink()

with OUTPUT_JSONL.open("w", encoding="utf-8") as out_f:
   
    for meta_path in INPUT_DIR.glob("*/*.json"):
        for line in meta_path.open("r", encoding="utf-8"):
            rec = json.loads(line)
            #print(rec)
            parts = rec["wav"].split("/")
           
            if len(parts) < 4:
                continue
            new_wav = f"KO/{parts[0]}{parts[1][-1]}/{parts[-1]}"
            out = {
                "wav": new_wav,
                "text": rec.get("text", ""),
                "sr": "",
                "duration": rec.get("duration", 0.0),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"emilia.jsonl creation complete: {OUTPUT_JSONL}")
