# Korean Sing Speaker Speech Dataset
# download from https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset


import json

from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
metadata_path = project_root / "data" / "kss" / "metadata.txt"
jsonl_path    = project_root / "data" / "kss" / "kss.jsonl"



with open(metadata_path, "r", encoding="utf-8") as infile, open(jsonl_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        parts = line.strip().split("|")
        if len(parts) != 6:
            print(f"Skipping invalid line: {line}")
            continue
        
      
        wav_path, text, sr, _, duration, _ = parts



       
        json_data = {
            "wav": f"wavs/{wav_path}",
            "text": text,
            "sr": sr,
            "duration": float(duration),
            #"speaker": "kss",
        }
        print(json_data)

      
        outfile.write(json.dumps(json_data, ensure_ascii=False) + "\n")

print(f"JSONL conversion complete: {jsonl_path}")
