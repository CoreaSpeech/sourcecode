import json
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────
# Main Functions
# ────────────────────────────────────────────────────────────────

def compute_diarization(audio_path: Path, hf_token: str) -> int:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device(device))

    diarization = pipeline({"uri": "sample", "audio": str(audio_path)})
    speaker_set = set(label for _, _, label in diarization.itertracks(yield_label=True))
    return len(speaker_set)

def compute_utmos(audio_path: Path) -> float:
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device)

    wav, sr = librosa.load(audio_path, sr=None, mono=True)
    wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
    score = predictor(wav_tensor, sr)
    return score.item()

def run_enrichment(input_jsonl_path: str, output_jsonl_path: str, hf_token: str):
    input_path = Path(input_jsonl_path)
    output_path = Path(output_jsonl_path)

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    updated_data = []
    for line in tqdm(lines, desc="Diarization & UTMOS"):
        data = json.loads(line)
        audio_path = input_path.parent / data["wav"]

        if not audio_path.exists():
            print(f"[!] Missing audio file: {audio_path}")
            continue

        utmos = compute_utmos(audio_path)
        speakers = compute_diarization(audio_path, hf_token)

        data["utmos"] = utmos
        data["n_speakers"] = speakers
        updated_data.append(data)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in updated_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Output saved: {output_path}")
