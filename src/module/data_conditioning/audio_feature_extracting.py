import json
from pathlib import Path
from tqdm import tqdm
from typing import Union, Literal
import torch
import librosa
from pyannote.audio import Pipeline


class AudioFeatureExtractor:
    """
    Extracts audio features such as speaker count (diarization) and UTMOS score
    and enriches JSONL records with these fields.
    """
    def __init__(self, hf_token: str, device: str = None):
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize speaker diarization pipeline
        self.diar_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(torch.device(self.device))

        # Initialize UTMOS predictor
        self.utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        ).to(self.device)

    def compute_diarization(self, audio_path: Path) -> int:
        """
        Perform speaker diarization and return the number of unique speakers.
        """
        result = self.diar_pipeline({"uri": audio_path.stem, "audio": str(audio_path)})
        speaker_labels = {label for _, _, label in result.itertracks(yield_label=True)}
        return len(speaker_labels)

    def compute_utmos(self, audio_path: Path) -> float:
        """
        Compute the UTMOS score for the given audio file.
        """
        wav, sr = librosa.load(str(audio_path), sr=None, mono=True)
        wav_tensor = torch.from_numpy(wav).to(self.device).unsqueeze(0)
        score = self.utmos_model(wav_tensor, sr)
        return float(score.item())

    def run_enrichment(self, input_jsonl: str, output_jsonl: str) -> None:
        """
        Read input JSONL, enrich each record with "n_speakers" and "utmos",
        and write the results to output JSONL.
        """
        input_path = Path(input_jsonl)
        output_path = Path(output_jsonl)


        lines = input_path.read_text(encoding="utf-8").splitlines()
        enriched = []

        for line in tqdm(lines, desc="Enriching Audio Features"):
            data = json.loads(line)
            audio_file = input_path.parent / data.get("wav", "")

            if not audio_file.exists():
                print(f"[!] Audio file not found: {audio_file}")
                continue

            data["n_speakers"] = self.compute_diarization(audio_file)
            data["utmos"] = self.compute_utmos(audio_file)
            enriched.append(data)

        with output_path.open("w", encoding="utf-8") as outf:
            for record in enriched:
                json.dump(record, outf, ensure_ascii=False)
                outf.write("\n")

        print(f"Enriched data saved to {output_path}")

"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich JSONL with speaker and UTMOS features"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--hf-token", required=True, help="Hugging Face auth token for diarization"
    )

    args = parser.parse_args()
    extractor = AudioFeatureExtractor(hf_token=args.hf_token)
    extractor.run_enrichment(args.input, args.output)
"""