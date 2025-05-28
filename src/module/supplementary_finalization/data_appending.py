import random
import json
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import os

class DataAppender:
    """
    Combines audio segments per speaker under a max duration,
    exports augmented WAVs, and writes metadata JSONL.
    """
    def __init__(self, max_total_duration: int = 30, random_seed: int = 74):
        self.max_total_duration = max_total_duration  # seconds
        self.random_seed = random_seed

    @staticmethod
    def calculate_weights(
        n_bucket: dict[int, int],
        m_bucket: dict[int, int],
        total_speakers: int,
        m_gamma_low: int,
        epsilon: float = 1e-6
    ) -> dict[int, float]:
        """
        Compute sampling weights for bucket balancing.
        """
        n_total = sum(n_bucket.values()) or 1
        m_total = sum(m_bucket.values()) or 1
        n_w = {i: (n_total - n_bucket[i]) / n_total for i in range(30)}
        m_w = {i: (m_total - m_bucket[i]) / m_total for i in range(30)}
        weights: dict[int, float] = {}
        for i in range(30):
            w = n_w[i] * (1- 1/(total_speakers+epsilon)) + m_w[i] * (1/(total_speakers+epsilon))
            weights[i] = w if w > 0 else 1.0
        gamma = {i: weights[min(i*(30//m_gamma_low), 29)] for i in range(1, m_gamma_low+1)}
        return gamma

    def load_data(self, input_jsonl_path: str) -> dict[str, list[dict]]:
        """
        Load entries from JSONL and group by speaker folder.
        """
        base_dir = Path(input_jsonl_path).parent
        speaker_data: dict[str, list[dict]] = {}
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                dur = entry.get('duration', 0)
                if dur >= self.max_total_duration:
                    continue
                wav_rel = Path(entry['wav'])
                print(f"wav_rel: {wav_rel}")
                full_wav = base_dir / wav_rel
                print(f"full_wav: {full_wav}")
                if "emilia" in str(full_wav):
                    last_folder = str(wav_rel).split("/")[-1]
                    parts = last_folder.split("_")
                    key = "_".join(parts[:3])
                else:
                    # By default, use the entire wav_rel.parent path as the key
                    key = wav_rel.parent.as_posix()





                speaker_data.setdefault(key, []).append({
                    'wav': full_wav,
                    'duration': dur,
                    'text': entry.get('text', ''),
                    'sr': entry.get('sr') or entry.get('tr') or '',
                    'N2gkPlus': entry.get('N2gkPlus', ''),
                    'speaker': key
                })
        print(f"Loaded {len(speaker_data)} speaker groups from {input_jsonl_path}")
        #print(f"len(speaker_data): {len(speaker_data)}")

        return speaker_data
    

    def run_appending(self, input_jsonl_path: str, output_jsonl_path: str) -> None:

        jsonl_buffer = []  # Buffer
        file_counter = {}
        n_bucket = {i: 0 for i in range(30)}

        output_meta = Path(output_jsonl_path)
        output_meta.parent.mkdir(parents=True, exist_ok=True)
        wav_dir = output_meta.parent / 'wavs_appended'
        wav_dir.mkdir(parents=True, exist_ok=True)


        speaker_data = self.load_data(input_jsonl_path)
        total_rows = sum(len(wavs) for wavs in speaker_data.values())
        total_duration = sum(wav['duration'] for wavs in speaker_data.values() for wav in wavs)
        total_speakers = len(speaker_data)

        random.seed(self.random_seed)

        for group_key, wav_data_list in tqdm(speaker_data.items(), desc="Processing Speakers", unit="speaker"):
            selected_data = []
            m_bucket = {i: 0 for i in range(30)}
            m_threshold = len(wav_data_list) // 465 + 1
            back_count = 0

            current_speaker_total_duration = sum(w['duration'] for w in wav_data_list)
            m = current_speaker_total_duration / len(wav_data_list)
            m_gamma = 30 / m
            m_gamma_low = int(m_gamma)

            with tqdm(total=len(wav_data_list), desc=f"{group_key}", unit="file") as pbar:
                while wav_data_list:
                    weights = self.calculate_weights(n_bucket, m_bucket, total_speakers, m_gamma_low)
                    num_selected = random.choices(range(1, m_gamma_low + 1), weights=weights.values(), k=1)[0]
                    num_selected = min(num_selected, len(wav_data_list))
                    selected_files = random.sample(wav_data_list, num_selected)
                    selected_durations = sum(d['duration'] for d in selected_files)

                    if selected_durations < self.max_total_duration:
                        current_combined = AudioSegment.empty()
                        current_text = ""
                        current_sr = ""
                        current_n2gkplus = ""

                        valid_files = []
                        for data in selected_files:

                            #real_path = os.path.join(rel_path, f"{dataset_name}/{data['wav']}") # Emilia_Dataset, f"{KO}/{B...."" or Emilia_Dataset, f"{KO}/{kss/wavs.."" 
                            real_path = data['wav']
                            if not os.path.exists(real_path):
                                print(f"Warning: File not found: {real_path}")
                                continue
                            try:
                                # Process based on extension
                                if str(real_path).endswith('.mp3'):
                                    wav = AudioSegment.from_mp3(real_path)
                                else:  # If WAV file
                                    wav = AudioSegment.from_wav(real_path)
                            except FileNotFoundError:
                                print(f"Warning: File not found, skipping: {real_path}")
                                continue
                            
                            #wav = AudioSegment.from_wav(real_path)

                            if len(current_combined) > 0:
                                fade = 500
                                current_combined = current_combined.fade_out(fade).append(wav.fade_in(fade))
                            else:
                                current_combined += wav

                            current_text += " " + data['text']
                            current_sr += " " + data['sr'] if data['sr'] is not None else ""
                            current_n2gkplus += " " + data['N2gkPlus']


                            #print(f"current_n2gkplus : {current_n2gkplus}")
                            valid_files.append(data)

                        if not valid_files:
                            wav_data_list = [x for x in wav_data_list if x not in selected_files]
                            continue

                        bucket = int(selected_durations)
                        n_bucket[bucket] += 1
                        m_bucket[bucket] += 1

                        if back_count < 5 and m_bucket[bucket] + 1 > m_threshold:
                            back_count += 1
                            continue
                        else:
                            back_count = 0

                        selected_data.extend(valid_files)
                        wav_data_list = [x for x in wav_data_list if x not in selected_data]

                        speaker = valid_files[0]['speaker']
                        if speaker not in file_counter:
                            file_counter[speaker] = 1

                        
                        speaker_output_dir = wav_dir / speaker


                        os.makedirs(speaker_output_dir, exist_ok=True)

                        file_name = f"{file_counter[speaker]:06d}.wav"
                        output_path = os.path.join(speaker_output_dir, file_name)


                        current_combined.export(output_path, format="wav")

                        jsonl_data = {
                            #"wav": output_path.replace(output_dir, f"{label}_wavs_augmented_t_{t}_beta_{beta}"),
                            "wav": output_path.replace('../data/',''),
                            "duration": selected_durations,
                            "text": current_text.strip(),
                            "sr": current_sr.strip(),
                            "N2gkPlus" : current_n2gkplus.strip(),
                            "speaker": speaker
                        }
                        #print(f"final_jsonl_data : {jsonl_data}")
                        #exit()
                        jsonl_buffer.append(jsonl_data)

                        file_counter[speaker] += 1
                        pbar.update(len(valid_files))


                # Print the number of samples aggregated for each interval (calculated as a percentage)
                print("Sample distribution across 1~29 second buckets:")
                total_samples_sum = sum(n_bucket.values())  # Sum of total samples
                for i in range(0, 30):
                    percent = (n_bucket[i] / total_samples_sum) * 100  # Calculate ratio by number of samples
                    print(f"Bucket {i} seconds: {n_bucket[i]} samples ({percent:.2f}%)")

            total_speakers -= 1

        # Save all at the end
        with open(input_jsonl_path, 'w', encoding='utf-8') as f:
            for row in jsonl_buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")