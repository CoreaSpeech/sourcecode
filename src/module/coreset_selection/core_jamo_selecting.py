import json
import csv
import math
import random
import re
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

from module.coreset_selection.utils import convert_char_to_jamo, calculate_utmos_threshold


class JamoBigram:
    """
    Provides JamoBigram-based jamo pair statistics and coreset filtering functionality.
    """
    def __init__(
        self,
        t: int = 500,
        beta: float = 0.0001,
        utmos_threshold: float | None = None,
        csv_total_table: str | None = None,
        utmos_mode: str = "dynamic",
        utmos_dynamic_type: str = "mad",
        num_workers: int = 4,
        total_table_path: str | None = None, 
    ):
        
        self.t = t
        self.beta = beta
        self.utmos_threshold = utmos_threshold
        self.utmos_mode = utmos_mode
        self.utmos_dynamic_type = utmos_dynamic_type
        self.num_workers = num_workers
        self.total_table_path = total_table_path

       
        self.lookup_table = self.create_pair_lookup()
        max_index = max(self.lookup_table.values())
        self.total_jamo_pair_count = {i: 0 for i in range(max_index + 1)}

       
        if csv_total_table:
            print("Loading Pre-defined CSV table\n")
            self.load_total_csv_table(csv_total_table)

        

    def create_pair_lookup(self):
        CHOSUNG_LIST = [
            'ᄀ','ᄁ','ᄂ','ᄃ','ᄄ','ᄅ','ᄆ','ᄇ','ᄈ','ᄉ',
            'ᄊ','ᄋ','ᄌ','ᄍ','ᄎ','ᄏ','ᄐ','ᄑ','ᄒ'
        ]
        JOONGSUNG_LIST = [
            'ᅡ','ᅢ','ᅣ','ᅤ','ᅥ','ᅦ','ᅧ','ᅨ','ᅩ','ᅪ',
            'ᅫ','ᅬ','ᅭ','ᅮ','ᅯ','ᅰ','ᅱ','ᅲ','ᅳ','ᅴ','ᅵ',
        ]
        JONGSUNG_LIST = [
            'ᆨ','ᆩ','ᆪ','ᆫ','ᆬ','ᆭ','ᆮ','ᆯ','ᆰ','ᆱ',
            'ᆲ','ᆴ','ᆵ','ᆶ','ᆷ','ᆸ','ᆹ','ᆺ','ᆻ','ᆼ',
            'ᆽ','ᆾ','ᆿ','ᇀ','ᇁ','ᇂ','ㄸ'
        ]
        
        lookup_table = {}
        idx = 0
        
        for cho in CHOSUNG_LIST:
            for jung in JOONGSUNG_LIST:
                key = f"'{cho}','{jung}'"
                lookup_table[key] = idx
                idx += 1
       
        for jung in JOONGSUNG_LIST:
            for jong in JONGSUNG_LIST:
                key = f"'{jung}','{jong}'"
                lookup_table[key] = idx
                idx += 1
       
        for jong in JONGSUNG_LIST:
            for cho in CHOSUNG_LIST:
                key = f"'{jong}','{cho}'"
                lookup_table[key] = idx
                idx += 1
        
        for jung in JOONGSUNG_LIST:
            for cho in CHOSUNG_LIST:
                key = f"'{jung}','{cho}'"
                lookup_table[key] = idx
                idx += 1
        return lookup_table

    def _process_chunk(self, chunk):
        """
        Processes a given chunk (list of jamos) using a sliding window approach 
        and returns partial counts as a dictionary.
        """
        partial_counts = {}
        n = len(chunk)
        for i in range(n - 1):
            token1 = chunk[i]
            token2 = chunk[i+1]
            key = f"'{token1}','{token2}'"
            if key in self.lookup_table:
                index = self.lookup_table[key]
                partial_counts[index] = partial_counts.get(index, 0) + 1
        return partial_counts

    def count_lookup_pairs_parallel(self, flattened_jamo_list, num_workers=4):
        """
        flattened_jamo_list: For example, a flat list of jamos like ['ᄋ','ᅡ','ㄴ','ㄴ','ᅧ','ㅇ'].
        
        This method splits flattened_jamo_list into num_workers chunks (with one element overlap at chunk boundaries),
        processes each chunk in parallel using _process_chunk, and sums the partial results from each chunk.
        
        It also cumulatively updates self.total_jamo_pair_count.
        
        Returns: {index: total count for this text, ...}
        """
        n = len(flattened_jamo_list)
        if n < 2:
            return {}
        
        # Chunk splitting: Add one element to the end of each chunk to preserve sliding window boundaries
        chunk_size = n // num_workers
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            # The last chunk goes up to n
            end = (i+1)*chunk_size + 1 if i < num_workers - 1 else n
            chunks.append(flattened_jamo_list[start:end])
        
        partial_counts_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                partial_counts_list.append(future.result())
        
        # Sum partial results and update cumulatively
        final_counts = {}
        for partial in partial_counts_list:
            for index, cnt in partial.items():
                final_counts[index] = final_counts.get(index, 0) + cnt
                self.total_jamo_pair_count[index] += cnt
        return final_counts


    def extract_korean(self, text: str) -> str:
        """
        Removes all non-Hangul (가-힣) characters from the text, leaving only Hangul.
        """
        return re.sub(r'[^가-힣]', '', text)

    def count_lookup_pairs_from_text(self, text):
        """
        Takes plain text as input, generates a flattened list of jamos using the 
        convert_char_to_jamo function, removes spaces (' '), and then calls the 
        count_lookup_pairs_parallel method to return results in the format {index: count}.
        """
        text_ko = self.extract_korean(text)
        nested_jamo_list = convert_char_to_jamo(text_ko)
        flattened_jamo_list = [char for sublist in nested_jamo_list for char in sublist if char != ' ']
        return self.count_lookup_pairs_parallel(flattened_jamo_list)

   
    def filter_instance(self, global_count, t=500, beta=0.0001,sample_utmos=0, utmos_threshold=None):
        """
        Calculates the probability of keeping the jamo for a given global_count.
          - Always True if global_count is less than or equal to t.
          - If greater than t, kept with a probability of p_keep = exp(-beta * (global_count - t)).
        """
        if global_count <= t:
            return True
        else:
            p_keep = math.exp(-beta * (global_count - t))
            if utmos_threshold is not None :
                return (random.random() < p_keep) and (sample_utmos >= utmos_threshold)
            else :
                return random.random() < p_keep

    def should_keep_sample(self, sample_jamobigram, sample_utmos, t=500, beta=0.0001, utmos_threshold=None):

        """
        sample_jamobigram: The "JamoBigram" dictionary for the sample (key: jamo id, value: count).
        Uses self.total_jamo_pair_count for global counts.
        
        If the keep condition (filter_instance) is met for at least one jamo, the sample is kept (True).
        """
        
        for jamo_id, count in sample_jamobigram.items():


            global_count = self.total_jamo_pair_count.get(int(jamo_id), 0)
            
            if self.filter_instance(global_count, t, beta, sample_utmos, utmos_threshold):
                return True
        return False

    def filter_samples(self, samples, t=500, beta=0.0001, utmos_threshold=None):
        """
        samples: A list of dictionaries, where each sample has a "JamoBigram" key (e.g., [{...}, {...}, ...]).
        
        Checks the JamoBigram information of each sample and returns only those samples 
        that satisfy the keep condition (should_keep_sample) based on self.total_jamo_pair_count.
        """
        filtered_samples = []
        for sample in samples:
            sample_jamobigram = sample.get("JamoBigram", {})
            sample_utmos = sample.get("utmos", 0)

            if self.should_keep_sample(sample_jamobigram, sample_utmos,t, beta,  utmos_threshold):
                filtered_samples.append(sample)


        return filtered_samples   

    def save_total_counts(self):
        # Ensure parent directory exists for total_table_path
        total_counts = self.total_jamo_pair_count
        with open(self.total_table_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Jamo_Pair", "Count"])
            for index, count in total_counts.items():
                writer.writerow([index, count])
        print(f"Total counts saved to CSV at: {self.total_table_path}")

    def load_total_csv_table(self, csv_path: str):
        """
        Reads a local CSV file to update self.total_jamo_pair_count.
        The CSV file must have a header ("Jamo_Pair", "Count") in the first line.
        
        Args:
            csv_path (str): Full path to the CSV file (including directory and filename).
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"CSV file not found at: {csv_file}")
            return
        with csv_file.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            # Read each row of the CSV, convert index and count to int, and assign to self.total_jamo_pair_count
            self.total_jamo_pair_count = {int(row["Jamo_Pair"]): int(row["Count"]) for row in reader}
        print(f"Total jamo pair count loaded from CSV at: {csv_file}")

    def apply_jamobigram(self, input_jsonl: str, output_jsonl: str) -> None:
        """
        For each record in the input JSONL, adds a 'JamoBigram' field, 
        cumulatively updates total_jamo_pair_count, and saves an intermediate JSONL 
        (calls save_total_counts if needed).
        """
        inp = Path(input_jsonl)
        out = Path(output_jsonl)
        records = []
        with inp.open('r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Applying JamoBigram'):
                data = json.loads(line)
                counts = self.count_lookup_pairs_from_text(data.get('N2gkPlus', ''))
                data['JamoBigram'] = {str(k): v for k, v in counts.items()}
                records.append(data)
        with out.open('w', encoding='utf-8') as f:
            for rec in records:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        # Save total counts to CSV
        if self.total_table_path:
            self.save_total_counts()

    def run_selection(self, input_jsonl: str, output_jsonl: str) -> None:
        """
        Performs filtering only on a JSONL file where JamoBigram has already been applied, 
        and saves the result.
        """
        inp = Path(input_jsonl)
        out = Path(output_jsonl)
        # 1) Calculate UTMOS threshold
        if self.utmos_threshold is None:
            self.utmos_threshold = calculate_utmos_threshold(
                jsonl_path=inp,
                mode=self.utmos_mode,
                dynamic_type=self.utmos_dynamic_type
            )
            print(f"[UTMOS threshold] = {self.utmos_threshold:.4f}")

        """
        filtered = []
        for s in tqdm(samples, desc='Filtering JamoBigram samples'):
            if self.should_keep_sample(
                {int(k): v for k,v in s.get('JamoBigram', {}).items()},
                s.get('utmos', 0),
                t=self.t,
                beta=self.beta,
                utmos_threshold=self.utmos_threshold
            ):
                filtered.append(s)
        print(f"Filtered {len(filtered)} / {len(samples)} samples")
        """
        samples = []
        # Read all samples from the input JSONL file
        with inp.open("r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}\n{e}")
        
        # Filter samples based on the BalJamo object's total_jamo_pair_count
        print(f't : {self.t}, beta : {self.beta}')
        filtered_samples = self.filter_samples(samples, self.t, self.beta, self.utmos_threshold)
        print(f"Filtered {len(filtered_samples)} / {len(samples)} samples")

        # 4) Save
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            for s in filtered_samples:
                json.dump(s, f, ensure_ascii=False)
                f.write("\n")
        print(f"Selection output saved to: {out}")