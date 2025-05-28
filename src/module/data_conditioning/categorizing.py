import json
import re
from pathlib import Path
from tqdm import tqdm


class LNCat:
    """
    Categorizes text by language combinations and determines English convertibility.
    """
    def __init__(self):
        self.kor_pattern = re.compile(r'[\uac00-\ud7a3]+')
        self.eng_pattern = re.compile(r'[A-Za-z]+')
        self.jpn_pattern = re.compile(r'[\u3040-\u30ff\u31f0-\u31ff]+')
        self.chn_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.num_pattern = re.compile(r'[0-9]+')
        self.units = {"kg", "cm", "g", "km", "m", "mm", "l", "ml", "t", "ha", "mg"}
        self.allowed_categories = {"ko_only", "ko_num", "ko_en", "ko_en_num"}

    def categorize(self, text: str) -> str:
        has_kor = bool(self.kor_pattern.search(text))
        has_eng = bool(self.eng_pattern.search(text))
        has_jpn = bool(self.jpn_pattern.search(text))
        has_chn = bool(self.chn_pattern.search(text))
        has_num = bool(self.num_pattern.search(text))
        total_langs = sum((has_kor, has_eng, has_jpn, has_chn))

        if has_kor:
            if has_eng and has_num and not (has_jpn or has_chn):
                return "ko_en_num"
            if has_num and not (has_eng or has_jpn or has_chn):
                return "ko_num"
            if has_eng and not (has_jpn or has_chn):
                return "ko_en"
            if has_jpn and not (has_eng or has_chn):
                return "ko_jp"
            if has_chn and not (has_eng or has_jpn):
                return "ko_zh"
            if total_langs > 1:
                return "ko_other"
            return "ko_only"

        if has_eng and not (has_jpn or has_chn):
            return "en_only"
        if has_jpn and not (has_eng or has_chn):
            return "jp_only"
        if has_chn and not (has_eng or has_jpn):
            return "zh_only"
        if total_langs > 1:
            return "other_other"
        return "other"

    def is_en_convertable(self, text: str) -> bool:
        tokens = self.eng_pattern.findall(text)
        if not tokens:
            return False
        for token in tokens:
            lower = token.lower()
            if lower in self.units or re.fullmatch(r'[A-Z]{1,4}', token) or re.fullmatch(r'[a-z]', token):
                continue
            return False
        return True
      
    def _filter_record(self, data: dict, category: str) -> bool:
        
        """
        Returns True if record passes all filters:
        - n_speakers == 1 (if present)
        - category in allowed_categories
        - if category is ko_en or ko_en_num then en_convertable must be True
        """
        if data.get('n_speakers', 1) != 1:
            return False
        if category not in self.allowed_categories:
            return False
        if category in {'ko_en', 'ko_en_num'} and not data.get('en_convertable', False):
            return False
        return True

    def run_categorization(self, input_jsonl: str, output_jsonl: str) -> None:
        input_path = Path(input_jsonl)
        output_path = Path(output_jsonl)
        entries = []

        with input_path.open('r', encoding='utf-8') as infile:
            for line in tqdm(infile, desc='Categorizing'):
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get('text', '')
                category = self.categorize(text)
                data['LNCat'] = category
                data['en_convertable'] = self.is_en_convertable(text) if category in {'ko_en', 'ko_en_num'} else None

                # Apply filtering via helper
                if not self._filter_record(data, category):
                    continue
                entries.append(data)

        with output_path.open('w', encoding='utf-8') as outfile:
            for record in entries:
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')

        print(f"Categorized output saved to: {output_path}")
