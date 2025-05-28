from typing_extensions import Text
import re
from typing import Union, Literal
from pathlib import Path
from tqdm import tqdm
import json
class N2gk:

    # ------------------- English-Korean Dictionary -------------------
    ENGLISH_NUMBER_MAP = {
        0: '제로', 1: '원', 2: '투', 3: '쓰리', 4: '포', 5: '파이브',
        6: '식스', 7: '세븐', 8: '에잇', 9: '나인', 10: '텐'
    }


    # ------------------- Numeral Dictionary -------------------
    GOOYO_SIP = {
        10: '열', 20: '스물', 30: '서른', 40: '마흔',
        50: '쉰', 60: '예순', 70: '일흔', 80: '여든', 90: '아흔'
    }

    BASIC_NATIVE = {
        1: ('하나', '한'),
        2: ('둘', '두'),
        3: ('셋', '세'),
        4: ('넷', '네'),
        5: ('다섯', '다섯'),
        6: ('여섯', '여섯'),
        7: ('일곱', '일곱'),
        8: ('여덟', '여덟'),
        9: ('아홉', '아홉')
    }

    GOOYO_PREFIX_TENS = {20: '스무'}

    NUM_KOR = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    UNIT_SMALL = ['', '십', '백', '천']
    UNIT_LARGE = ['', '만', '억', '조', '경']
    NEVER_SKIP_ONE = {'억', '조', '경'}

    EXCEPTION_CASES = {
        r'\b20\s?살\b': '스무 살',
        r'\b1\s?등\b': '일 등',
        r'(?<!\d)(0?6)\s*월': '유월',
        r'(?<!\d)(10)\s*월': '시월',
    }


    def __init__(self, natural=True):
        self.natural = natural
        self.natural = natural
        self.UNIT_CATEGORIES = [
            self.UnitCategory(['명', '사람', '마리','번째','시', '배', '방', '가구', '게임', '건', '세트'], 'native', self),
            self.UnitCategory(['개', '가지', '개비', '잔','번', '장','병', '권', '벌', '곳','시간','척', "차례", '바퀴', '경기', '골'], 'native', self),
            self.UnitCategory(['초', '분', '일', '주', '개월', '월','년'], 'hanja', self),
            self.UnitCategory(['점', '포인트', '퍼센트','%' '레벨', '점수', '등급','등', '개국', '볼트'], 'hanja', self),
            self.UnitCategory(['원', '달러', '유로', '엔', '조','페소', '베럴'], 'hanja', self),
            self.UnitCategory(['kg','Kg','mg','g','t','T', 'l','L', 'ml','cm','mm','m','km', 'k'
                               '킬로그램','미리그램','그램','톤','리터','미리리터','센치미터','미리미터','미터','키로미터','케이'], 'hanja', self, convert_unit_name=True),
            self.UnitCategory(['회', '차', '기', '호', '페이지', '장'], 'hanja', self),
            self.UnitCategory(['코어', '스레드', '파일', '채널', '명령어'], 'hanja', self),
            self.UnitCategory(['살', '연세','춘추'], 'native', self),
            self.UnitCategory(['도', '℃', '°C', 'C'], 'hanja', self, convert_unit_name=True),

        ]
        pairs = []
        for cat in self.UNIT_CATEGORIES:
            for unit in cat.units:
                pairs.append((unit, cat))
        # Sort by length of unit name in descending order
        self.unit_category_pairs = sorted(pairs, key=lambda x: len(x[0]), reverse=True)

    # ------------------- Conversion Functions -------------------

    def convert_english_number(self, text: str) -> str:
        pattern = r'([a-zA-Z]+)(\d+)'  # English + number pattern

        def replacer(match):
            english_part = match.group(1)  # English part (e.g., K)
            number_part = int(match.group(2))   # Number part (e.g., 2)

            # If the number is between 1 and 10, convert the number to Korean
            if 0 <= number_part <= 10:
                number_in_korean = self.ENGLISH_NUMBER_MAP[number_part]
            else:
                number_in_korean = str(number_part)  # Otherwise, use the number as is

            # Return the result after converting English + number to Korean
            return f"{english_part} {number_in_korean}"

        # Convert text using regular expressions
        return re.sub(pattern, replacer, text)

    def to_gooyo(self, num, prefix=False):
        if num <= 9:
            base = self.BASIC_NATIVE.get(num)
            return base[1] if prefix else base[0] if base else '영'
        elif num == 10:
            return '열'
        elif num < 100:
            tens = (num // 10) * 10
            ones = num % 10
            if prefix and ones == 0 and tens in self.GOOYO_PREFIX_TENS:
                return self.GOOYO_PREFIX_TENS[tens]
            return self.GOOYO_SIP.get(tens, '') + self.to_gooyo(ones, prefix=prefix) if ones else self.GOOYO_SIP.get(tens)
        else:
            raise ValueError("Native Korean numerals are supported up to 99.")

    def split_number_chunks(self, num_str, chunk_size=4):
        return [num_str[max(i - chunk_size, 0):i] for i in range(len(num_str), 0, -chunk_size)][::-1]

    def convert_small_unit(self, chunk, natural=True):
        result = ''
        length = len(chunk)
        for i, digit_char in enumerate(chunk):
            digit = int(digit_char)
            if digit == 0: continue
            unit = self.UNIT_SMALL[length - i - 1]
            result += unit if digit == 1 and unit and natural else self.NUM_KOR[digit] + unit
        return result

    def to_hanja_int(self, num: int, natural=True) -> str:
        #print("hi to_hanja")
        if num == 0:
            return '영'
        if num < 0:
            return '마이너스 ' + self.to_hanja(-num, natural)
        chunks = self.split_number_chunks(str(num))

        result = ''
        for i, chunk in enumerate(chunks):
            part = self.convert_small_unit(chunk.zfill(4), natural)
            unit = self.UNIT_LARGE[len(chunks) - i - 1]
            """
            if part == '일':
                if (natural and unit not in self.NEVER_SKIP_ONE) or (not natural and unit in self.NEVER_SKIP_ONE):
                    part = ''
            """

            if part == '일' and unit:
                if (natural and unit not in self.NEVER_SKIP_ONE) or (not natural and unit in self.NEVER_SKIP_ONE):
                    part = ''
            result += part + unit
        return result



    def to_hanja(self, num, natural=True) -> str:

        if isinstance(num, float):
            #print(f"num : {num}")
            int_part = int(num)
            #print(f"int_part : {int_part}")
            frac_part_str = str(num).split('.')[1]
            #print(f"fac_part_str : {frac_part_str}")

            int_kor = self.to_hanja(int_part, natural)
            frac_kor = ''.join(
                self.NUM_KOR[int(ch)] if ch != '0' else '영'      # If 0, then '영'
                for ch in frac_part_str
            )
            return f"{int_kor}점{frac_kor}"

        if isinstance(num, str):

            try:
                num = float(num) if '.' in num else int(num)
                return self.to_hanja(num, natural)
            except:
                return str(num)

        if num == 0:
            return '영'
        if num < 0:
            return '마이너스 ' + self.to_hanja(-num, natural)

        chunks = self.split_number_chunks(str(num))
        if len(chunks) > 5:
            return str(num)
        result = ''
        for i, chunk in enumerate(chunks):
            chunk_num = int(chunk)
            if chunk_num == 0:
                continue
            part = self.convert_small_unit(chunk.zfill(4), natural)

            unit = self.UNIT_LARGE[len(chunks) - i - 1]
            if part == '일' and unit:
                if (natural and unit not in self.NEVER_SKIP_ONE) or (not natural and unit in self.NEVER_SKIP_ONE):
                    part = ''
            result += part + unit
        return result

    def n2gk_with_unit(self, num: int, unit: str) -> str:
        for cat in self.UNIT_CATEGORIES:
            if cat.matches(unit):
                return cat.apply(num, unit, natural=self.natural)
        return self.to_hanja(num, natural=self.natural) + unit

    # ------------------- Unit Categories -------------------
    class UnitCategory:
        def __init__(self, units, style, outer, convert_unit_name=False):
            self.units = set(units)
            self.style = style
            self.outer = outer
            self.convert_unit_name = convert_unit_name
            self.unit_name_map = {
                'kg': '킬로그램', 'Kg': '킬로그램',
                'g': '그램',
                'mg': '밀리그램',
                't': '톤', 'T': '톤',
                'l': '리터', 'L': '리터',
                'ml': '밀리리터',
                'cm': '센티미터',
                'mm': '밀리미터',
                'm': '미터',
                'km': '킬로미터',
                'k': '케이', 'K': '케이',
                'ha': '헥타르',
            }

        def matches(self, unit: str) -> bool:
            return unit in self.units

        def apply(self, num: int, unit: str, natural=True) -> str:
            display_unit = unit
            """
            # If convert_unit_name is True, search for prefix mapping
            if self.convert_unit_name:
                for key in sorted(self.unit_name_map, key=len, reverse=True):  # Match from the longest
                    if unit.startswith(key):
                        display_unit = self.unit_name_map[key] + unit[len(key):]  # Convert prefix mapping
                        break
            """
            display_unit = self.unit_name_map[unit] if self.convert_unit_name and unit in self.unit_name_map else unit

            if self.style == 'native':
                #return self.outer.to_gooyo(num, prefix=True) + ' ' + display_unit
                return self.outer.to_gooyo(num, prefix=True) + display_unit
            else:
                #return self.outer.to_hanja(num, natural=natural) + ' ' + display_unit
                return self.outer.to_hanja(num, natural=natural) + display_unit



    # ------------------- Practical Parsing Functions -------------------
    def convert_phone_numbers(self, text: str) -> str:
        DIGIT_KOR = ['공', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']

        def convert_number_str_to_korean(num_str: str) -> str:
            return ''.join([DIGIT_KOR[int(d)] for d in num_str])

        pattern_hyphen = r'(?<!\d)(\d{3})-(\d{3,4})-(\d{4})(?!\d)'
        pattern_full = r'(?<!\d)(\d{11})(?!\d)'

        def hyphen_replacer(match):
            return f"{convert_number_str_to_korean(match.group(1))}-{convert_number_str_to_korean(match.group(2))}-{convert_number_str_to_korean(match.group(3))}"

        def full_replacer(match):
            num = match.group(1)
            return f"{convert_number_str_to_korean(num[:3])}-{convert_number_str_to_korean(num[3:7])}-{convert_number_str_to_korean(num[7:])}"

        text = re.sub(pattern_hyphen, hyphen_replacer, text)
        text = re.sub(pattern_full, full_replacer, text)
        return text

    def apply_exceptions(self, text: str) -> str:
        for pattern, replacement in self.EXCEPTION_CASES.items():
            text = re.sub(pattern, replacement, text)
        return text

    def convert_pure_numbers(self, text: str) -> str:
        pattern = r'(?<![\d가-힣])(\d{1,3}(?:,\d{3})*|\d+)(?![\d가-힣])'
        #pattern = r'(?<![\d가-힣])(\d{1,3}(?:,\d{3})*|\d+)'

        def replacer(m):
            num = int(m.group(1).replace(',', ''))
            return self.to_hanja(num, natural=self.natural)

        return re.sub(pattern, replacer, text)

    def convert_numbers_whatever(self, text: str) -> str:
        pattern = r'(\d{1,3}(?:,\d{3})*|\d+)'
        #pattern = r'(?<![\d가-힣])(\d{1,3}(?:,\d{3})*|\d+)'

        def replacer(m):
            num = int(m.group(1).replace(',', ''))
            return self.to_hanja(num, natural=self.natural)

        return re.sub(pattern, replacer, text)

    def insert_space_around_numbers(self,text: str) -> str:
        # Add space before Korean/English attached to a number
        text = re.sub(r'([가-힣a-zA-Z])(\d)', r'\1 \2', text)
        # Add space after a number attached to Korean/English
        text = re.sub(r'(\d)([가-힣a-zA-Z])', r'\1 \2', text)
        return text


    def parse_and_convert_sentence_with_range(self, sentence: str) -> str:
        range_pattern = r'(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*~\s*(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*([가-힣a-zA-Z]+)'

        def range_replacer(match): # Since numbers have a ',' every 3 digits
            left_raw = match.group(1).replace(',', '')
            right_raw = match.group(2).replace(',', '')
            unit = match.group(3)

            try:
                left = float(left_raw) if '.' in left_raw else int(left_raw)

                right = float(right_raw) if '.' in right_raw else int(right_raw)
                #print(f"left : {left}")
                #print(f"right: {right}")
                l = self.n2gk_with_unit(left, unit).replace(f'{unit}','')
                r = self.n2gk_with_unit(right, unit).replace(f'{unit}','')
                #print(f"l : {l}")
                #print(f"r : {r}")
                return f"{l}에서 {r} {unit}"
            except:
                return match.group(0)

        sentence = re.sub(range_pattern, range_replacer, sentence)
        #print(f"n2gk(range replacer) : {sentence}")
        return self.parse_and_convert_sentence(sentence)


    def parse_and_convert_sentence(self, sentence: str) -> str:
        pattern = r'(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s?([가-힣a-zA-Z]+)'

        def replacer(match):
            raw_number = match.group(1).replace(',', '')
            #print(f"n2gk replacer (raw number) : {raw_number}")
            word = match.group(2)
            #print(f"n2gk replacer (word) : {word}")

            try:
                num = float(raw_number) if '.' in raw_number else int(raw_number)

                # traditional way just using FOR iteration
                """
                for cat in self.UNIT_CATEGORIES:
                    for unit in cat.units:
                        if word.startswith(unit):
                            replaced = cat.apply(num, unit, natural=self.natural)
                            return replaced + word[len(unit):]
                """
                for unit, cat in self.unit_category_pairs:
                    if word.startswith(unit):
                        converted = cat.apply(num, unit, natural= self.natural)
                       
                        return converted + word[len(unit):]

            except:
                pass

            return match.group(0)

        return re.sub(pattern, replacer, sentence)


    def convert_float_numbers(self, text: str) -> str:
        pattern = r'(\d+\.\d+)'

        def replacer(match):
            num_str = match.group(1)
            try:
                zeros_count = 0
                #print(f"num_str : {num_str}")
                if num_str[-1] == "0":
                    zeros_count = len(num_str) - len(num_str.rstrip('0'))
                num = float(num_str)
                return self.to_hanja(num) + "영"*zeros_count
            except:
                return num_str

        return re.sub(pattern, replacer, text)

    def convert_comma_separated_numbers_with_unit(self, text: str) -> str:
        # e.g., "7, 8시" → "일곱, 여덟 시"
        pattern = r'((\d{1,3})(?:\s*,\s*\d{1,3})+)\s*([가-힣]+)'

        def replacer(match):
            number_part = match.group(1)  # "7, 8"
            unit = match.group(3)         # "시"
            numbers = [int(n.strip()) for n in number_part.split(', ')]

            results = []
            for num in numbers:
                converted = self.n2gk_with_unit(num, unit).replace(f' {unit}', '')
                results.append(converted)

            #return ', '.join(results) + ' ' + unit
            return ', '.join(results) + unit

        return re.sub(pattern, replacer, text)


    def __call__(self, sentence: str) -> str: ## super().__call__(sentence)

        sentence = self.apply_exceptions(sentence)
        #print(f"n2gk : apply exception : {sentence}")
        sentence = self.convert_english_number(sentence)
        #print(f"n2gk : convert english number : {sentence}")
        sentence = self.convert_phone_numbers(sentence)
        #print(f"n2gk : convert phone numbers : {sentence}")

        #sentence = self.convert_comma_separated_numbers_with_unit(sentence)  # ✅ 이 줄 추가


        sentence = self.parse_and_convert_sentence_with_range(sentence)
        #print(f"n2gk : parse and convert sentence with range : {sentence}")

        sentence = self.insert_space_around_numbers(sentence)
        #print(f"n2gk : insert space around numbers : {sentence}")
        sentence = self.convert_float_numbers(sentence)
        #print(f"n2gk : convert float numbers : {sentence}")
        sentence = self.convert_pure_numbers(sentence)
        #print(f"n2gk : conert pure numbers : {sentence}")

        return sentence

    def run_n2gk(
            self,
            input_jsonl_path: Union[str, Path],
            output_jsonl_path: Union[str, Path]
    ) -> None:
        """
        Reads JSONL, normalizes each record['text'], adds 'N2gk' field with normalized text,
        and writes to output JSONL.
        """
        input_path = Path(input_jsonl_path)
        output_path = Path(output_jsonl_path)

        records = []
        for line in tqdm(input_path.open('r', encoding='utf-8'), desc='Normalizing (N2gk)'):
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get('text', '')
            data['N2gk'] = self(text)
            records.append(data)

        with output_path.open('w', encoding='utf-8') as outf:
            for rec in records:
                json.dump(rec, outf, ensure_ascii=False)
                outf.write('\n')

        print(f"N2gk-normalized output saved to: {output_path}")


class N2gkPlus(N2gk):
    
    UNIT_MAPPING = {
        "KM": "킬로미터",
        "MM": "밀리미터",
        "M": "미터",
        "CM": "센티미터",
        "KG": "킬로그램",
        "G": "그램",
        "MG": "밀리그램",
        "L": "리터",
        "ML": "밀리리터",
        "HA" : "헥타르",
        "㎡" : "제곱미터",
        "V" : "볼트",
        "㎾" : "키로와트",

    }

    COMMON_ABBR_MAPPING = {
        #"TV": "티비",
        #"PC": "피시",
        #"AI": "에이아이",
        #"AS": "에이에스",
        #"USB": "유에스비",
        #"CPU": "씨피유",
        #"GPU": "지피유",
        #"LED": "엘이디",
        "RAM": "램",
        "LAN": "랜",
        #"DVD": "디브이디",
        #"VIP": "브이아이피",
        #"EPS": "이에스피",
        #"XY": "엑스와이",
        #"FC": "에프씨",
        #"CCTV": "씨씨티비",
        #"IT": "아이티",
        #"K2": "케이투",
        #"ATM": "에이티엠",
        #"CF": "씨에프",
        #"CS": "씨에스",
        #"MVP": "엠브이피",
        #"KBL" : "케이비엘",
        #"BNK" : "비엔케이",
        #"GPT" : "쥐피티",
        #"PK" : "피케이",
        "ME TOO" : "미투",
        "KAI" : "카이",
        "OPEC" : "오펙",


    }

    COMPANY_MAPPING = {
        #"KBS": "케이비에스",
        #"SK": "에스케이",
        #"LG": "엘지",
        #"BMW": "비엠더블유",
        "NASA": "나사",
        #"CNN": "씨엔엔",
        #"IBM": "아이비엠",
        #"UN": "유엔",
        #"EU": "이유",
        #"DRX": "디알엑스",
        #"SKT": "에스케이티",
        #"KT": "케이티",
        #"WHO": "더블유에이치오",
        #"KTB": "케이티비",
        #"KTF": "케이티에프",
        #"PCS": "피씨에스",
        #"BBC": "비비씨",
        #"DER": "디이알",
        "FIFA": "피파",
        #"WWW" : "더블류더블류더블류",
        #"SKG" : "에스케이쥐"
        "KIA" : "기아",

    }

    SINGLE_LETTER_MAPPING = {
        "A": "에이",
        "B": "비",
        "C": "씨",
        "D": "디",
        "E": "이",
        "F": "에프",
        "H": "에이치",
        "G": "지",
        "I": "아이",
        "J": "제이",
        "K": "케이",
        "L": "엘",
        "M": "엠",
        "N": "엔",
        "O": "오",
        "P": "피",
        "Q": "큐",
        "R": "알",
        "S": "에스",
        "T": "티",
        "U": "유",
        "V": "브이",
        "W": "더블유",
        "X": "엑스",
        "Y": "와이",
        "Z": "지",
    }

   
    SPECIAL_SYMBOL_MAPPING = {

        "％": "퍼센트",
        "%p": "퍼센트포인트",
        "% p": "퍼센트포인트",
        "&": "앤",
        "$": "달러",
        "#": "샵",
        "@": "앳",
        "+": "플러스",
        "-": "마이너스",
        "±": "플러스마이너스",
        #"=": "이퀄", #는
        "㎝": "cm",
        "㎜": "mm",
        "㎏": "kg",
        "㎖": "ml",
        "℃": "도",
        "～": "~",
        "ｍ": "m ",
        "㎞": "km",
        "㎎": "mg",

        "_x000D_": "",
        "㎡" : "제곱미터",
        "㎥" : "세제곱미터",
        "코로나 19": "코로나 일구",
        "코로나19": "코로나 일구",
                
        "%": "퍼센트",



    }

    # single korean character mapping
    """
    ㄱ[기역], ㄴ[니은], ㄷ[디귿], ㄹ[리을], ㅁ[미음],

    ㅂ[비읍], ㅅ[시옫], ㅇ[이응], ㅈ[지읃], ㅊ[치읃],

    ㅋ[키윽], ㅌ[티읃], ㅍ[피읍], ㅎ[히읃]
    """
    SINGLE_KOREAN_MAPPING = {
        "ㄱ" : "기역",
        "ㄴ" : "니은",
        "ㄷ" : "디귿",
        "ㄹ" : "리을",
        "ㅁ" : "미음",
        "ㅂ" : "비읍",
        "ㅅ" : "시옫",
        "ㅇ" : "이응",
        "ㅈ" : "지읃",
        "ㅊ" : "치읃",
        "ㅋ" : "키윽",
        "ㅌ" : "티읃",
        "ㅍ" : "피읍",
        "ㅎ" : "히읃"
    }

    #HISTORY_EVENT_MAPPING = 

    def __init__(self, natural=True):
     
        super().__init__(natural)


        self.WORD_MAPPING = {
            **self.UNIT_MAPPING,
            **self.COMMON_ABBR_MAPPING,
            **self.COMPANY_MAPPING,
            #**self.SINGLE_LETTER_MAPPING
            #**self.SPECIAL_SYMBOL_MAPPING
        }

    def apply_special_symbol_mapping(self, text: str) -> str:
    
        for symbol, replacement in self.SPECIAL_SYMBOL_MAPPING.items():
            text = re.sub(re.escape(symbol), replacement, text)
        return text

    def remove_symbols(self, text: str, erase_in_parentheses=True) -> str:
        # Remove all content within parentheses (including parentheses themselves)
        if erase_in_parentheses:
            text = re.sub(r"\([^)]*\)", "", text)

       
        char_map = {
            "<": "", ">": "", "=": "", "[": "", "]": "",
            "《": "", "》": "", "△": "", "＞": "", "＜": "",
            "‘": "", "’": "", "`": "", "”": "", "●": "",
            "≪": "", "≫": "", "「": "", "」": "", "/": "",
            "·": " ", "…": "", "▷": "",
            "(": "", ")": "", "㈜": "", "�": "",
            "ú": "", "◆": "", "ㆍ": "", "\n": "", #"_x000D_": "",

            "": "", "": "",
            "×": "", "°": "", "±": "", "•": "", "™": "",
            "®": "", "©": "",
            "\"": ""

        }

      
        translation_table = str.maketrans(char_map)

#      
        text = text.translate(translation_table)
        return text

    def apply_word_mapping(self, text: str) -> str:
        # Add space between English and Korean words
        text = re.sub(r'([a-zA-Z])([가-힣])', r'\1 \2', text)
        text = re.sub(r'([가-힣])([a-zA-Z])', r'\1 \2', text)

        text = ''.join([self.SINGLE_LETTER_MAPPING.get(c, c) for c in text])

        return text

    def apply_single_korean_mapping(self, text: str) -> str:
        
      
        pattern_seq = r'([' + re.escape(''.join(self.SINGLE_KOREAN_MAPPING.keys())) + r']+)'

        def seq_replacer(match):
            seq = match.group(0)
            
            return ''.join(self.SINGLE_KOREAN_MAPPING.get(char, char) for char in seq)

        
        text = re.sub(pattern_seq, seq_replacer, text)
        return text


    def convert_history_event(self, text: str) -> str:
        history_keys = ['사건','혁명','절','전쟁','선언','운동', '항쟁','독립','민주화', '진상', '정변','군사' ]
        unit_keys    = {u for cat in self.UNIT_CATEGORIES for u in cat.units}

        
        pat = re.compile(r'(?P<num>\d+(?:\.\d+)+)')

        def _repl(m):
            num_dot = m.group('num')
            tail    = text[m.end():]

           
            words = re.findall(r'\b(\S+?)\b', tail)[:3]

            first_tag = None
            for w in words:
               
                if any(w.startswith(u) for u in unit_keys):
                    first_tag = 'unit'
                    break
                
                if any((h in w) for h in history_keys):
                    first_tag = 'history'
                    break

           
            if first_tag == 'history':
                return ''.join(self.NUM_KOR[int(d)] for d in num_dot if d.isdigit())
           
            return num_dot

        return pat.sub(_repl, text)

    def __call__(self, sentence: str) -> str:
        sentence = self.remove_symbols(sentence)
        #print(f"n2kg+ : remove symbols : {sentence}")
        sentence = self.apply_special_symbol_mapping(sentence)
        #print(f"n2gk+ : apply special symbol mapping : {sentence}")
        sentence = self.apply_single_korean_mapping(sentence)
        #print(f"n2gk+ : apply single korean mapping : {sentence}")
        sentence = self.convert_history_event(sentence)
        #print(f"n2gk+ : convert_history_event : {sentence}")
        sentence = super().__call__(sentence)
        #print(f"n2gk+ : call N2gk : {sentence}")
        sentence = self.apply_word_mapping(sentence)
        #print(f"n2gk+ : apply_word_mapping : {sentence}")
        return sentence

    def run_n2gkplus(
            self,
            input_jsonl_path: Union[str, Path],
            output_jsonl_path: Union[str, Path]
    ) -> None:
        """
        Reads JSONL, normalizes each record['text'], adds 'N2gkPlus' field with normalized text,
        and writes to output JSONL.
        """
        input_path = Path(input_jsonl_path)
        output_path = Path(output_jsonl_path)

        records = []
        for line in tqdm(input_path.open('r', encoding='utf-8'), desc='Normalizing (N2gkPlus)'):
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get('text', '')
            data['N2gkPlus'] = self(text)
            records.append(data)

        with output_path.open('w', encoding='utf-8') as outf:
            for rec in records:
                json.dump(rec, outf, ensure_ascii=False)
                outf.write('\n')

        print(f"N2gkPlus-normalized output saved to: {output_path}")
