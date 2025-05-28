from pathlib import Path
import json
import numpy as np
import jieba
from jamo import hangul_to_jamo
from pypinyin import lazy_pinyin, Style

try:
    from scipy.stats import median_abs_deviation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def convert_char_to_jamo(text_list, polyphone=True):
    
    final_text_list = []
    god_knows_why_en_testset_contains_zh_quote = str.maketrans(
        {""": '"', """: '"', "'": "'", "'": "'"}
    )
    custom_trans = str.maketrans({";": ","})

    for text in text_list:
        char_list = []
        ko_buffer = []  
        text = text.translate(god_knows_why_en_testset_contains_zh_quote)
        text = text.translate(custom_trans)

        for seg in jieba.cut(text):  
            seg_byte_len = len(bytes(seg, "UTF-8"))

            if all("가" <= char <= "힣" for char in seg): 
                if not ko_buffer and char_list and char_list[-1] != " ":
                    char_list.append(" ")
                ko_buffer.append(seg)

            else:  
                if ko_buffer:  
                    joined_ko = "".join(ko_buffer)  
                    ko_buffer = []  
                    #converted_ko = g2p(joined_ko)
                    converted_ko = hangul_to_jamo(joined_ko)
                    char_list.extend(list(converted_ko))

                if seg_byte_len == len(seg):  
                    if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                        char_list.append(" ")
                    char_list.extend(seg)
                elif polyphone and seg_byte_len == 3 * len(seg):  
                    seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                    for c in seg:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                        char_list.append(c)
                else:  
                    for c in seg:
                        if ord(c) < 256:
                            char_list.extend(c)
                        else:
                            if c not in "。，、；：？！《》【】—…":
                                char_list.append(" ")
                                char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                            else:  
                                char_list.append(c)

        if ko_buffer:
            joined_ko = "".join(ko_buffer)
            #converted_ko = g2p(joined_ko)
            converted_ko = hangul_to_jamo(joined_ko)
            char_list.extend(list(converted_ko))

        final_text_list.append(char_list)

    return final_text_list


def calculate_utmos_threshold(jsonl_path: Path,
                               mode: str = "dynamic",
                               static_value: float = 3.5,
                               dynamic_type: str = "mad",
                               x: float = 0.5,
                               q: float = 0.60,
                               k_max: float = 0.0,
                               k_min: float = 2.5,
                               mu_ref: float = 3.0) -> float:
    """
    Calculates the filtering threshold based on the 'utmos' distribution within jsonl_path.
    
    mode:
        - "static"  : Returns the given static_value as is.
        - "dynamic" : Calculation based on distribution (uses dynamic_type).
    dynamic_type (if mode == "dynamic"):
        - "mu+xsigma" : τ = μ - x·σ
        - "quantile"  : τ = F^{-1}(q)
        - "mad"       : τ = median - k·MAD
    """

    if mode == "static":
        print(f"[UTMOS Threshold] Static mode. Returning: {static_value}")
        return static_value

    # Read UTMOS scores
    scores = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "utmos" in data:
                scores.append(float(data["utmos"]))
    if not scores:
        raise ValueError(f"⚠️ No 'utmos' values found in {jsonl_path}")

    scores = np.array(scores)
    mu, sig = scores.mean(), scores.std()
    print(f"[UTMOS Threshold] Dynamic mode. μ: {mu:.4f}, σ: {sig:.4f}")

    if dynamic_type == "mu+xsigma":
        return float(mu - x * sig)

    elif dynamic_type == "quantile":
        return float(np.quantile(scores, q))

    elif dynamic_type == "mad":
        if mu >= mu_ref:
            k = k_max
        else:
            ratio = (mu_ref - mu) / mu_ref
            k = max(k_min, k_max * (1 - ratio))
            print(f"[UTMOS Threshold] Dynamic k: {k:.4f}")

        med = np.median(scores)
        if _HAS_SCIPY:
            mad = median_abs_deviation(scores, scale="normal")
        else:
            mad = np.median(np.abs(scores - med)) * 1.4826

        return float(med - k * mad)

    else:
        raise ValueError("Invalid dynamic_type: choose 'mu+xsigma', 'quantile', or 'mad'")