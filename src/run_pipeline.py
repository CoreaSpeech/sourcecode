from pathlib import Path
from module.data_conditioning.audio_feature_extracting import AudioFeatureExtractor
from module.data_conditioning.categorizing import LNCat
from module.data_conditioning.normalization import N2gkPlus
from module.coreset_selection.core_jamo_selecting import JamoBigram
from module.supplementary_finalization.data_appending import DataAppender

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HF_TOKEN          = "hf_YOUR_TOKEN_HERE"
USE_NATURAL       = True

# your *raw* input JSONLs
RAW_JSONLS        = [
    "../data/kss/kss.jsonl",
    "../data/emilia/emilia.jsonl",
    # …
]

# after phase1, we’ll merge all these normalized files here:
MERGED_NORMALIZED = Path("../data/all_normalized_merged.jsonl")
# where we’ll dump the global jamo counts CSV:
GLOBAL_CSV        = Path("../data/total_jamo_counts.csv")
# ────────────────────────────────────────────────────────────────────────────────

def phase1_and_merge() -> list[Path]:
    """Run steps 1–3 on each RAW_JSONLS and merge all the *_normalized.jsonl into one file."""
    normalized_paths: list[Path] = []

    for raw in RAW_JSONLS:
        inp  = Path(raw)
        stem = inp.stem
        print(f"\n>>> Phase1 on {stem}.jsonl")

        # 1) Audio‐UTMOS & speaker count enrichment
        print("Step 1: Audio‐UTMOS & speaker count enrichment")
        feat = inp.with_name(f"{stem}_features.jsonl")
        AudioFeatureExtractor(hf_token=HF_TOKEN) \
            .run_enrichment(str(inp), str(feat))

        feat = inp.with_name(f"{stem}_features.jsonl")
        
        # 2) LNCat + in‐place category filtering
        print("Step 2: LNCat + in‐place category filtering")
        cat = inp.with_name(f"{stem}_categorized.jsonl")
        LNCat().run_categorization(str(feat), str(cat))

        # 3) N2gkPlus normalization
        print("Step 3: N2gkPlus normalization")
        norm = inp.with_name(f"{stem}_normalized.jsonl")
        N2gkPlus(natural=USE_NATURAL) \
            .run_n2gkplus(str(cat), str(norm))

        normalized_paths.append(norm)

    # merge all normalized files into one
    print(f"\n>>> Merging {len(normalized_paths)} normalized JSONLs → {MERGED_NORMALIZED}")
    MERGED_NORMALIZED.parent.mkdir(parents=True, exist_ok=True)
    with MERGED_NORMALIZED.open("w", encoding="utf-8") as out:
        for fn in normalized_paths:
            for line in fn.open("r", encoding="utf-8"):
                out.write(line)
    return normalized_paths


def build_global_jamo_csv():
    """
    Read the single merged normalized JSONL, compute every
    record’s JamoBigram counts (accumulating into self.total_…),
    and dump those global counts out to GLOBAL_CSV.
    """
    print("\n>>> Computing + saving global JamoBigram counts …")
    # we'll write an intermediate file with per‐record JamoBigram if you like,
    # but only the CSV is strictly needed:
    with_jamo = MERGED_NORMALIZED.with_name("all_with_jamo.jsonl")

    jam = JamoBigram(
        total_table_path=str(GLOBAL_CSV),     # where to save the global counts
        num_workers=4
    )
    # this will read MERGED_NORMALIZED, annotate every line with "JamoBigram", update
    # jam.total_jamo_pair_count, write out with_jamo, and then jam.save_total_counts() → GLOBAL_CSV
    jam.apply_jamobigram(str(MERGED_NORMALIZED), str(with_jamo))


def phase2_selection_and_appending(norm_paths: list[Path]):
    """
    For each of your per‐dataset *_normalized.jsonl:
      4) load jamo‐counts from GLOBAL_CSV and filter → *_selected.jsonl
      5) data appending / wav concatenation → *_appended.jsonl
    """
    for norm in norm_paths:
        stem     = norm.stem.replace("_normalized", "")
        jbapplied = norm.with_name(f"{stem}_jbapplied.jsonl")
        selected = norm.with_name(f"{stem}_selected.jsonl")
        appended = norm.with_name(f"{stem}_appended.jsonl")

        print(f"\n>>> Phase2 on {norm.name}")
        print("Stpe 3: Jamo Bigram Applying each")
        # 3) Jamo Bigram applying
        jam = JamoBigram(
            total_table_path=str(norm.with_name(f"{stem}_total_table.csv")),     # where to save the global counts
            num_workers=4
            )
        jam.apply_jamobigram(str(norm), str(jbapplied))
        # 4) core‐set filtering (loads GLOBAL_CSV under the hood)
        print("Step 4: core-set filtering")
        JamoBigram(
            t=500,
            beta=0.0001,
            csv_total_table=str(GLOBAL_CSV),
            num_workers=4
        ).run_selection(str(jbapplied), str(selected))

        # 5) create appended audio + JSONL
        print("Step: Data Appending")
        DataAppender().run_appending(str(selected), str(appended))


if __name__ == "__main__":
    # Phase 1: enrichment → categorize → normalize → merge
    #norm_list = phase1_and_merge()

    norm_list = [
        Path("../data/emilia/emilia_normalized.jsonl"),
        Path("../data/kss/kss_normalized.jsonl")
    ]
    # Build global counts CSV from merged normalized data
    build_global_jamo_csv()

    # Phase 2: selection & appending on each normalized file
    phase2_selection_and_appending(norm_list)

    print("\nAll done!") 
