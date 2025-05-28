# CoreaSpeech: Korean Speech Corpus via JAMO-based Coreset Selection for Efficient and Robust Korean Speech Generation

<div align="center">
  <img width="500px" src="./assets/coreaspeech_logo.png" alt="CoreaSpeech Logo">
</div>

## News
- **2025/05/15**: CoreaSpeech dataset, pipeline code, PEFT-TTS model, and Korean Universal Benchmark released!

## Overview
This repository contains the source code for a pipeline designed for processing Korean speech data. It is the official repository for **CoreaSpeech, a 700-hour Korean speech corpus from 21,449 speakers, refined using a Jamo-based coreset selection pipeline.**

### Key Features
- **Data Conditioning Pipeline**:
    - **Speaker Diarization**: Utilizes `pyannote/speaker-diarization-3.1` for segmenting audio by speaker.
    - **Text Categorization (LNCat)**: Selectively retains utterances based on their convertibility into Korean graphemes (details in `src/module/data_conditioning/categorizing.py`).
    - **Korean Text Normalization (N2gk+)**: Employs **N2gk+** for normalizing numerals, English words, and special characters in Korean text (see `src/module/data_conditioning/normalization.py`).
    - **Audio Feature Extraction**: Extracts audio features (details in `src/module/data_conditioning/audio_feature_extracting.py`).
- **Coreset Selection**:
    - **Jamo-based Selection**: Implements a **Jamo bigram-based strategy** for selecting a phonetically diverse coreset (see `src/module/coreset_selection/core_jamo_selecting.py`).
    - **Audio Quality Filtering**: Filters data based on **dynamic, dataset-specific UTMOS thresholds** (utils in `src/module/coreset_selection/utils.py`).
- **Supplementary Finalization**:
    - **Data Appending (Duration Balancing)**: Balances utterance durations by concatenating short segments from the same speaker (see `src/module/supplementary_finalization/data_appending.py`).

## Project Structure

```
CoreaSpeech/sourcecode/
├── src/
│   ├── dataset/                # Scripts to prepare specific datasets (e.g., Emilia, KSS)
│   ├── module/                 # Core processing modules
│   │   ├── coreset_selection/  # Coreset selection logic
│   │   ├── data_conditioning/  # Data cleaning, normalization, feature extraction
│   │   └── supplementary_finalization/ # Final data processing steps
│   └── run_pipeline.py         # Main script to run the entire pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CoreaSpeech/sourcecode.git
    cd sourcecode
    ```

2.  **Create a Python virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `pyannote.audio` and its related models/dependencies might require careful installation. Please refer to the official `pyannote.audio` documentation for detailed instructions if you encounter issues or need specific diarization models. Some `pyannote` dependencies are commented out in `requirements.txt` and may need to be installed manually.*

4.  **Hugging Face Token (Required for `pyannote.audio`):**
    Some functionalities, particularly speaker diarization using `pyannote.audio`, require a Hugging Face token with `read` access to download pre-trained models. Ensure you have a token and are logged in (`huggingface-cli login`) or provide the token when prompted/configured.
    The `run_pipeline.py` script and `audio_feature_extracting.py` accept an `hf_token` argument.

## Usage

The main pipeline can be executed using `run_pipeline.py`. This script orchestrates the different stages of data processing.

```bash
python src/run_pipeline.py [arguments]
```

**Key Arguments for `run_pipeline.py` (and other scripts):**

(Please refer to the script's argument parser for a complete list and up-to-date details. Below are common ones.)

*   `--emilia_raw_path`: Path to the raw Emilia dataset.
*   `--emilia_output_jsonl`: Path to save the processed Emilia dataset (jsonl format).
*   `--kss_raw_path`: Path to the raw KSS dataset.
*   `--kss_output_jsonl`: Path to save the processed KSS dataset (jsonl format).
*   `--coreset_input_jsonl`: Input jsonl file for coreset selection.
*   `--coreset_output_jsonl`: Output jsonl file after coreset selection.
*   `--hf_token`: Hugging Face token (if needed for models like pyannote).
*   `--utmos_mode`, `--utmos_static_value`, `--utmos_dynamic_type`, etc.: Parameters for UTMOS-based filtering in coreset selection.
*   Many scripts take input and output paths, often for `.jsonl` files or directories.

**Example (Conceptual - adapt paths and arguments as needed):**

```bash
# Example: Preparing Emilia dataset, then KSS, then running coreset selection
python src/dataset/prepare_emilia.py --raw_path /path/to/emilia_raw --output_jsonl data/emilia_prepared.jsonl
python src/dataset/prepare_kss.py --raw_path /path/to/kss_raw --output_jsonl data/kss_prepared.jsonl

# (Combine emilia_prepared.jsonl and kss_prepared.jsonl manually or with a script if needed for coreset input)

python src/module/coreset_selection/core_jamo_selecting.py \
    --input_jsonl data/combined_prepared.jsonl \
    --output_jsonl data/coreset_selected.jsonl \
    --column_name_for_jamo "text" \
    --column_name_for_utmos "utmos_score" \
    --utmos_mode "dynamic" \
    --utmos_dynamic_type "quantile" \
    --utmos_quantile_value 0.1

# The run_pipeline.py script aims to automate these stages.
# Check its arguments and implementation for direct pipeline execution.
python src/run_pipeline.py \
    --emilia_raw_path /path/to/emilia_raw --emilia_output_jsonl data/emilia_processed.jsonl \
    --kss_raw_path /path/to/kss_raw --kss_output_jsonl data/kss_processed.jsonl \
    --coreset_input_jsonl data/combined_for_coreset.jsonl --coreset_output_jsonl data/final_coreset.jsonl \
    --hf_token YOUR_HF_TOKEN
```

*Note: The exact flow and combination of inputs/outputs for `run_pipeline.py` should be verified from its source code, as it may expect specific intermediate files or naming conventions.*

## Modules Overview

*   **`src/dataset/`**: Contains scripts to preprocess and prepare specific public or private speech datasets into a common `.jsonl` format suitable for the pipeline.
    *   `prepare_emilia.py`: Processes the Emilia dataset.
    *   `prepare_kss.py`: Processes the KSS dataset.
*   **`src/module/data_conditioning/`**: Modules for cleaning and standardizing data.
    *   `audio_feature_extracting.py`: Performs speaker diarization using `pyannote/speaker-diarization-3.1` to ensure single-speaker segments.
    *   `categorizing.py`: Implements LNCat for selective text categorization based on convertibility to Korean graphemes.
    *   `normalization.py`: Advanced Korean text normalization using N2gk+, handling numerals, English words, etc.
    *   `speech_tag_enrich.py`: Potentially enriches data with speech-related tags (e.g., from diarization).
*   **`src/module/coreset_selection/`**: Modules for selecting a representative subset of the data.
    *   `core_jamo_selecting.py`: Implements Jamo bigram-based coreset selection and dynamic UTMOS filtering.
    *   `utils.py`: Utility functions, including UTMOS threshold calculation.
*   **`src/module/supplementary_finalization/`**: Scripts for final processing steps.
    *   `data_appending.py`: Balances utterance durations by concatenating short segments from the same speaker.

## License

The CoreaSpeech Dataset, Korean Universal Testset, and model checkpoints are released under the [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

The preprocessing, training, and evaluation code (contained in this repository) are released under the [MIT license](https://opensource.org/licenses/MIT).

## TODO

*   [ ] Finalize and document the exact flow of `run_pipeline.py`.
*   [ ] Add comprehensive unit tests for each module.
*   [ ] Provide a clear `config.yaml` or argument structure for all customizable parameters.
*   [x] Add a suitable open-source license.
*   [ ] Include more detailed examples of how to run the pipeline with sample data (if possible).
*   [ ] Refine `requirements.txt` with specific, tested versions for all libraries.

---
