# CoreaSpeech: Korean Speech Corpus via JAMO-based Coreset Selection for Efficient and Robust Korean Speech Generation

<div align="center">
  <img width="500px" src="./assets/coreaspeech_logo.png" alt="CoreaSpeech Logo">
</div>

## News
- **2025/05/15**: CoreaSpeech dataset, pipeline code, PEFT-TTS model, and Korean Universal Benchmark released!

## Overview
This repository contains the source code for a pipeline designed for processing Korean speech data. It includes modules for data conditioning (diarization, normalization, categorization), coreset selection based on JAMO and audio quality metrics, and supplementary finalization.

### Key Features
- **Data Conditioning Pipeline**:
    - **Speaker Diarization**: Utilizes `pyannote.audio` for segmenting audio by speaker.
    - **Text Categorization**: Categorizes text data (details الوطنية in `src/module/data_conditioning/categorizing.py`).
    - **Korean Text Normalization**: Employs `N2gk` and `N2gkPlus` for normalizing numerals, English words, and special characters in Korean text (see `src/module/data_conditioning/normalization.py`).
    - **Audio Feature Extraction**: Extracts audio features (details in `src/module/data_conditioning/audio_feature_extracting.py`).
- **Coreset Selection**:
    - **JAMO-based Selection**: Implements a JAMO bigram-based strategy for selecting a phonetically diverse coreset (see `src/module/coreset_selection/core_jamo_selecting.py`).
    - **Audio Quality Filtering**: Can filter data based on UTMOS predictions or other quality metrics (utils in `src/module/coreset_selection/utils.py`).
- **Supplementary Finalization**:
    - **Data Appending**: Scripts to append or combine data, potentially for balancing utterance durations (see `src/module/supplementary_finalization/data_appending.py`).

### (Optional) Language & Domain Statistics
| Category        | Description                                  | Details                                      |
|-----------------|----------------------------------------------|----------------------------------------------|
| Language        | Korean                                       | Primarily standard Korean accents            |
| Total Hours     | 700 hours                                   | Selected from ~2000+ hours of raw data       |
| Speakers        | 21,449                                      | Diverse speaker demographics                 |
| Utterance Length| Balanced distribution from 0 to 30 seconds   | Achieved via data appending                  |
| Text Source     | Various public domain and licensed sources   | Normalized and categorized                   |

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
    *Note: `pyannote.audio` and its related models/dependencies might require careful installation. Please refer to the official `pyannote.audio` documentation for detailed instructions if you encounter issues or need specific diarization models. Some `pyannote` dependencies are commented out in `requirements.txt` and may need to be installed પાણી.

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
    *   `audio_feature_extracting.py`: Likely handles speaker diarization (using `pyannote`) and could be extended for other audio features.
    *   `categorizing.py`: Script for text categorization.
    *   `normalization.py`: Advanced Korean text normalization.
    *   `speech_tag_enrich.py`: Potentially enriches data with speech-related tags (e.g., from diarization).
*   **`src/module/coreset_selection/`**: Modules for selecting a representative subset of the data.
    *   `core_jamo_selecting.py`: Implements JAMO-based coreset selection logic.
    *   `utils.py`: Utility functions, including UTMOS threshold calculation.
*   **`src/module/supplementary_finalization/`**: Scripts for final processing steps.
    *   `data_appending.py`: Appends data, possibly to balance utterance durations or combine datasets.

## License

This project is currently not under a specific license. Please specify a license (e.g., MIT, Apache 2.0) if you intend for others to use, modify, or distribute this code.

## TODO

*   [ ] Finalize and document the exact flow of `run_pipeline.py`.
*   [ ] Add comprehensive unit tests for each module.
*   [ ] Provide a clear `config.yaml` or argument structure for all customizable parameters.
*   [ ] Add a suitable open-source license.
*   [ ] Include more detailed examples of how to run the pipeline with sample data (if possible).
*   [ ] Refine `requirements.txt` with specific, tested versions for all libraries.

---
*This README was updated to reflect the current codebase. Further details and specific instructions should be added as the project evolves.* 