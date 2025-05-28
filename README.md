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
    *Note: `pyannote.audio` installation can be complex. Refer to its official documentation for detailed setup and troubleshooting.*

4.  **Hugging Face Token (for `pyannote.audio`):**
    Ensure `pyannote.audio` can access your Hugging Face token. This is typically done via `huggingface-cli login` or by setting the `HF_TOKEN` variable directly within `src/run_pipeline.py` (as detailed in the Usage section for `run_pipeline.py`).

## Usage

The main pipeline can be executed using `src/run_pipeline.py`. This script orchestrates the different stages of data processing and is primarily configured internally (see the example below for `run_pipeline.py`).

Individual scripts used within the pipeline (like `prepare_*.py`, `core_jamo_selecting.py`, `audio_feature_extracting.py`) accept various command-line arguments.

**Key Arguments for Individual Data Processing Scripts:**

(Please refer to each script's argument parser or source code for a complete and up-to-date list. Below are common examples.)

*   `--input_jsonl`, `--output_jsonl`: Paths for input and output `.jsonl` files.
*   `--raw_path`: Path to raw dataset files or directories.
*   `--hf_token`: Hugging Face token (may be required by scripts using Hugging Face models like `pyannote.audio`).
*   `--column_name_for_jamo`, `--column_name_for_utmos`: Specific column names in data files.
*   `--utmos_mode`, `--utmos_static_value`, `--utmos_dynamic_type`, etc.: Parameters for UTMOS-based filtering.
*   Many scripts take various other paths and configuration parameters.

**Example (Conceptual - adapt paths and arguments as needed):**

The following examples illustrate individual steps of the data processing pipeline. The `src/run_pipeline.py` script is designed to automate these stages.

1.  **Preparing Individual Datasets:**
    ```bash
    python src/dataset/prepare_emilia.py --raw_path /path/to/emilia_raw --output_jsonl data/emilia_prepared.jsonl
    python src/dataset/prepare_kss.py --raw_path /path/to/kss_raw --output_jsonl data/kss_prepared.jsonl
    ```
    *(You might need to combine the outputs of these preparation scripts into a single file for the next steps if processing multiple datasets.)*

2.  **Running Coreset Selection (Example):**
    ```bash
    python src/module/coreset_selection/core_jamo_selecting.py \
        --input_jsonl data/combined_prepared.jsonl \
        --output_jsonl data/coreset_selected.jsonl \
        --column_name_for_jamo "text" \
        --column_name_for_utmos "utmos_score" \
        --utmos_mode "dynamic" \
        --utmos_dynamic_type "quantile" \
        --utmos_quantile_value 0.1
    ```

3.  **Using the Automated Pipeline (`run_pipeline.py`):**

    The `src/run_pipeline.py` script automates the entire process from raw data to the final coreset. To use it:
    *   **Modify the `CONFIG` section** at the top of `src/run_pipeline.py` to specify your input dataset paths (e.g., `RAW_JSONLS`) and desired output paths.
    *   **Set your Hugging Face token** by editing the `HF_TOKEN` variable within the script if you haven't configured it globally. This is required for `pyannote.audio` (speaker diarization).
    *   Then, run the script:
        ```bash
        python src/run_pipeline.py
        ```

*Note: Always review the `CONFIG` section and internal logic of `src/run_pipeline.py` to understand its exact behavior and ensure it aligns with your data and environment before execution.*

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
