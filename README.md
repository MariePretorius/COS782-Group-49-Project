# COS760 / COS782 Group 49 Project

This repository contains code and experiment outputs for detecting machine-generated text in African languages. The project builds human-vs-machine datasets for Afrikaans, English, and isiZulu, then evaluates both traditional TF-IDF classifiers and transformer models.

## Project Layout

- `Code/src/` contains the Python scripts for dataset creation, model training, evaluation, and analysis.
- `Code/data/` contains raw/generated inputs and processed train, validation, and test CSV files.
- `Code/outputs/` contains generated model outputs from runs.
- `Code/experiment_results/` contains saved result summaries used for reporting.

## Setup

Run the commands from the `Code` directory:

```bash
cd Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The transformer experiments require `torch` and Hugging Face model downloads, so they may take longer and need an internet connection the first time they run.

## Run the Code

Rebuild the basic train/test dataset:

```bash
python src/build_basic_dataset.py
```

Run the basic TF-IDF baselines:

```bash
python src/run_basic_experiments.py
```

Rebuild the enhanced train/validation/test dataset:

```bash
python src/build_enhanced_dataset.py
```

Run the enhanced traditional ML experiments:

```bash
python src/run_enhanced_experiments.py
```

Run a transformer experiment, for example AfroXLM-R:

```bash
python src/run_enhanced_model.py --model Davlan/afro-xlmr-base-76L_script --tag afroxlmr
```

Useful optional environment variables for transformer runs:

```bash
COS760_NUM_TRAIN_EPOCHS=3 COS760_TRAIN_BATCH_SIZE=8 COS760_EVAL_BATCH_SIZE=16 python src/run_enhanced_model.py --model Davlan/afro-xlmr-base-76L_script --tag afroxlmr
```

Outputs are written under `Code/outputs/`, including predictions, metrics, per-language metrics, error summaries, and comparison CSV files.

## Short Explanation

The code combines human text samples with machine-generated paraphrases and labels each row as either `human` or `machine`. The basic experiment uses train/test splits, while the enhanced experiment adds a validation split for model and threshold selection. Traditional models use word and character TF-IDF features with logistic regression or linear SVM classifiers. Transformer scripts fine-tune Hugging Face sequence-classification models and evaluate them overall and per language.
