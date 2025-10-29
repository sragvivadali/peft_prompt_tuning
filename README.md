# PEFT Prompt Tuning

## Overview

This repository contains implementations and experiments for Parameter-Efficient Fine-Tuning (PEFT) of large language models via prompt tuning techniques. The project uses Python and Jupyter Notebooks to explore methods that fine-tune pre-trained models efficiently without full retraining.

## Features

- Demonstrations of prompt tuning techniques leveraging PEFT methodologies.
- Jupyter Notebook-based experiments to visualize and evaluate model tuning.
- Python scripts and utilities to implement tuning pipelines.
- Lightweight approach to adapt large models with reduced computational resources.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages: transformers, datasets, torch (customize based on your environment)

### Installation

1. Clone the repository:
```
git clone https://github.com/sragvivadali/peft_prompt_tuning.git
cd peft_prompt_tuning
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Launch Jupyter Notebook:
```
jupyter notebook
```

4. Open and run notebooks to explore prompt tuning experiments.

## How to Run

1. Prepare your dataset files (`train.csv`, `eval.csv`, and `test.csv`) inside a folder (default is `./benchmark`).

2. Run the main training and evaluation script from the command line:
```
python main.py
--peft PTuning
--model 7b
--query imputation
-f ./benchmark
-p "Predict the missing values"
-t "corrupted data"
-l "gt values"
--train 100
--test 50
--eval 50
--pred 50
```


Replace arguments as needed:
- `--peft`: Choose `PTuning` or `LoRA`
- `--model`: Model size, e.g., `7b` or `13b`
- `--query`: Task type like `imputation` or `extrapolation`
- `-f / --folder`: Dataset folder path
- `-p / --prompt`: Initial prompt text for tuning
- `-t / --text`: Dataset column name for input text
- `-l / --label`: Dataset column name for ground truth labels
- `--train`, `--test`, `--eval`, `--pred`: Number of samples for each phase

3. Monitor training progress, evaluation metrics, and generated output files in the designated folder.


## Usage

- Explore the notebooks to understand different aspects of prompt tuning.
- Modify parameters and data within the notebooks to test on your own datasets.
- Use included Python scripts for automated tuning workflows.

## Contributing

Contributions are welcome! Feel free to fork the repo, open issues, or submit pull requests to enhance tutorials, add models, or improve documentation.


## Contact

For questions, issues, or collaborations, please reach out via GitHub or email.

