# HREM Evaluation Framework

This repository provides a robust and extensible framework for evaluating Hierarchical Recurrent Memory (HREM) models and other baseline neural network architectures. The framework is designed with a focus on rigor, reproducibility, and ease of use, incorporating tools for training, analysis, hyperparameter tuning, and interactive visualization.

## Features

- **Modular Models**: Easily extendable model architecture, with `MLP`, `HRM`, and a modular `HREM` (for ablation studies) included.
- **Flexible Datasets**: Support for multiple tasks, with the sequence reversal task and a character-level language modeling task implemented.
- **Configuration-Driven**: All experiments are defined in simple JSON files, ensuring perfect reproducibility.
- **Robust Training Engine**: A `Trainer` class that handles device management, training, evaluation, logging, and model checkpointing.
- **Automated Testing**: A full suite of unit tests with `pytest` and a CI pipeline with GitHub Actions to ensure code quality.
- **Advanced Hyperparameter Search**: Integrated support for `Optuna` to automate the search for optimal hyperparameters.
- **Interactive Dashboard**: A `Streamlit`-based GUI to browse, visualize, and compare experiment results.
- **Comprehensive Analysis**: Tools for both programmatic and interactive analysis of results, including a pre-built Jupyter notebook.

## Project Structure

The project uses a standard `src`-layout for clean and maintainable code.

```
.
├── .github/workflows/ci.yml  # CI pipeline configuration
├── configs/                  # Experiment configuration files (JSON)
│   └── default.json
├── notebooks/                # Jupyter notebooks for analysis
│   └── Analyze_Results.ipynb
├── results/                  # Output directory for models, logs, and plots
├── src/                      # Main source code
│   ├── __init__.py
│   ├── analysis.py           # Functions for plotting and reporting
│   ├── datasets.py           # Dataset classes
│   ├── models.py             # Model architectures (MLP, HRM, HREM)
│   ├── search.py             # Hyperparameter search script
│   └── training.py           # The core Trainer class
├── tests/                    # Unit tests
├── dashboard.py              # The Streamlit dashboard script
├── hrem.py                   # The original script (kept for reference)
├── main.py                   # Main entry point for running experiments
├── pyproject.toml            # Project configuration
└── requirements.txt          # Project dependencies
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode:**
    This step makes the `src` package available to all scripts and is crucial for the imports to work correctly.
    ```bash
    pip install -e .
    ```

## Usage

### Running Experiments

To run an experiment, use the `main.py` script and provide the path to a configuration file.

```bash
python main.py configs/default.json
```
The results, including the best model, logs, and plots, will be saved in a subdirectory inside `results/`.

### Running Tests

To run the unit test suite, use `pytest`:

```bash
pytest
```

### Running the Interactive Dashboard

To launch the Streamlit dashboard and browse experiment results:

```bash
streamlit run dashboard.py
```

### Running Hyperparameter Search

To start an Optuna hyperparameter search (as defined in `src/search.py`):

```bash
python src/search.py
```

### Analyzing Results

For interactive analysis, you can use the provided Jupyter notebook. First, ensure you have `jupyter` installed (`pip install jupyter`), then launch the notebook server:

```bash
jupyter notebook notebooks/Analyze_Results.ipynb
```

## Configuration

Experiments are configured using JSON files in the `configs/` directory. The structure is as follows:

```json
{
  "experiment_name": "name-of-your-experiment",
  "model": {
    "name": "mlp",
    "params": { ... }
  },
  "dataset": {
    "name": "reverse",
    "params": { ... }
  },
  "training": {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "loss": "bce"
  }
}
```
You can create new config files to define new experiments.
