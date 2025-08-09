# HREM Evaluation Framework

This repository provides a robust and extensible framework for evaluating Hierarchical Recurrent Memory (HREM) models and other baseline neural network architectures. The framework is designed with a focus on rigor, reproducibility, and ease of use, incorporating tools for training, analysis, hyperparameter tuning, and interactive visualization.

## Features

- **Modular Models**: Easily extendable model architecture, with `MLP`, `HRM`, and a modular `HREM` (for ablation studies) included.
- **Flexible Datasets**: Support for multiple tasks, with the sequence reversal task and a character-level language modeling task implemented.
- **Configuration-Driven**: All experiments are defined in simple JSON files, ensuring perfect reproducibility.
- **Robust Training Engine**: A `Trainer` class that handles device management, training, evaluation, logging, and model checkpointing.
- **Automated Testing**: A full suite of unit tests with `pytest` and a CI pipeline with GitHub Actions to ensure code quality.
- **Advanced Hyperparameter Search**: Integrated support for `Optuna` to automate the search for optimal hyperparameters.
- **Interactive GUI**: A PyQt6-based GUI to launch, monitor, and analyze experiments in real-time.
- **Comprehensive Analysis**: Tools for both programmatic and interactive analysis of results.

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
│   ├── models/               # Model architectures (MLP, HRM, HREM)
│   ├── search.py             # Hyperparameter search script
│   └── training.py           # The core Trainer class
├── tests/                    # Unit tests
├── gui.py                    # The PyQt6 GUI application
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

### Running the GUI

The primary way to interact with this framework is through the PyQt6 GUI. To launch it, run:

```bash
python gui.py
```

From the GUI, you can:
- Launch new experiments from config files.
- View a list of all past and running experiments.
- Monitor the learning curves of running experiments in real-time.
- Analyze the results of completed experiments.

### Running Experiments via Command Line

While the GUI is recommended, you can still run experiments directly from the command line. To run a single experiment, use the `main.py` script:

```bash
python main.py configs/default.json
```

To start an Optuna hyperparameter search, use `search.py`:

```bash
python search.py configs/copy_transformer_search.json
```

### Running Tests

To run the unit test suite, use `pytest`:

```bash
pytest
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
    "learning_rate": 0.001
  }
}
```
You can create new config files to define new experiments. The framework also supports a modular configuration system using an `extends` keyword to promote reusability. See the existing configs for examples.
