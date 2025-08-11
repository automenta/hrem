# HREM Evaluation Framework

This repository provides a robust and extensible framework for evaluating Hierarchical Recurrent Memory (HREM) models and other baseline neural network architectures. The framework is designed with a focus on rigor, reproducibility, and ease of use, incorporating tools for training, analysis, hyperparameter tuning, and interactive visualization.

## Features

- **Modular Models**: Easily extendable model architecture, with `MLP`, `HRM`, and a modular `HREM` (for ablation studies) included.
- **Flexible Datasets**: Support for multiple tasks, with the sequence reversal task and a character-level language modeling task implemented.
- **Configuration-Driven**: All experiments are defined in simple JSON files, ensuring perfect reproducibility.
- **Robust Training Engine**: A `Trainer` class that handles device management, training, evaluation, logging, and model checkpointing.
- **Automated Testing**: A full suite of unit tests with `pytest` and a CI pipeline with GitHub Actions to ensure code quality.
- **Advanced Hyperparameter Search**: Integrated support for `Optuna` to automate the search for optimal hyperparameters.
- **Interactive GUI for Experiment-Driven Development**: A PyQt6-based GUI designed to manage, visualize, and analyze experiments. The GUI is built to support an intuitive, semi-autonomous workflow for exploring hyperparameter spaces and model architectures. Key GUI features include:
    - **Paired Experiment Racing**: Launch a 'challenger' model against one or more 'baseline' models. The framework automatically generates matched configurations for a fair, head-to-head comparison on performance and efficiency metrics (parameter count, epoch time).
    - **Research Tree Visualization**: A graph-based view to track the lineage of your experiments, making it easy to see the evolution from one idea to the next.
    - **N-Dimensional Analysis**: A powerful scatter plot view that allows you to visualize all your experiments across different hyperparameters and results. Visually identify trends, outliers, and Pareto frontiers.
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

The GUI is organized into several tabs:

- **Experiments**: The main dashboard. View a filterable, sortable table of all your experiments. Select one or more experiments to view their learning curves, compare their configurations, and manage them (clone, rename, delete).
- **Research Tree**: A "skill tree" for your research. This view shows the parent-child relationships between your experiments, providing an intuitive map of your exploration process.
- **Analysis**: A powerful scatter plot for visualizing the entire experiment space. Plot any hyperparameter or result against another, and use a third metric for color-coding. This view also visually connects experiments that were run as part of a "race", making it easy to see performance gaps.
- **Search**: Manage and monitor `Optuna` hyperparameter searches.

A key workflow is **Paired Experiment Racing**:
1.  Click "Launch New".
2.  In the dialog, select your primary "challenger" configuration file.
3.  Give the race a base name.
4.  Select one or more baseline models (e.g., `lstm`, `transformer`) to race against.
5.  Launch the race. The system will automatically create and run experiments for your challenger and all selected baselines with matching training and dataset parameters.

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
