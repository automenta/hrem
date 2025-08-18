# HREM Evaluation Framework

This repository provides a robust and extensible framework for evaluating Hierarchical Recurrent Memory (HREM) models and other baseline neural network architectures. The framework is designed with a focus on rigor, reproducibility, and ease of use, incorporating tools for training, analysis, hyperparameter tuning, and interactive visualization.

## Features

### Core Framework Features

- **Modular Models**: Easily extendable model architecture, with `MLP`, `HRM`, and a modular `HREM` (for ablation studies) included.
- **Flexible Datasets**: Support for multiple tasks, with the sequence reversal task and a character-level language modeling task implemented.
- **Configuration-Driven**: All experiments are defined in simple JSON files, ensuring perfect reproducibility.
- **Robust Training Engine**: A `Trainer` class that handles device management, training, evaluation, logging, and model checkpointing.
- **Automated Testing**: A full suite of unit tests with `pytest` and a CI pipeline with GitHub Actions to ensure code quality.
- **Advanced Hyperparameter Search**: Integrated support for `Optuna` to automate the search for optimal hyperparameters.
- **Comprehensive Analysis**: Tools for both programmatic and interactive analysis of results.

### GUI Features

The current version of the GUI provides a complete workflow for **Experiment Racing**. This includes:
- **Race Launcher**: A dedicated dialog to configure and launch a "race," pitting a challenger model against one or more baselines under identical conditions.
- **Configuration Editor**: An integrated tool to view or make temporary, in-memory modifications to a challenger's configuration before launching a race. The editor validates inputs to prevent configuration errors.
- **Live Race Monitor**: A dashboard that provides a live, side-by-side comparison of all race participants, with real-time plots of training/testing loss, and access to configurations and logs.

## Project Structure

The project uses a standard `src`-layout for clean and maintainable code.

```
.
├── .github/workflows/ci.yml  # CI pipeline configuration
├── configs/                  # Experiment configuration files (JSON)
│   ├── base/                 # Base configs for models and datasets
│   └── default.json
├── results/                  # Output directory for models, logs, and plots
├── src/                      # Main source code
│   ├── gui/                  # Source for the PyQt6 GUI
│   │   ├── config_editor.py
│   │   ├── experiment_manager.py
│   │   ├── race_launcher.py
│   │   └── race_monitor.py
│   ├── models/               # Model architectures
│   ├── analysis.py
│   ├── datasets.py
│   ├── search.py
│   └── training.py
├── tests/                    # Unit tests
├── gui.py                    # Entry point for the GUI application
├── main.py                   # Entry point for command-line experiments
└── pyproject.toml            # Project configuration
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
    The project and its development dependencies can be installed with a single command.
    ```bash
    pip install -e .[dev]
    ```
    **Note on PyTorch:** This project uses PyTorch. For some systems, you may need to install it with a specific index URL to get the correct version for your hardware (e.g., CPU-only or a specific CUDA version). If you have issues, please see the [PyTorch website](https://pytorch.org/get-started/locally/) for instructions and install it separately before running the command above. For example:
    ```bash
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
    ```


## Usage

The primary way to interact with this framework is through the PyQt6 GUI.

### Running the GUI

To launch the application, run:

```bash
python gui.py
```

This will open the **Race Launcher** window, which is the starting point for running experiments.

### The "Experiment Race" Workflow

A key workflow is the **Paired Experiment Race**. This paradigm is a powerful method for rigorous algorithm evaluation. Instead of comparing a new model to a generic, pre-existing baseline, a "race" puts a "challenger" model head-to-head against one or more baseline architectures on a specific task. The framework ensures a fair comparison by using the *exact same* training and dataset parameters for all participants, isolating the architectural differences.

This approach is not just about winning; it's about learning. By racing algorithms against each other under controlled conditions, you can build a deep understanding of the trade-offs between model complexity, efficiency, and performance.

### How to Launch a Race

1.  **Run the GUI** with `python gui.py`.
2.  In the **Race Launcher** window, fill out the fields:
    - **Race Name:** A descriptive name for your race (e.g., `my-hrem-vs-lstm`).
    - **Task / Dataset:** Select the dataset all models will be trained on.
    - **Challenger Model:** Click "Select Config..." to choose your main model's configuration file.
    - **View/Edit...:** After selecting a config, you can use this button to view or make temporary, in-memory changes to the challenger's configuration for this specific race. This is useful for quick experiments without creating new files.
    - **Race Against Baselines:** Select one or more baseline models (e.g., `lstm`, `transformer`) to race against.
    - **Notes:** Add any notes about the race.
3.  Click **"Launch Race"**.

This will close the launcher and open the **Race Monitor** window.

### Monitoring the Race

The **Race Monitor** provides a live look at the experiments as they run.
- Each participant (challenger and baselines) gets its own plot showing its training and testing loss curves.
- If there is an error loading a model's results, a clear error message will be displayed in place of its plot.
- For each participant, you can click:
    - **"View Config"** to see the exact configuration used for that run in a read-only viewer.
    - **"View Log"** to see the raw stdout/stderr log file for the training process.
- A summary table at the bottom shows the live status, latest test loss, and notes (e.g., "Winning", "Losing") for all participants.
- Closing the Race Monitor window will stop all associated training processes.

## GUI Roadmap & Future Vision

The current GUI provides a robust workflow for launching and monitoring head-to-head experiment races. Our long-term vision is to expand this into a comprehensive Experiment Management System. The following features are planned for future releases:

- **Mission Control**: A main dashboard to view, filter, sort, and manage all past experiments (e.g., cloning, archiving, deleting).
- **Research Tree**: A graph-based view to track the lineage of your experiments, making it easy to see the evolution from one idea to the next.
- **N-Dimensional Analysis**: A powerful scatter plot view that allows you to visualize all your experiments across different hyperparameters and results to visually identify trends, outliers, and Pareto frontiers.

These features will build upon the current foundation to create a truly intuitive and powerful tool for machine learning research.

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
PYTHONPATH=. pytest
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
