# HREM Evaluation Framework

This repository provides a robust and extensible framework for evaluating Hierarchical Recurrent Memory (HREM) models and other baseline neural network architectures. The framework is designed with a focus on rigor, reproducibility, and ease of use, incorporating tools for training, analysis, hyperparameter tuning, and interactive visualization.

## Features

### Core Framework Features

-   **Modular Models**: Easily extendable model architecture, with `MLP`, `HRM`, and a modular `HREM` (for ablation studies) included.
-   **Flexible Datasets**: Support for multiple tasks, with the sequence reversal task and a character-level language modeling task implemented.
-   **Configuration-Driven**: All experiments are defined in simple JSON files, ensuring perfect reproducibility.
-   **Robust Training Engine**: A `Trainer` class that handles device management, training, evaluation, logging, and model checkpointing.
-   **Automated Testing**: A full suite of unit tests with `pytest` and a CI pipeline with GitHub Actions to ensure code quality.
-   **Advanced Hyperparameter Search**: Integrated support for `Optuna` to automate the search for optimal hyperparameters.
-   **Comprehensive Analysis**: Tools for both programmatic and interactive analysis of results.

### GUI: The Experiment Dashboard

The framework includes a powerful PyQt6-based graphical user interface, the **Experiment Dashboard**, which serves as a mission control for all your research. It provides a comprehensive suite of tools to manage, visualize, and analyze your experiments.

-   **Unified Experiment View**: See all your experiments in a filterable and sortable table. View their status, model, dataset, and final results at a glance.
-   **Research Tree**: Visualize the lineage of your experiments in an intuitive tree view. Easily track how ideas evolved, which experiments were cloned from others, and the overall structure of your research.
-   **N-Dimensional Analysis**: A powerful scatter plot view that allows you to visualize all your completed experiments across different hyperparameters (e.g., `learning_rate`, `model.params.hidden_size`) and results (`final_loss`). This tool helps you visually identify trends, outliers, and Pareto frontiers.
-   **Wizard-Driven Race Creation**: A guided, multi-step wizard makes it easy to set up new "races." Pit a "challenger" model against multiple baselines, select a dataset, and even override training parameters for a fair, head-to-head comparison.
-   **Live Race Monitor**: Once a race is launched, a dedicated monitor provides a live, side-by-side comparison of all participants with real-time plots of training/testing loss.
-   **Rich Experiment Management**: Right-click on any experiment to access a context menu with actions like:
    -   Renaming, cloning, or archiving.
    -   Viewing logs in a dedicated, searchable log viewer.
    -   Stopping running experiments.
    -   Deleting results.
-   **Archive Browser**: View and restore previously archived experiments.

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
│   │   ├── experiment_dashboard.py
│   │   ├── race_wizard.py
│   │   └── race_monitor.py
│   ├── models/               # Model architectures
│   ├── ...
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

This will open the **Experiment Dashboard**, which is the central hub for managing your experiments.

### The Experiment Dashboard

The dashboard provides three main views, accessible via tabs:

1.  **📋 Table View**: This is the default view. It shows a table of all your experiments. You can filter by name, status, or dataset, and sort by any column. Right-click an experiment to open the context menu.
2.  **🌳 Tree View**: This view organizes your experiments hierarchically. If you clone an experiment, the new one will appear as a child of the original, allowing you to track your research history.
3.  **📈 Analysis View**: This powerful tool lets you create scatter plots to explore the relationship between hyperparameters and results for all *completed* experiments. Select different parameters for the X and Y axes, and even use a third parameter to control the size of the points. Hover over a point to see the experiment's name and details.

### The "Experiment Race" Workflow

A key workflow is the **Paired Experiment Race**. This paradigm is a powerful method for rigorous algorithm evaluation. Instead of comparing a new model to a generic, pre-existing baseline, a "race" puts a "challenger" model head-to-head against one or more baseline architectures on a specific task.

### How to Launch a Race

1.  From the **Experiment Dashboard**, click the **" Launch New Race"** button.
2.  This opens the **Race Creation Wizard**, which will guide you through the setup:
    -   **Race Details**: Give your race a unique name and add descriptive notes.
    -   **Challenger Model**: Define your main "challenger." You can create a new configuration from a template, or select an existing one from a file. You can also edit the configuration in-memory for this specific race.
    -   **Select Baselines**: Choose one or more standard models (e.g., `lstm`, `transformer`) to race against.
    -   **Training Configuration**: Select the dataset for the race. You can also override training parameters like epochs or learning rate for *all* participants to ensure a fair comparison.
    -   **Summary**: Review all your settings before launching.
3.  Click **"Finish"** to launch the race.

This will close the wizard and automatically open the **Race Monitor**.

### Monitoring the Race

The **Race Monitor** provides a live look at the experiments as they run.
-   Each participant (challenger and baselines) gets its own plot showing its training and testing loss curves.
-   You can view the configuration or log file for any participant.
-   Closing the Race Monitor window will stop all associated training processes.

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
