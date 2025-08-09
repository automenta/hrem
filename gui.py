import json
import os
import subprocess
import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QLabel,
    QPushButton,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QTimer


# --- Constants ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
REFRESH_INTERVAL_MS = 2000
LAUNCH_DELAY_MS = 500
INITIAL_SPLITTER_SIZES = [300, 900]
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"

# Experiment Statuses
STATUS_RUNNING = "Running"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"
STATUS_UNKNOWN = "Unknown"


class ExperimentManager:
    """
    Handles the logic for launching, monitoring, and loading experiment results.
    """

    def __init__(self):
        self.processes = {}  # Tracks running subprocesses: {exp_name: Popen_obj}

    def launch_experiment(self, config_path):
        """
        Launches an experiment in a new process and tracks it.
        """
        # Extract a unique name for the experiment from the config path
        base_name = os.path.basename(config_path)
        exp_name = base_name.replace(".json", "")

        process = subprocess.Popen(
            [sys.executable, "main.py", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes[exp_name] = process

    def stop_experiment(self, exp_name):
        """
        Stops a running experiment process.
        """
        if exp_name in self.processes:
            self.processes[exp_name].terminate()
            return True
        return False

    def get_experiment_statuses(self):
        """
        Checks the status of all experiments and returns a dictionary.
        """
        statuses = {}

        # First, get all experiment directories on disk
        if os.path.exists(RESULTS_DIR):
            for exp_name in os.listdir(RESULTS_DIR):
                if os.path.isdir(os.path.join(RESULTS_DIR, exp_name)):
                    # Default status for experiments on disk is Completed
                    statuses[exp_name] = STATUS_COMPLETED

        # Now, check the tracked processes for more accurate statuses
        finished_processes = []
        for exp_name, process in self.processes.items():
            return_code = process.poll()
            if return_code is None:
                statuses[exp_name] = STATUS_RUNNING
            else:
                finished_processes.append(exp_name)
                if return_code == 0:
                    statuses[exp_name] = STATUS_COMPLETED
                else:
                    statuses[exp_name] = STATUS_FAILED

        # Clean up finished processes from the tracking dict
        for exp_name in finished_processes:
            del self.processes[exp_name]

        return statuses

    def load_experiment_results(self, exp_name):
        """
        Loads results for an experiment.
        Returns a tuple: (data, error_message).
        """
        live_file = os.path.join(RESULTS_DIR, exp_name, "live.json")
        results_file = os.path.join(RESULTS_DIR, exp_name, "results.json")

        def _load_json(path):
            try:
                with open(path, "r") as f:
                    return json.load(f), None
            except json.JSONDecodeError:
                return None, f"Error: Corrupted JSON file at {path}"
            except IOError:
                return None, f"Error: Could not read file at {path}"

        if os.path.exists(live_file):
            data, err = _load_json(live_file)
            if err:
                return None, err
            # Live file has a different structure
            return data.get("full_results", {}), None

        if os.path.exists(results_file):
            return _load_json(results_file)

        return None, None  # No results found, but not an error

    def load_experiment_config(self, exp_name):
        """
        Loads the config for a given experiment.
        Returns a tuple: (config_data, error_message).
        """
        config_file = os.path.join(RESULTS_DIR, exp_name, "config.json")
        if not os.path.exists(config_file):
            return None, "Config file not found."

        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
                return config_data, None
        except (json.JSONDecodeError, IOError) as e:
            return None, f"Error reading config: {e}"


class MainGUI(QMainWindow):
    """
    The main window for the experiment GUI.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experimentation Platform")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.manager = ExperimentManager()

        self._init_ui()
        self.refresh_ui()

        # --- Timer for live updates ---
        self.timer = QTimer()
        self.timer.setInterval(REFRESH_INTERVAL_MS)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start()

    def _init_ui(self):
        """
        Initializes the UI components.
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Experiments"))
        self.exp_list_widget = QListWidget()
        self.exp_list_widget.itemSelectionChanged.connect(
            self.update_selected_experiment_display
        )
        left_layout.addWidget(self.exp_list_widget)
        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.refresh_ui)
        left_layout.addWidget(self.refresh_button)
        self.launch_button = QPushButton("Launch New Experiment")
        self.launch_button.clicked.connect(self.launch_new_experiment)
        self.stop_button = QPushButton("Stop Experiment")
        self.stop_button.clicked.connect(self.stop_selected_experiment)
        self.stop_button.setEnabled(False)  # Disabled by default
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.launch_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)
        splitter.addWidget(left_panel)

        # --- Right Panel (Splitter for Plot and Config) ---
        right_panel = QSplitter(Qt.Orientation.Vertical)
        self.plot_widget = pg.PlotWidget()
        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setFontFamily("monospace")
        right_panel.addWidget(self.plot_widget)
        right_panel.addWidget(self.config_display)
        right_panel.setSizes([600, 200])  # Initial sizes for plot and config view
        splitter.addWidget(right_panel)
        splitter.setSizes(INITIAL_SPLITTER_SIZES)

    def launch_new_experiment(self):
        """
        Opens a file dialog to select a config file and launches the experiment.
        """
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Select Experiment Config", CONFIGS_DIR, "JSON files (*.json)"
        )
        if config_path:
            self.manager.launch_experiment(config_path)
            # A small delay to allow the experiment to create its directory
            QTimer.singleShot(LAUNCH_DELAY_MS, self.populate_experiment_list)

    def populate_experiment_list(self):
        """
        Populates the experiment list with names and statuses.
        """
        current_selection = (
            self.exp_list_widget.currentItem().text().split(" ")[0]
            if self.exp_list_widget.currentItem()
            else None
        )
        self.exp_list_widget.clear()
        statuses = self.manager.get_experiment_statuses()
        for exp_name in sorted(statuses.keys()):
            status = statuses[exp_name]
            item_text = f"{exp_name} ({status})"
            self.exp_list_widget.addItem(item_text)
            if exp_name == current_selection:
                self.exp_list_widget.setCurrentRow(self.exp_list_widget.count() - 1)

    def refresh_ui(self):
        """
        Refreshes the entire UI by updating the list and the display.
        """
        self.populate_experiment_list()
        self.update_selected_experiment_display()

    def update_selected_experiment_display(self):
        """
        Displays the results of the selected experiment.
        """
        current_item = self.exp_list_widget.currentItem()
        self.plot_widget.clear()
        self.config_display.clear()

        if not current_item:
            self.plot_widget.setTitle("No experiment selected")
            self.stop_button.setEnabled(False)
            return

        exp_name = current_item.text().split(" ")[0]

        # Update plot
        results_data, res_error = self.manager.load_experiment_results(exp_name)
        if res_error:
            QMessageBox.warning(self, "Result file error", res_error)
        elif results_data:
            train_loss = results_data.get("train_loss", [])
            test_loss = results_data.get("test_loss", [])

            self.plot_widget.setTitle(f"Learning Curves: {exp_name}")
            self.plot_widget.plot(train_loss, pen="b", name="Train Loss")
            self.plot_widget.plot(test_loss, pen="r", name="Test Loss")
            self.plot_widget.addLegend()
        else:
            self.plot_widget.setTitle(f"No results available for: {exp_name}")

        # Update config view
        config_data, conf_error = self.manager.load_experiment_config(exp_name)
        if conf_error:
            self.config_display.setText(conf_error)
        elif config_data:
            self.config_display.setText(json.dumps(config_data, indent=4))

        # Manage stop button state
        statuses = self.manager.get_experiment_statuses()
        is_running = statuses.get(exp_name) == STATUS_RUNNING
        self.stop_button.setEnabled(is_running)

    def stop_selected_experiment(self):
        """
        Stops the currently selected experiment if it is running.
        """
        current_item = self.exp_list_widget.currentItem()
        if not current_item:
            return

        exp_name = current_item.text().split(" ")[0]
        if self.manager.stop_experiment(exp_name):
            print(f"Stop signal sent to experiment: {exp_name}")
            self.refresh_ui()
        else:
            QMessageBox.warning(
                self, "Error", f"Could not stop {exp_name}. It may not be running."
            )


def main():
    app = QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
