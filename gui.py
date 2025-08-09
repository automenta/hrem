import sys
import os
import json
import pyqtgraph as pg
import sys
import os
import json
import subprocess
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
)
from PyQt6.QtCore import Qt, QTimer


class MainGUI(QMainWindow):
    """
    The main window for the experiment GUI.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experimentation Platform")
        self.setGeometry(100, 100, 1200, 800)

        # --- Main Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Splitter to make sections resizable ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel (Experiment List) ---
        left_panel = QWidget()
        self.left_layout = QVBoxLayout(left_panel)
        self.left_layout.addWidget(QLabel("Experiments"))

        self.exp_list_widget = QListWidget()
        self.exp_list_widget.itemSelectionChanged.connect(self.display_experiment_results)
        self.left_layout.addWidget(self.exp_list_widget)

        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.populate_experiment_list)
        self.left_layout.addWidget(self.refresh_button)

        self.launch_button = QPushButton("Launch New Experiment")
        self.launch_button.clicked.connect(self.launch_experiment)
        self.left_layout.addWidget(self.launch_button)
        splitter.addWidget(left_panel)

        self.populate_experiment_list()

        # --- Right Panel (Plotting) ---
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.plot_widget = pg.PlotWidget()
        self.right_layout.addWidget(self.plot_widget)
        splitter.addWidget(right_panel)

        # Adjust initial sizes of the splitter
        splitter.setSizes([300, 900])

        # --- Timer for live updates ---
        self.timer = QTimer()
        self.timer.setInterval(2000)  # Update every 2 seconds
        self.timer.timeout.connect(self.display_experiment_results)
        self.timer.start()

    def launch_experiment(self):
        """
        Opens a file dialog to select a config file and launches the experiment.
        """
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Select Experiment Config", "configs", "JSON files (*.json)"
        )
        if config_path:
            # Run main.py in a new process
            subprocess.Popen([sys.executable, "main.py", config_path])
            # A small delay to allow the experiment to create its directory
            QTimer.singleShot(500, self.populate_experiment_list)

    def populate_experiment_list(self):
        """
        Scans the results directory and populates the experiment list widget.
        """
        self.exp_list_widget.clear()
        results_dir = "results"
        if os.path.exists(results_dir):
            exp_names = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])
            self.exp_list_widget.addItems(exp_names)

    def display_experiment_results(self):
        """
        Displays the results of the selected experiment, live or completed.
        """
        current_item = self.exp_list_widget.currentItem()
        if not current_item:
            return

        exp_name = current_item.text()
        live_file = os.path.join("results", exp_name, "live.json")
        results_file = os.path.join("results", exp_name, "results.json")

        results_data = None
        if os.path.exists(live_file):
            try:
                with open(live_file, "r") as f:
                    results_data = json.load(f).get('full_results', {})
            except json.JSONDecodeError:
                pass  # Ignore if file is being written
        elif os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    results_data = json.load(f)
            except json.JSONDecodeError:
                pass # Ignore if file is corrupted

        if results_data:
            train_loss = results_data.get('train_loss', [])
            test_loss = results_data.get('test_loss', [])

            self.plot_widget.clear()
            self.plot_widget.setTitle(f"Learning Curves: {exp_name}")
            self.plot_widget.plot(train_loss, pen='b', name='Train Loss')
            self.plot_widget.plot(test_loss, pen='r', name='Test Loss')
            self.plot_widget.addLegend()


def main():
    app = QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
