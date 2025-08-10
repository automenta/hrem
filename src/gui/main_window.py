import json
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
    QTabWidget,
)
from PyQt6.QtCore import Qt, QTimer

from .experiment_manager import ExperimentManager, STATUS_RUNNING
from .search_manager import SearchManager

# --- Constants ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
REFRESH_INTERVAL_MS = 2000
LAUNCH_DELAY_MS = 500
INITIAL_SPLITTER_SIZES = [300, 900]
CONFIGS_DIR = "configs"


class MainGUI(QMainWindow):
    """
    The main window for the experiment GUI.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experimentation Platform")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.manager = ExperimentManager()
        self.search_manager = SearchManager()

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
        main_layout = QVBoxLayout(main_widget)

        # --- Create Tab Widget ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Experiments Tab ---
        self.experiments_tab = QWidget()
        self.tabs.addTab(self.experiments_tab, "Experiments")
        self._create_experiments_tab()

        # --- Search Tab ---
        self.search_tab = QWidget()
        self.tabs.addTab(self.search_tab, "Search")
        self._create_search_tab()

    def _create_experiments_tab(self):
        """
        Creates the layout and widgets for the 'Experiments' tab.
        """
        layout = QHBoxLayout(self.experiments_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Left Panel (Experiment List) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Experiments"))
        self.exp_list_widget = QListWidget()
        self.exp_list_widget.itemSelectionChanged.connect(
            self.update_selected_experiment_display
        )
        left_layout.addWidget(self.exp_list_widget)

        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.refresh_ui)
        button_layout.addWidget(self.refresh_button)

        self.launch_button = QPushButton("Launch New Experiment")
        self.launch_button.clicked.connect(self.launch_new_experiment)
        button_layout.addWidget(self.launch_button)

        self.stop_button = QPushButton("Stop Experiment")
        self.stop_button.clicked.connect(self.stop_selected_experiment)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        left_layout.addLayout(button_layout)
        splitter.addWidget(left_panel)

        # --- Right Panel (Plot and Config) ---
        right_panel = QSplitter(Qt.Orientation.Vertical)
        self.plot_widget = pg.PlotWidget()
        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setFontFamily("monospace")
        right_panel.addWidget(self.plot_widget)
        right_panel.addWidget(self.config_display)
        right_panel.setSizes([600, 200])
        splitter.addWidget(right_panel)
        splitter.setSizes(INITIAL_SPLITTER_SIZES)

    def _create_search_tab(self):
        """
        Creates the layout and widgets for the 'Search' tab.
        """
        layout = QHBoxLayout(self.search_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Left Panel (Search List) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Hyperparameter Searches"))
        self.search_list_widget = QListWidget()
        self.search_list_widget.itemSelectionChanged.connect(self.update_search_display)
        left_layout.addWidget(self.search_list_widget)

        button_layout = QHBoxLayout()
        self.launch_search_button = QPushButton("Launch New Search")
        self.launch_search_button.clicked.connect(self.launch_new_search)
        button_layout.addWidget(self.launch_search_button)

        self.stop_search_button = QPushButton("Stop Search")
        self.stop_search_button.clicked.connect(self.stop_selected_search)
        self.stop_search_button.setEnabled(False)
        button_layout.addWidget(self.stop_search_button)

        left_layout.addLayout(button_layout)
        splitter.addWidget(left_panel)

        # --- Right Panel (Search Output) ---
        self.search_output_display = QTextEdit()
        self.search_output_display.setReadOnly(True)
        self.search_output_display.setFontFamily("monospace")
        splitter.addWidget(self.search_output_display)
        splitter.setSizes(INITIAL_SPLITTER_SIZES)

    def launch_new_experiment(self):
        """
        Opens a file dialog to select a config file and launches the experiment.
        """
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Select Experiment Config", CONFIGS_DIR, "Config files (*.json *.conf)"
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
        self.populate_search_list()
        self.update_search_display()

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

    def launch_new_search(self):
        """
        Opens a file dialog to select a config file and launches a search.
        """
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Select Search Config", CONFIGS_DIR, "Config files (*.json *.conf)"
        )
        if config_path:
            self.search_manager.launch_search(config_path)
            QTimer.singleShot(LAUNCH_DELAY_MS, self.populate_search_list)

    def populate_search_list(self):
        """
        Populates the search list with names and statuses.
        """
        current_selection = (
            self.search_list_widget.currentItem().text().split(" ")[0]
            if self.search_list_widget.currentItem()
            else None
        )
        self.search_list_widget.clear()
        statuses = self.search_manager.get_search_statuses()
        for search_name, status in statuses.items():
            item_text = f"{search_name} ({status})"
            self.search_list_widget.addItem(item_text)
            if search_name == current_selection:
                self.search_list_widget.setCurrentRow(
                    self.search_list_widget.count() - 1
                )

    def update_search_display(self):
        """
        Displays the output of the selected search.
        """
        current_item = self.search_list_widget.currentItem()
        self.search_output_display.clear()

        if not current_item:
            self.stop_search_button.setEnabled(False)
            return

        search_name = current_item.text().split(" ")[0]
        # TODO: Read and display the live output from the search process
        self.search_output_display.setText(f"Display for {search_name} coming soon...")

        statuses = self.search_manager.get_search_statuses()
        is_running = statuses.get(search_name) == "Running"
        self.stop_search_button.setEnabled(is_running)

    def stop_selected_search(self):
        """
        Stops the currently selected search if it is running.
        """
        current_item = self.search_list_widget.currentItem()
        if not current_item:
            return

        search_name = current_item.text().split(" ")[0]
        if self.search_manager.stop_search(search_name):
            print(f"Stop signal sent to search: {search_name}")
            self.populate_search_list()
        else:
            QMessageBox.warning(
                self, "Error", f"Could not stop {search_name}. It may not be running."
            )
