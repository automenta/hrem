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

from .experiment_manager import ExperimentManager
from .search_manager import SearchManager
from .constants import (
    CONFIGS_DIR,
    INITIAL_SPLITTER_SIZES,
    LAUNCH_DELAY_MS,
    REFRESH_INTERVAL_MS,
    STATUS_RUNNING,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


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
        self.current_train_loss = []
        self.current_test_loss = []

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

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._create_experiments_tab()
        self._create_search_tab()

    def _create_management_tab(
        self,
        tab_name: str,
        title: str,
        list_widget: QListWidget,
        selection_changed_fn: callable,
        buttons: list,
        right_panel: QWidget,
    ):
        """
        Creates a standardized management tab with a list on the left and a display panel on the right.
        """
        tab = QWidget()
        self.tabs.addTab(tab, tab_name)
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel(title))
        list_widget.itemSelectionChanged.connect(selection_changed_fn)
        left_layout.addWidget(list_widget)

        button_layout = QHBoxLayout()
        for button in buttons:
            button_layout.addWidget(button)
        left_layout.addLayout(button_layout)
        splitter.addWidget(left_panel)

        # --- Right Panel ---
        splitter.addWidget(right_panel)
        splitter.setSizes(INITIAL_SPLITTER_SIZES)

    def _create_experiments_tab(self):
        """
        Creates the layout and widgets for the 'Experiments' tab.
        """
        self.exp_list_widget = QListWidget()

        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.refresh_ui)
        self.launch_button = QPushButton("Launch New")
        self.launch_button.clicked.connect(self.launch_new_experiment)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_selected_experiment)
        self.stop_button.setEnabled(False)

        right_panel = QSplitter(Qt.Orientation.Vertical)
        self.plot_widget = pg.PlotWidget()
        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setFontFamily("monospace")
        right_panel.addWidget(self.plot_widget)
        right_panel.addWidget(self.config_display)
        right_panel.setSizes([600, 200])

        self._create_management_tab(
            "Experiments",
            "Experiments",
            self.exp_list_widget,
            self.update_selected_experiment_display,
            [self.refresh_button, self.launch_button, self.stop_button],
            right_panel,
        )

        # --- Setup for plot interactivity ---
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.plot_label = pg.TextItem()
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        self.plot_widget.addItem(self.plot_label, ignoreBounds=True)
        self.v_line.hide()
        self.h_line.hide()
        self.plot_label.hide()

        self.plot_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_plot_hover,
        )

    def _create_search_tab(self):
        """
        Creates the layout and widgets for the 'Search' tab.
        """
        self.search_list_widget = QListWidget()

        self.launch_search_button = QPushButton("Launch New")
        self.launch_search_button.clicked.connect(self.launch_new_search)
        self.stop_search_button = QPushButton("Stop")
        self.stop_search_button.clicked.connect(self.stop_selected_search)
        self.stop_search_button.setEnabled(False)

        self.search_output_display = QTextEdit()
        self.search_output_display.setReadOnly(True)
        self.search_output_display.setFontFamily("monospace")

        self._create_management_tab(
            "Search",
            "Hyperparameter Searches",
            self.search_list_widget,
            self.on_search_selection_changed,
            [self.launch_search_button, self.stop_search_button],
            self.search_output_display,
        )

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
        statuses = self.manager.get_experiment_statuses()
        self._update_list_widget(self.exp_list_widget, statuses)

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
        self.current_train_loss, self.current_test_loss = [], []

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
            self.current_train_loss = results_data.get("train_loss", [])
            self.current_test_loss = results_data.get("test_loss", [])

            self.plot_widget.setTitle(f"Learning Curves: {exp_name}")
            self.plot_widget.plot(self.current_train_loss, pen="b", name="Train Loss")
            self.plot_widget.plot(self.current_test_loss, pen="r", name="Test Loss")
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

    def _on_plot_hover(self, event):
        """
        Handles mouse hover events on the plot to display data points.
        """
        pos = event[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            if 0 <= x < len(self.current_train_loss):
                index = int(round(x))
                if (
                    0 <= index < len(self.current_train_loss)
                    and 0 <= index < len(self.current_test_loss)
                ):
                    train_val = self.current_train_loss[index]
                    test_val = self.current_test_loss[index]
                    self.plot_label.setText(
                        f"Epoch: {index}\nTrain: {train_val:.4f}\nTest: {test_val:.4f}"
                    )
                    self.plot_label.setPos(x, y)
                    self.v_line.setPos(x)
                    self.h_line.setPos(y)
                    self.v_line.show()
                    self.h_line.show()
                    self.plot_label.show()
                    return

        self.v_line.hide()
        self.h_line.hide()
        self.plot_label.hide()

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

    def _update_list_widget(self, list_widget: QListWidget, new_statuses: dict):
        """
        Updates a QListWidget in-place with new items and statuses.
        This method avoids clearing and re-populating the list to prevent flicker,
        unless the set of items has changed.
        """
        current_items = {
            list_widget.item(i).text().split(" ")[0]: list_widget.item(i)
            for i in range(list_widget.count())
        }

        if set(current_items.keys()) != set(new_statuses.keys()):
            # Fallback to clear-and-repopulate if items were added or removed
            current_selection = (
                list_widget.currentItem().text().split(" ")[0]
                if list_widget.currentItem()
                else None
            )
            list_widget.clear()
            for name in sorted(new_statuses.keys()):
                status = new_statuses[name]
                item_text = f"{name} ({status})"
                list_widget.addItem(item_text)
                if name == current_selection:
                    list_widget.setCurrentRow(list_widget.count() - 1)
        else:
            # Just update the text of existing items
            for name, item in current_items.items():
                status = new_statuses[name]
                item_text = f"{name} ({status})"
                if item.text() != item_text:
                    item.setText(item_text)

    def populate_search_list(self):
        """
        Populates the search list with names and statuses.
        """
        statuses = self.search_manager.get_search_statuses()
        self._update_list_widget(self.search_list_widget, statuses)

    def update_search_display(self):
        """
        Displays the output of the selected search.
        """
        current_item = self.search_list_widget.currentItem()

        if not current_item:
            self.stop_search_button.setEnabled(False)
            return

        search_name = current_item.text().split(" ")[0]

        # Append any new output from the search process
        new_output = self.search_manager.get_search_output(search_name)
        if new_output:
            self.search_output_display.moveCursor(
                self.search_output_display.textCursor().End
            )
            self.search_output_display.insertPlainText(new_output)

        statuses = self.search_manager.get_search_statuses()
        is_running = statuses.get(search_name) == STATUS_RUNNING
        self.stop_search_button.setEnabled(is_running)

    def on_search_selection_changed(self):
        """
        Clears the search output display when the selection changes.
        """
        self.search_output_display.clear()
        self.update_search_display()

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
