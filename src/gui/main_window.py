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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QInputDialog,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer

from .experiment_manager import ExperimentManager
from .search_manager import SearchManager
from .archive_dialog import ArchiveManagerDialog
from .trajectory_view import TrajectoryView
from .scatter_plot_view import ScatterPlotView
from .launch_dialog import LaunchExperimentDialog
from .constants import (
    CONFIGS_DIR,
    INITIAL_SPLITTER_SIZES,
    LAUNCH_DELAY_MS,
    REFRESH_INTERVAL_MS,
    STATUS_RUNNING,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


class NumericTableWidgetItem(QTableWidgetItem):
    """
    A QTableWidgetItem that sorts numerically.
    """
    def __lt__(self, other):
        try:
            # Attempt to convert text to float for comparison
            return float(self.text()) < float(other.text())
        except (ValueError, TypeError):
            # Fallback to string comparison if conversion fails
            return super().__lt__(other)


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
        self.comparison_list = []
        self.current_train_loss = []
        self.current_test_loss = []
        self.exp_table_headers = []

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
        self._create_trajectory_tab()
        self._create_analysis_tab()
        self._create_search_tab()

    def _create_management_tab(
        self,
        tab_name: str,
        title: str,
        left_content_widget: QWidget,
        selection_changed_fn: callable,
        buttons: list,
        right_panel: QWidget,
    ):
        """
        Creates a standardized management tab with a content widget on the left and a display panel on the right.
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

        # Connect selection changed signal if the widget has it
        if hasattr(left_content_widget, "itemSelectionChanged"):
            left_content_widget.itemSelectionChanged.connect(selection_changed_fn)
        elif hasattr(left_content_widget, "selectionModel"):  # For QTableWidget
            left_content_widget.selectionModel().selectionChanged.connect(
                selection_changed_fn
            )

        left_layout.addWidget(left_content_widget)

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
        # --- Left Panel Widgets ---
        self.exp_table = QTableWidget()
        self.exp_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.exp_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.exp_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.exp_table.verticalHeader().setVisible(False)
        header = self.exp_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.exp_table.setSortingEnabled(True)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by name...")
        self.filter_input.textChanged.connect(self.filter_experiments)

        # Container for filter and table
        left_content_container = QWidget()
        left_content_layout = QVBoxLayout(left_content_container)
        left_content_layout.setContentsMargins(0, 0, 0, 0)
        left_content_layout.addWidget(self.filter_input)
        left_content_layout.addWidget(self.exp_table)

        # --- Buttons ---
        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.refresh_ui)
        self.launch_button = QPushButton("Launch New")
        self.launch_button.clicked.connect(self.launch_new_experiment)
        self.archive_manager_button = QPushButton("Manage Archives...")
        self.archive_manager_button.clicked.connect(self.open_archive_manager)
        self.select_parent_button = QPushButton("Select Parent")
        self.select_parent_button.clicked.connect(self.select_parent_experiment)
        self.select_parent_button.setEnabled(False)
        self.clone_button = QPushButton("Clone")
        self.clone_button.clicked.connect(self.clone_selected_experiment)
        self.clone_button.setEnabled(False)
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_selected_experiment)
        self.rename_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_selected_experiments)
        self.stop_button.setEnabled(False)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.archive_selected_experiments)
        self.delete_button.setEnabled(False)
        self.compare_all_button = QPushButton("Compare All Visible")
        self.compare_all_button.clicked.connect(self._compare_all_visible)
        self.clear_comparison_button = QPushButton("Clear Comparison")
        self.clear_comparison_button.clicked.connect(self._clear_comparison)
        self.clear_comparison_button.setEnabled(False)


        # --- Right Panel ---
        right_panel = QSplitter(Qt.Orientation.Vertical)

        # Plotting area
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0,0,0,0)
        self.plot_widget = pg.PlotWidget()

        # Metric selector
        metric_selector_layout = QHBoxLayout()
        metric_selector_layout.addWidget(QLabel("Metric:"))
        self.metric_selector = QComboBox()
        self.metric_selector.currentTextChanged.connect(self.update_selected_experiment_display)
        metric_selector_layout.addWidget(self.metric_selector)
        metric_selector_layout.addStretch()

        plot_layout.addLayout(metric_selector_layout)
        plot_layout.addWidget(self.plot_widget)

        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setFontFamily("monospace")

        self.diff_table = QTableWidget()
        self.diff_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diff_table.verticalHeader().setVisible(False)
        self.diff_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.diff_table.setVisible(False) # Initially hidden

        right_panel.addWidget(plot_container)
        right_panel.addWidget(self.config_display)
        right_panel.addWidget(self.diff_table)
        right_panel.setSizes([500, 200, 100])

        self._create_management_tab(
            "Experiments",
            "Experiments",
            left_content_container,
            self.update_selected_experiment_display,
            [
                self.refresh_button,
                self.launch_button,
                self.clone_button,
                self.rename_button,
                self.stop_button,
                self.delete_button,
                self.archive_manager_button,
                self.select_parent_button,
                self.compare_all_button,
                self.clear_comparison_button,
            ],
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

    def _create_trajectory_tab(self):
        """
        Creates the layout and widgets for the 'Trajectory' tab.
        """
        self.trajectory_tab = QWidget()
        self.tabs.addTab(self.trajectory_tab, "Research Tree")
        layout = QVBoxLayout(self.trajectory_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.trajectory_view = TrajectoryView(self.manager)
        layout.addWidget(self.trajectory_view)

    def _create_analysis_tab(self):
        """
        Creates the layout and widgets for the 'Analysis' tab.
        """
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Analysis")
        layout = QVBoxLayout(self.analysis_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scatter_plot_view = ScatterPlotView(self.manager)
        self.scatter_plot_view.experiment_selected.connect(self.select_experiment_by_name)
        self.scatter_plot_view.experiments_selected_for_filtering.connect(
            self._filter_experiments_from_analysis
        )
        layout.addWidget(self.scatter_plot_view)

    def _filter_experiments_from_analysis(self, names: list):
        """
        Filters the experiment list based on a selection from the analysis tab.
        """
        if not names:
            return

        # Switch to the experiments tab
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Experiments":
                self.tabs.setCurrentIndex(i)
                break

        # Apply a special filter
        filter_text = f"name:{','.join(names)}"
        self.filter_input.setText(filter_text)


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

    def _get_column_index(self, column_name: str) -> int:
        """
        Gets the index of a column by its header name.
        Returns -1 if the column name is not found.
        """
        try:
            return self.exp_table_headers.index(column_name)
        except ValueError:
            print(f"Warning: Column '{column_name}' not found in experiment table.")
            return -1

    def launch_new_experiment(self):
        """
        Opens a custom dialog to launch a new experiment or a race.
        """
        dialog = LaunchExperimentDialog(self)
        if dialog.exec():
            launch_info = dialog.get_launch_info()
            self.manager.launch_experiment_race(launch_info)
            QTimer.singleShot(LAUNCH_DELAY_MS, self.refresh_ui)

    def populate_experiment_list(self):
        """
        Populates the experiment table with detailed information.
        """
        self.exp_table.setSortingEnabled(False)

        # Store current selection and filter to restore them later
        current_selection = self.get_selected_experiment_name()
        filter_text = self.filter_input.text()

        self.exp_table.setRowCount(0)

        experiments = self.manager.get_experiments_data()

        self.exp_table_headers = [
            "Compare", "Name", "Status", "Model", "Dataset", "LR", "Final Loss",
            "Params", "Epoch Time (s)", "Parent", "Race ID", "Created"
        ]
        self.exp_table.setColumnCount(len(self.exp_table_headers))
        self.exp_table.setHorizontalHeaderLabels(self.exp_table_headers)

        # --- Get column indices once ---
        name_col = self._get_column_index("Name")
        status_col = self._get_column_index("Status")
        model_col = self._get_column_index("Model")
        dataset_col = self._get_column_index("Dataset")
        lr_col = self._get_column_index("LR")
        loss_col = self._get_column_index("Final Loss")
        params_col = self._get_column_index("Params")
        time_col = self._get_column_index("Epoch Time (s)")
        parent_col = self._get_column_index("Parent")
        race_col = self._get_column_index("Race ID")
        created_col = self._get_column_index("Created")
        compare_col = self._get_column_index("Compare")


        for row, exp_data in enumerate(experiments):
            self.exp_table.insertRow(row)

            # --- Checkbox for comparison ---
            if compare_col != -1:
                chk_box_widget = QWidget()
                chk_box_layout = QHBoxLayout(chk_box_widget)
                chk_box_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chk_box_layout.setContentsMargins(0,0,0,0)
                compare_checkbox = QCheckBox()
                # Use a lambda to pass the experiment name to the handler
                compare_checkbox.stateChanged.connect(
                    lambda state, name=exp_data["name"]: self._on_compare_checkbox_changed(state, name)
                )
                chk_box_layout.addWidget(compare_checkbox)
                self.exp_table.setCellWidget(row, compare_col, chk_box_widget)

            # --- Other data ---
            if name_col != -1: self.exp_table.setItem(row, name_col, QTableWidgetItem(exp_data["name"]))
            if status_col != -1: self.exp_table.setItem(row, status_col, QTableWidgetItem(exp_data["status"]))
            if model_col != -1: self.exp_table.setItem(row, model_col, QTableWidgetItem(str(exp_data["model"])))
            if dataset_col != -1: self.exp_table.setItem(row, dataset_col, QTableWidgetItem(str(exp_data["dataset"])))
            if lr_col != -1: self.exp_table.setItem(row, lr_col, NumericTableWidgetItem(str(exp_data["lr"])))
            if loss_col != -1: self.exp_table.setItem(row, loss_col, NumericTableWidgetItem(str(exp_data["final_loss"])))
            if params_col != -1: self.exp_table.setItem(row, params_col, NumericTableWidgetItem(str(exp_data["params"])))
            if time_col != -1: self.exp_table.setItem(row, time_col, NumericTableWidgetItem(str(exp_data["epoch_time"])))
            if parent_col != -1: self.exp_table.setItem(row, parent_col, QTableWidgetItem(exp_data["parent"]))
            if race_col != -1: self.exp_table.setItem(row, race_col, QTableWidgetItem(exp_data["race_id"]))
            if created_col != -1: self.exp_table.setItem(row, created_col, QTableWidgetItem(exp_data["created"]))

        self.exp_table.setSortingEnabled(True)
        self.exp_table.resizeColumnsToContents()

        name_col = self._get_column_index("Name")
        if name_col != -1:
            self.exp_table.horizontalHeader().setSectionResizeMode(name_col, QHeaderView.ResizeMode.Stretch)

        # Restore filter and selection
        self.filter_experiments(filter_text)
        self.select_experiment_by_name(current_selection)


    def filter_experiments(self, text):
        """
        Filters the experiment table by name based on the input text.
        Handles both substring search and exact name matching from analysis tab.
        """
        name_col = self._get_column_index("Name")
        if name_col == -1:
            return

        # Handle special filter from analysis tab
        if text.startswith("name:"):
            filter_names = set(text.replace("name:", "").split(','))
            for i in range(self.exp_table.rowCount()):
                name_item = self.exp_table.item(i, name_col)
                if name_item:
                    self.exp_table.setRowHidden(i, name_item.text() not in filter_names)
        else:
            # Standard substring search
            for i in range(self.exp_table.rowCount()):
                name_item = self.exp_table.item(i, name_col)
                if name_item:
                    self.exp_table.setRowHidden(i, text.lower() not in name_item.text().lower())

    def get_selected_experiment_names(self):
        """
        Gets the names of all currently selected experiments in the table.
        """
        selected_names = []
        name_col = self._get_column_index("Name")
        if name_col == -1:
            return []

        selected_rows = self.exp_table.selectionModel().selectedRows()
        for index in selected_rows:
            name_item = self.exp_table.item(index.row(), name_col)
            if name_item:
                selected_names.append(name_item.text())
        return selected_names

    def get_selected_experiment_name(self):
        """
        Gets the name of the currently selected experiment in the table.
        If multiple are selected, returns the first one.
        Returns None if no row is selected.
        """
        names = self.get_selected_experiment_names()
        return names[0] if names else None

    def select_experiment_by_name(self, name_to_select: str):
        """
        Selects the row in the experiment table corresponding to the given name.
        """
        if not name_to_select:
            return

        name_col = self._get_column_index("Name")
        if name_col == -1:
            return

        for i in range(self.exp_table.rowCount()):
            name_item = self.exp_table.item(i, name_col)
            if name_item and name_item.text() == name_to_select:
                self.exp_table.selectRow(i)
                break

    def _on_compare_checkbox_changed(self, state, exp_name):
        """
        Handles the state change of a compare checkbox.
        """
        if state == Qt.CheckState.Checked.value:
            if exp_name not in self.comparison_list:
                self.comparison_list.append(exp_name)
        else:
            if exp_name in self.comparison_list:
                self.comparison_list.remove(exp_name)

        self.update_selected_experiment_display()

    def archive_selected_experiments(self):
        """
        Deletes (archives) the currently selected experiments.
        """
        exp_names = self.get_selected_experiment_names()
        if not exp_names:
            QMessageBox.warning(self, "Action Failed", "No experiments selected.")
            return

        reply = QMessageBox.question(
            self,
            f"Archive {len(exp_names)} Experiments",
            f"Are you sure you want to archive {len(exp_names)} experiments?\n\n"
            "This is a soft delete. The experiments will be moved to the archive, "
            "from where they can be restored or permanently deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success_count = 0
            fail_count = 0
            for name in exp_names:
                success, _ = self.manager.archive_experiment(name)
                if success:
                    success_count += 1
                else:
                    fail_count += 1

            summary_message = f"Archived {success_count} experiments."
            if fail_count > 0:
                summary_message += f"\nFailed to archive {fail_count} experiments."

            QMessageBox.information(self, "Archive Complete", summary_message)
            self.refresh_ui()

    def open_archive_manager(self):
        """
        Opens the dialog to manage archived experiments.
        """
        dialog = ArchiveManagerDialog(self.manager, self)
        dialog.exec()
        # Refresh the main list in case experiments were restored
        self.refresh_ui()

    def rename_selected_experiment(self):
        """
        Renames the currently selected experiment.
        """
        exp_names = self.get_selected_experiment_names()
        if len(exp_names) != 1:
            QMessageBox.warning(self, "Action Failed", "Please select exactly one experiment to rename.")
            return
        exp_name = exp_names[0]


        new_name, ok = QInputDialog.getText(
            self,
            "Rename Experiment",
            f"Enter a new name for '{exp_name}':",
            QLineEdit.EchoMode.Normal,
            exp_name,
        )

        if ok and new_name:
            success, message = self.manager.rename_experiment(exp_name, new_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_ui()
                self.select_experiment_by_name(new_name)
            else:
                QMessageBox.warning(self, "Error", message)

    def clone_selected_experiment(self):
        """
        Clones the currently selected experiment.
        """
        exp_names = self.get_selected_experiment_names()
        if len(exp_names) != 1:
            QMessageBox.warning(self, "Action Failed", "Please select exactly one experiment to clone.")
            return
        exp_name = exp_names[0]

        new_name, ok = QInputDialog.getText(
            self,
            "Clone Experiment",
            f"Enter a name for the clone of '{exp_name}':",
            QLineEdit.EchoMode.Normal,
            f"{exp_name}_clone",
        )

        if ok and new_name:
            success, message = self.manager.clone_experiment(exp_name, new_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_ui()
                self.select_experiment_by_name(new_name)
            else:
                QMessageBox.warning(self, "Error", message)

    def select_parent_experiment(self):
        """
        Finds and selects the parent of the currently selected experiment.
        """
        exp_names = self.get_selected_experiment_names()
        if len(exp_names) != 1:
            return # Silently fail, button should be disabled
        exp_name = exp_names[0]

        config, _ = self.manager.load_experiment_config(exp_name)
        parent = config.get("parent_experiment") if config else None

        if parent and parent != "N/A":
            self.select_experiment_by_name(parent)
        else:
            QMessageBox.information(self, "No Parent", "This experiment has no parent.")

    def _clear_comparison(self):
        """
        Clears the comparison list and unchecks all associated checkboxes.
        """
        self.comparison_list.clear()

        compare_col = self._get_column_index("Compare")
        if compare_col == -1:
            return

        # Block signals to avoid triggering _on_compare_checkbox_changed repeatedly
        self.exp_table.blockSignals(True)
        for i in range(self.exp_table.rowCount()):
            cell_widget = self.exp_table.cellWidget(i, compare_col)
            if cell_widget:
                check_box = cell_widget.findChild(QCheckBox)
                if check_box and check_box.isChecked():
                    check_box.setChecked(False)
        self.exp_table.blockSignals(False)

        self.update_selected_experiment_display()

    def _compare_all_visible(self):
        """
        Adds all currently visible experiments to the comparison list.
        """
        compare_col = self._get_column_index("Compare")
        name_col = self._get_column_index("Name")
        if compare_col == -1 or name_col == -1:
            return

        # Block signals to avoid triggering _on_compare_checkbox_changed repeatedly
        self.exp_table.blockSignals(True)
        for i in range(self.exp_table.rowCount()):
            if not self.exp_table.isRowHidden(i):
                name_item = self.exp_table.item(i, name_col)
                if name_item:
                    exp_name = name_item.text()
                    if exp_name not in self.comparison_list:
                        self.comparison_list.append(exp_name)

                    cell_widget = self.exp_table.cellWidget(i, compare_col)
                    if cell_widget:
                        check_box = cell_widget.findChild(QCheckBox)
                        if check_box and not check_box.isChecked():
                            check_box.setChecked(True)
        self.exp_table.blockSignals(False)

        self.update_selected_experiment_display()


    def refresh_ui(self):
        """
        Refreshes the entire UI by updating the list and the display.
        """
        self.populate_experiment_list()
        self.update_selected_experiment_display()
        self.populate_search_list()
        self.update_search_display()
        self.trajectory_view.draw_graph()
        self.scatter_plot_view.update_plot()

    def update_selected_experiment_display(self):
        """
        Displays the results of the selected experiment or compares multiple experiments.
        """
        self.plot_widget.clear()
        self.config_display.clear()

        # Update button states related to comparison
        self.clear_comparison_button.setEnabled(len(self.comparison_list) > 0)

        # Update button states based on selection
        selected_names = self.get_selected_experiment_names()
        num_selected = len(selected_names)
        statuses = self.manager.get_experiment_statuses()

        are_any_running = False
        are_all_stopped = True
        if num_selected > 0:
            selected_statuses = [statuses.get(name) for name in selected_names]
            are_any_running = any(s == STATUS_RUNNING for s in selected_statuses)
            are_all_stopped = all(s != STATUS_RUNNING for s in selected_statuses)

        self.clone_button.setEnabled(num_selected == 1 and are_all_stopped)
        self.rename_button.setEnabled(num_selected == 1 and are_all_stopped)
        self.delete_button.setEnabled(num_selected > 0 and are_all_stopped)
        self.stop_button.setEnabled(num_selected > 0 and are_any_running)

        # For single selections, some logic is handled in _display_single_experiment
        if num_selected != 1:
            self.select_parent_button.setEnabled(False)

        if self.comparison_list:
            self._display_comparison()
        elif num_selected == 1:
            self._display_single_experiment()
        else:
            # No selection or multiple selection outside of compare mode
            self.plot_widget.setTitle("No experiment selected")
            self.config_display.clear()
            self.diff_table.setVisible(False)


    def _display_single_experiment(self):
        """
        Displays the plot and config for a single selected experiment.
        """
        self.diff_table.setVisible(False)
        self.config_display.setVisible(True)

        exp_name = self.get_selected_experiment_name() # Should be guaranteed to be one
        if not exp_name: # Should not happen if called correctly
            self.plot_widget.setTitle("No experiment selected")
            return

        self._update_metric_selector([exp_name])
        self.current_train_loss, self.current_test_loss = [], []


        # Update plot
        results_data, res_error = self.manager.load_experiment_results(exp_name)
        if res_error:
            QMessageBox.warning(self, "Result file error", res_error)
        elif results_data:
            metric_base_name = self.metric_selector.currentText()
            # Special handling for loss, which is the default
            if not metric_base_name and "train_loss" in results_data:
                metric_base_name = "loss"

            train_metric = "train_" + metric_base_name
            test_metric = "test_" + metric_base_name

            self.current_train_loss = results_data.get(train_metric, [])
            self.current_test_loss = results_data.get(test_metric, [])

            self.plot_widget.setTitle(f"Learning Curves: {exp_name} ({metric_base_name})")
            if self.current_train_loss:
                self.plot_widget.plot(self.current_train_loss, pen="b", name=f"Train {metric_base_name}")
            if self.current_test_loss:
                self.plot_widget.plot(self.current_test_loss, pen="r", name=f"Test {metric_base_name}")
            self.plot_widget.addLegend()
        else:
            self.plot_widget.setTitle(f"No results available for: {exp_name}")

        # Update config view
        config_data, conf_error = self.manager.load_experiment_config(exp_name)
        if conf_error:
            self.config_display.setText(conf_error)
        elif config_data:
            self.config_display.setText(json.dumps(config_data, indent=4))

        # Manage button states specific to single selection
        parent = config_data.get("parent_experiment") if config_data else None
        self.select_parent_button.setEnabled(bool(parent))

    def _display_comparison(self):
        """
        Displays the plots for all experiments in the comparison list.
        """
        self.config_display.setVisible(False)
        self.diff_table.setVisible(True)
        self._update_diff_table(self.comparison_list)
        self._update_metric_selector(self.comparison_list)

        self.plot_widget.addLegend()
        self.plot_widget.setTitle(f"Comparing {len(self.comparison_list)} experiments")

        # Buttons are handled by update_selected_experiment_display
        self.compare_all_button.setEnabled(True)

        colors = ["b", "r", "g", "c", "m", "y", "w"]
        metric_base_name = self.metric_selector.currentText()
        if not metric_base_name:
            return

        train_metric = "train_" + metric_base_name
        test_metric = "test_" + metric_base_name

        for i, exp_name in enumerate(self.comparison_list):
            results_data, _ = self.manager.load_experiment_results(exp_name)
            if results_data:
                train_vals = results_data.get(train_metric, [])
                test_vals = results_data.get(test_metric, [])
                color = colors[i % len(colors)]

                if train_vals:
                    self.plot_widget.plot(train_vals, pen=pg.mkPen(color, style=Qt.PenStyle.SolidLine), name=f"{exp_name} Train")
                if test_vals:
                    self.plot_widget.plot(test_vals, pen=pg.mkPen(color, style=Qt.PenStyle.DashLine), name=f"{exp_name} Test")

    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _update_diff_table(self, exp_names):
        """
        Populates the diff table with the differing parameters of the given experiments.
        """
        self.diff_table.setRowCount(0)
        if len(exp_names) < 2:
            self.diff_table.setVisible(False)
            return

        # --- Load and flatten configs ---
        configs = []
        for name in exp_names:
            config_data, _ = self.manager.load_experiment_config(name)
            if config_data:
                configs.append(self._flatten_dict(config_data))

        if len(configs) < 2:
            return

        # --- Find all unique keys and differing keys ---
        all_keys = set()
        for config in configs:
            all_keys.update(config.keys())

        diff_keys = set()
        for key in all_keys:
            first_val = configs[0].get(key)
            for i in range(1, len(configs)):
                if configs[i].get(key) != first_val:
                    diff_keys.add(key)
                    break

        # --- Populate table ---
        self.diff_table.setColumnCount(len(exp_names) + 1)
        headers = ["Parameter"] + exp_names
        self.diff_table.setHorizontalHeaderLabels(headers)

        sorted_diff_keys = sorted(list(diff_keys))
        self.diff_table.setRowCount(len(sorted_diff_keys))

        for row, key in enumerate(sorted_diff_keys):
            self.diff_table.setItem(row, 0, QTableWidgetItem(key))
            for col, config in enumerate(configs):
                value = config.get(key, "N/A")
                self.diff_table.setItem(row, col + 1, QTableWidgetItem(str(value)))

        self.diff_table.resizeColumnsToContents()

    def _update_metric_selector(self, exp_names):
        """
        Populates the metric selector with common metrics from the given experiments.
        """
        self.metric_selector.blockSignals(True)
        self.metric_selector.clear()

        if not exp_names:
            self.metric_selector.blockSignals(False)
            return

        # Find common metrics
        common_metrics = None
        for name in exp_names:
            results, _ = self.manager.load_experiment_results(name)
            if results:
                # Extract base metric names (e.g., 'loss' from 'train_loss')
                metrics = {
                    key.replace("train_", "").replace("test_", "")
                    for key, val in results.items()
                    if isinstance(val, list) and (key.startswith("train_") or key.startswith("test_"))
                }
                if common_metrics is None:
                    common_metrics = metrics
                else:
                    common_metrics.intersection_update(metrics)

        if common_metrics:
            # Ensure we have pairs (train/test) for a metric to be valid
            valid_metrics = []
            for metric in common_metrics:
                has_train = any(f"train_{metric}" in results.keys() for results, _ in (self.manager.load_experiment_results(n) for n in exp_names))
                has_test = any(f"test_{metric}" in results.keys() for results, _ in (self.manager.load_experiment_results(n) for n in exp_names))
                if has_train or has_test:
                    valid_metrics.append(metric)

            self.metric_selector.addItems(sorted(list(set(valid_metrics))))

        self.metric_selector.blockSignals(False)


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

    def stop_selected_experiments(self):
        """
        Stops the currently selected running experiments.
        """
        exp_names = self.get_selected_experiment_names()
        if not exp_names:
            return

        stopped_count = 0
        for name in exp_names:
            if self.manager.stop_experiment(name):
                stopped_count += 1

        if stopped_count > 0:
            print(f"Stop signal sent to {stopped_count} experiments.")
            # Give some time for processes to terminate before refreshing
            QTimer.singleShot(500, self.refresh_ui)

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
        statuses = self.search_manager.get_search_statuses()

        # In-place update to avoid flicker
        current_items = {
            self.search_list_widget.item(i).text().split(" ")[0]: self.search_list_widget.item(i)
            for i in range(self.search_list_widget.count())
        }

        if set(current_items.keys()) != set(statuses.keys()):
            current_selection = (
                self.search_list_widget.currentItem().text().split(" ")[0]
                if self.search_list_widget.currentItem()
                else None
            )
            self.search_list_widget.clear()
            for name in sorted(statuses.keys()):
                status = statuses[name]
                item_text = f"{name} ({status})"
                self.search_list_widget.addItem(item_text)
                if name == current_selection:
                    self.search_list_widget.setCurrentRow(self.search_list_widget.count() - 1)
        else:
            for name, item in current_items.items():
                status = statuses[name]
                item_text = f"{name} ({status})"
                if item.text() != item_text:
                    item.setText(item_text)

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
