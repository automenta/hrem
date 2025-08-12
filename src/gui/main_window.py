import json
import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QTabWidget,
    QTreeView,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QInputDialog,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer, QItemSelectionModel
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from .experiment_manager import ExperimentManager
from .archive_dialog import ArchiveManagerDialog
from .trajectory_view import TrajectoryView
from .scatter_plot_view import ScatterPlotView
from .unified_launch_dialog import UnifiedLaunchDialog
from .challenge_view import ChallengeView
from .challenge_launcher_dialog import ChallengeLauncherDialog
from .search_launcher_dialog import SearchLauncherDialog
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
        self._create_challenges_tab()

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
        self.exp_tree = QTreeView()
        self.exp_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.exp_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.exp_tree.setSortingEnabled(True)
        self.exp_tree.header().setStretchLastSection(True)
        self.exp_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by name...")
        self.filter_input.textChanged.connect(self.filter_experiments)

        # Container for filter and table
        left_content_container = QWidget()
        left_content_layout = QVBoxLayout(left_content_container)
        left_content_layout.setContentsMargins(0, 0, 0, 0)
        left_content_layout.addWidget(self.filter_input)
        left_content_layout.addWidget(self.exp_tree)

        # --- Buttons ---
        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.refresh_ui)
        self.launch_button = QPushButton("Launch...")
        self.launch_button.setToolTip("Launch a new experiment, challenge, or search.")
        self.launch_button.clicked.connect(self.launch_new_experiment) # This will be updated later
        self.archive_manager_button = QPushButton("Manage Archives...")
        self.archive_manager_button.clicked.connect(self.open_archive_manager)
        self.clone_button = QPushButton("Clone")
        self.clone_button.clicked.connect(self.clone_selected_experiment)
        self.clone_button.setEnabled(False)
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_selected_experiment)
        self.rename_button.setEnabled(False)
        self.select_parent_button = QPushButton("Select Parent")
        self.select_parent_button.setToolTip("Select the parent of this experiment in the tree.")
        self.select_parent_button.clicked.connect(self.select_parent_experiment)
        self.select_parent_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_selected_experiments)
        self.stop_button.setEnabled(False)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.archive_selected_experiments)
        self.delete_button.setEnabled(False)
        self.compare_button = QPushButton("Compare Selected")
        self.compare_button.clicked.connect(self.compare_selected)
        self.clear_comparison_button = QPushButton("Clear Comparison")
        self.clear_comparison_button.clicked.connect(self.clear_comparison)
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
                self.select_parent_button,
                self.stop_button,
                self.delete_button,
                self.archive_manager_button,
                self.compare_button,
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


    def _create_challenges_tab(self):
        """
        Creates the 'Challenges' tab for viewing experiment races.
        """
        self.challenge_view = ChallengeView(self.manager)
        self.tabs.addTab(self.challenge_view, "Challenges")
        self.challenge_view.experiment_selected.connect(self.select_experiment_by_name)

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


    def launch_new_experiment(self):
        """
        Opens a unified dialog to launch a new experiment, challenge, or search.
        """
        # For a new experiment, we might start with a default or empty config
        default_config = {"model": {}, "dataset": {}, "training": {}}
        dialog = UnifiedLaunchDialog(config=default_config, parent=self)

        if dialog.exec():
            launch_info = dialog.get_launch_info()
            if not launch_info:
                return

            run_type = launch_info.get("type")
            success = False
            message = "An unknown error occurred."

            if run_type == "Challenge":
                # The manager method for races is expected to exist
                if hasattr(self.manager, "launch_experiment_race"):
                    success, message = self.manager.launch_experiment_race(launch_info)
                else:
                    message = "Functionality to launch challenges is not implemented in the manager."
            elif run_type in ["Single Run", "Hyperparameter Search"]:
                success, message = self.manager.launch_experiment_from_config(
                    launch_info["config"], launch_info["name"]
                )
            else:
                message = f"Unknown run type '{run_type}' specified by the launch dialog."

            if success:
                QTimer.singleShot(LAUNCH_DELAY_MS, self.refresh_ui)
            else:
                QMessageBox.warning(self, "Launch Failed", message)

    def clone_selected_experiment(self):
        """
        Opens the unified launch dialog to clone the selected experiment.
        """
        exp_names = self.get_selected_experiment_names()
        if len(exp_names) != 1:
            QMessageBox.warning(self, "Action Failed", "Please select exactly one experiment to clone.")
            return
        original_name = exp_names[0]

        original_config, err = self.manager.load_experiment_config(original_name)
        if err:
            QMessageBox.warning(self, "Clone Failed", f"Could not load config for {original_name}: {err}")
            return

        # Suggest a new name for the clone
        new_name = f"{original_name}_clone"
        original_config["parent_experiment"] = original_name

        dialog = UnifiedLaunchDialog(config=original_config, exp_name=new_name, parent=self)
        if dialog.exec():
            launch_info = dialog.get_launch_info()
            if launch_info:
                # The dialog now handles setting the new name in the config
                success, message = self.manager.launch_experiment_from_config(
                    launch_info["config"], launch_info["name"]
                )
                if success:
                    QTimer.singleShot(LAUNCH_DELAY_MS, self.refresh_ui)
                    self.select_experiment_by_name(launch_info["name"])
                else:
                    QMessageBox.warning(self, "Launch Failed", message)

    def populate_experiment_list(self):
        """
        Populates the experiment tree with hierarchical information.
        """
        self.exp_tree.setSortingEnabled(False)
        current_selection = self.get_selected_experiment_name()

        graph = self.manager.get_experiment_graph()
        self.exp_model = QStandardItemModel()
        self.exp_model.setHorizontalHeaderLabels([
            "Name", "Type", "Status", "Model", "Dataset", "Final Loss", "Created"
        ])
        self.exp_tree.setModel(self.exp_model)

        # A map from experiment name to the tree item
        items = {}
        for name, data in graph["nodes"].items():
            # Create the list of column texts for the row
            row = [
                data["name"],
                data.get("type", "Single"),
                data["status"],
                data["model"],
                data["dataset"],
                data["final_loss"],
                data["created"],
            ]
            # Create a list of QStandardItem objects for the row
            qt_items = [QStandardItem(str(field)) for field in row]
            # Store the full data dict in the first item for later retrieval
            qt_items[0].setData(data, role=Qt.ItemDataRole.UserRole)
            items[name] = qt_items

        # Build the tree structure
        for parent_name, child_name in graph["edges"]:
            if parent_name in items and child_name in items:
                parent_item_row = items[parent_name]
                child_item_row = items[child_name]
                # The first item in the row acts as the parent for all other items in its row
                parent_item_row[0].appendRow(child_item_row)

        # Add only the roots to the model
        for root_name in graph["roots"]:
            if root_name in items:
                self.exp_model.appendRow(items[root_name])

        self.exp_tree.expandAll()
        self.exp_tree.setSortingEnabled(True)
        for i in range(self.exp_model.columnCount()):
            self.exp_tree.resizeColumnToContents(i)

        self.select_experiment_by_name(current_selection)


    def filter_experiments(self, text):
        """
        Recursively filters the experiment tree by name, status, model, etc.
        Supports queries like "my_exp model:mlp status:running" and the special
        "name:exp1,exp2" syntax from the analysis tab.
        """
        # 1. Parse the filter text
        text_lower = text.lower()
        name_filters = []
        kv_filters = {}

        # Handle the special case from the analysis tab first
        if text_lower.startswith("name:"):
            # This is a comma-separated list of exact names
            keys = text_lower[5:].split(',')
            kv_filters['name'] = keys
        else:
            # Standard space-separated filters
            for part in text_lower.split():
                if ":" in part:
                    key, value = part.split(":", 1)
                    kv_filters[key] = value
                else:
                    name_filters.append(part)

        def item_matches(item):
            """Checks if a single item matches all active filters."""
            if not item:
                return False

            data = item.data(role=Qt.ItemDataRole.UserRole)
            if not data:
                return False

            # Check substring name filters (for general text)
            item_name_lower = data.get("name", "").lower()
            for f in name_filters:
                if f not in item_name_lower:
                    return False

            # Check key-value filters
            for key, value in kv_filters.items():
                if key == 'name' and isinstance(value, list):
                    # Exact match for the comma-separated list
                    if item_name_lower not in value:
                        return False
                else:
                    # Substring match for other key-value pairs
                    if value not in str(data.get(key, "")).lower():
                        return False
            return True

        def recurse(parent_item):
            """
            Recursively applies the filter. A parent is visible if it matches
            the filter OR if any of its descendants are visible.
            """
            any_child_is_visible = False
            for r in range(parent_item.rowCount()):
                child_item = parent_item.child(r, 0)

                # A child is visible if its own children are visible
                any_grandchild_is_visible = recurse(child_item)

                # or if it matches the filter directly.
                self_matches = item_matches(child_item)

                is_visible = self_matches or any_grandchild_is_visible
                self.exp_tree.setRowHidden(r, parent_item.index(), not is_visible)

                if is_visible:
                    any_child_is_visible = True

            return any_child_is_visible

        if hasattr(self, "exp_model"):
            recurse(self.exp_model.invisibleRootItem())

    def get_selected_experiment_names(self):
        """
        Gets the names of all currently selected experiments in the tree.
        """
        selected_names = []
        if not hasattr(self, "exp_tree"):
            return []

        selection_model = self.exp_tree.selectionModel()
        if not selection_model:
            return []

        selected_indexes = selection_model.selectedRows(column=0)
        for index in selected_indexes:
            item = self.exp_model.itemFromIndex(index)
            if item:
                exp_data = item.data(role=Qt.ItemDataRole.UserRole)
                if exp_data and "name" in exp_data:
                    selected_names.append(exp_data["name"])
        return selected_names

    def get_selected_experiment_name(self):
        """
        Gets the name of the currently selected experiment in the tree.
        If multiple are selected, returns the first one.
        Returns None if no row is selected.
        """
        names = self.get_selected_experiment_names()
        return names[0] if names else None

    def select_experiment_by_name(self, name_to_select: str):
        """
        Selects the item in the experiment tree corresponding to the given name.
        """
        if not name_to_select or not hasattr(self, "exp_model"):
            return

        # QStandardItemModel.findItems is not recursive, so we do it manually
        def find_item_recursively(parent_item):
            for r in range(parent_item.rowCount()):
                item = parent_item.child(r, 0)
                if item and item.data(role=Qt.ItemDataRole.UserRole)["name"] == name_to_select:
                    return item
                # Recurse
                found_item = find_item_recursively(item)
                if found_item:
                    return found_item
            return None

        item_to_select = find_item_recursively(self.exp_model.invisibleRootItem())
        if item_to_select:
            self.exp_tree.selectionModel().select(
                item_to_select.index(),
                QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
            )
            self.exp_tree.scrollTo(item_to_select.index())

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

    def select_parent_experiment(self):
        """
        Finds and selects the parent of the currently selected experiment.
        """
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            return

        config, err = self.manager.load_experiment_config(exp_name)
        if err:
            # This can happen if the config file is deleted or corrupt.
            # We don't need to show a message box here, just do nothing.
            return

        parent_name = config.get("parent_experiment")
        if parent_name:
            self.select_experiment_by_name(parent_name)

    def clear_comparison(self):
        """
        Clears the comparison list.
        """
        self.comparison_list.clear()
        self.update_selected_experiment_display()

    def compare_selected(self):
        """
        Adds the selected experiments to the comparison list.
        """
        selected_names = self.get_selected_experiment_names()
        for name in selected_names:
            if name not in self.comparison_list:
                self.comparison_list.append(name)
        self.update_selected_experiment_display()


    def refresh_ui(self):
        """
        Refreshes the entire UI by updating the list and the display.
        """
        self.populate_experiment_list()
        self.update_selected_experiment_display()
        self.trajectory_view.draw_graph()
        self.scatter_plot_view.update_plot()
        if hasattr(self, "challenge_view"):
            self.challenge_view.refresh()

    def update_selected_experiment_display(self):
        """
        Displays the results of the selected experiment(s) or search.
        """
        self.plot_widget.clear()
        self.config_display.clear()
        self.diff_table.clear()
        self.diff_table.setRowCount(0)
        self.diff_table.setColumnCount(0)

        selected_names = self.get_selected_experiment_names()
        num_selected = len(selected_names)
        statuses = self.manager.get_experiment_statuses()

        # --- Update button states ---
        self.clear_comparison_button.setEnabled(len(self.comparison_list) > 0)
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
        self.compare_button.setEnabled(num_selected > 0)
        self.select_parent_button.setEnabled(False)  # Default to disabled; enabled in _display_single_experiment

        # --- Main display logic ---
        if self.comparison_list:
            self._display_comparison()
        elif num_selected == 1:
            exp_name = selected_names[0]
            exp_data = self.manager.get_experiment_graph()["nodes"].get(exp_name)
            if exp_data:
                if exp_data.get("type") == "Search":
                    self._display_search_summary(exp_name, exp_data)
                else:
                    self._display_single_experiment(exp_name, exp_data)
        else:
            # No selection or multiple selection outside of compare mode
            self.plot_widget.setTitle("No experiment selected")
            self.config_display.clear()
            self.diff_table.setVisible(False)

    def _display_search_summary(self, exp_name, exp_data):
        """
        Displays a summary of a hyperparameter search.
        """
        self.config_display.setVisible(True)
        self.plot_widget.setVisible(False) # No single plot for a search
        self.diff_table.setVisible(True)
        self.plot_widget.setTitle(f"Search Summary: {exp_name}")

        graph = self.manager.get_experiment_graph()
        children_names = [edge[1] for edge in graph["edges"] if edge[0] == exp_name]
        trials = [graph["nodes"][name] for name in children_names if name in graph["nodes"]]

        if not trials:
            self.config_display.setText("No trials found for this search yet.")
            self.diff_table.setVisible(False)
            return

        # --- Find best trial ---
        best_trial = None
        best_loss = float('inf')
        for trial in trials:
            try:
                loss = float(trial["final_loss"])
                if loss < best_loss:
                    best_loss = loss
                    best_trial = trial
            except (ValueError, TypeError):
                continue # Skip trials without a valid loss

        # --- Display best trial info ---
        if best_trial:
            best_trial_config, _ = self.manager.load_experiment_config(best_trial["name"])
            summary_text = (
                f"<b>Best Trial:</b> {best_trial['name']}<br>"
                f"<b>Best Test Loss:</b> {best_trial['final_loss']}<br><br>"
                f"<b>Best Parameters:</b><br>"
            )
            # We only show the tuned params for brevity
            search_config, _ = self.manager.load_experiment_config(exp_name)
            tuned_params = search_config.get("search", {}).get("params", {}).keys()

            flat_config = self._flatten_dict(best_trial_config)
            for p in tuned_params:
                summary_text += f"- {p}: {flat_config.get(p, 'N/A')}<br>"

            self.config_display.setHtml(summary_text)
        else:
            self.config_display.setText("No completed trials with valid loss values yet.")

        # --- Populate diff table with all trial results ---
        search_config, _ = self.manager.load_experiment_config(exp_name)
        tuned_params = list(search_config.get("search", {}).get("params", {}).keys())

        headers = ["Trial Name", "Status", "Final Loss"] + tuned_params
        self.diff_table.setColumnCount(len(headers))
        self.diff_table.setHorizontalHeaderLabels(headers)
        self.diff_table.setRowCount(len(trials))

        for row, trial in enumerate(trials):
            self.diff_table.setItem(row, 0, QTableWidgetItem(trial["name"]))
            self.diff_table.setItem(row, 1, QTableWidgetItem(trial["status"]))
            self.diff_table.setItem(row, 2, QTableWidgetItem(str(trial["final_loss"])))

            trial_config, _ = self.manager.load_experiment_config(trial["name"])
            if trial_config:
                flat_config = self._flatten_dict(trial_config)
                for i, key in enumerate(tuned_params):
                    val = flat_config.get(key, "N/A")
                    self.diff_table.setItem(row, 3 + i, QTableWidgetItem(str(val)))

        self.diff_table.resizeColumnsToContents()


    def _display_single_experiment(self, exp_name, exp_data):
        """
        Displays the plot and config for a single selected experiment.
        """
        self.diff_table.setVisible(False)
        self.config_display.setVisible(True)
        self.plot_widget.setVisible(True)

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
        self.plot_widget.setVisible(True)
        self._update_diff_table(self.comparison_list)
        self._update_metric_selector(self.comparison_list)

        self.plot_widget.addLegend()
        self.plot_widget.setTitle(f"Comparing {len(self.comparison_list)} experiments")

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
