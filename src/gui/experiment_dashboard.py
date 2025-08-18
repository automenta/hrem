import os
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QDialog,
    QLineEdit,
    QComboBox,
    QLabel,
    QFormLayout,
    QDialogButtonBox,
    QAbstractItemView,
    QMenu,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QTabWidget,
    QTreeWidgetItemIterator,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QKeySequence, QColor
import pyqtgraph as pg

from .experiment_manager import ExperimentManager
from .race_launcher import RaceLauncher
from .race_monitor import RaceMonitor
from .reusable_dialogs import InputDialog
from .utils import flatten_dict
from .log_viewer import LogViewer
from .archive_browser import ArchiveBrowser
from .constants import STATUS_RUNNING, STATUS_FAILED, STATUS_COMPLETED, STATUS_ERROR
from .styles import PALETTE


class NumericTableWidgetItem(QTableWidgetItem):
    """
    A custom QTableWidgetItem for sorting numbers correctly.
    It handles cases where the text might not be a valid number.
    """
    def __lt__(self, other):
        self_text = self.text()
        other_text = other.text()
        try:
            self_float = float(self_text)
            other_float = float(other_text)
            return self_float < other_float
        except ValueError:
            # If one is not a number, it can be treated as "smaller" or "larger"
            # Here, we'll just fall back to string comparison, which is stable.
            return self_text < other_text

class ExperimentDashboard(QMainWindow):
    """
    The main dashboard window for viewing and managing all experiments.
    """

    def __init__(self, manager: ExperimentManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Experiment Dashboard")
        self.setGeometry(100, 100, 1400, 800)

        # --- Data ---
        self.all_experiments_data = [] # Holds all data from the manager
        self.experiments_data = [] # Holds the filtered and sorted data to be displayed
        self.race_monitors = {}  # To track open race monitor windows
        self.log_viewers = {} # To track open log viewer windows
        self.archive_browser = None # To track the archive browser window

        # --- UI ---
        self._init_ui()
        self._init_timer()

        # --- Initial Load ---
        self.refresh_data()

    def _init_ui(self):
        # --- Central Widget & Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Toolbar / Actions ---
        action_layout = QHBoxLayout()
        self.launch_race_button = QPushButton(" Launch New Race")
        self.launch_race_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.launch_race_button.clicked.connect(self.launch_new_race)
        self.archive_button = QPushButton(" Archive")
        self.archive_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirLinkIcon))
        self.archive_button.clicked.connect(self.open_archive_browser)
        self.refresh_button = QPushButton(" Refresh")
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.clicked.connect(self.refresh_data)
        action_layout.addWidget(self.launch_race_button)
        action_layout.addWidget(self.archive_button)
        action_layout.addStretch()
        action_layout.addWidget(self.refresh_button)
        layout.addLayout(action_layout)

        # --- Filter Controls ---
        filter_box = QHBoxLayout()
        filter_box.addWidget(QLabel("Filter by:"))
        self.name_filter_input = QLineEdit()
        self.name_filter_input.setPlaceholderText("Name contains...")
        self.name_filter_input.textChanged.connect(self._apply_filters)
        filter_box.addWidget(self.name_filter_input)

        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItems(["All", STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED, STATUS_ERROR])
        self.status_filter_combo.currentIndexChanged.connect(self._apply_filters)
        filter_box.addWidget(QLabel("Status:"))
        filter_box.addWidget(self.status_filter_combo)

        self.dataset_filter_combo = QComboBox()
        # Populated dynamically
        self.dataset_filter_combo.currentIndexChanged.connect(self._apply_filters)
        filter_box.addWidget(QLabel("Dataset:"))
        filter_box.addWidget(self.dataset_filter_combo)
        filter_box.addStretch()
        layout.addLayout(filter_box)


        # --- Experiment Table ---
        self.table = QTableWidget()
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.setSortingEnabled(True) # Enable sorting
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.doubleClicked.connect(self.handle_double_click)

        # --- Tree View Tab ---
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Experiment", "Status", "Final Loss"])
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemDoubleClicked.connect(self.handle_double_click)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # --- Tab Widget ---
        self.tabs = QTabWidget()
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0,0,0,0)
        table_layout.addWidget(self.table)
        self.tabs.addTab(table_container, "📋 Table")

        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0,0,0,0)
        tree_layout.addWidget(self.tree)
        self.tabs.addTab(tree_container, "🌳 Tree")

        # --- Analysis Tab ---
        analysis_container = QWidget()
        analysis_layout = QVBoxLayout(analysis_container)
        analysis_layout.setContentsMargins(0, 0, 0, 0)

        # -- Selector controls --
        selector_layout = QHBoxLayout()
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.setToolTip("Select the metric or hyperparameter for the X-axis.")
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.setToolTip("Select the metric or hyperparameter for the Y-axis.")
        self.size_combo = QComboBox()
        self.size_combo.setToolTip("Select a metric to represent by point size (or 'None').")
        selector_layout.addWidget(QLabel("X-Axis:"))
        selector_layout.addWidget(self.x_axis_combo)
        selector_layout.addWidget(QLabel("Y-Axis:"))
        selector_layout.addWidget(self.y_axis_combo)
        selector_layout.addWidget(QLabel("Size:"))
        selector_layout.addWidget(self.size_combo)
        selector_layout.addStretch()
        analysis_layout.addLayout(selector_layout)

        # -- Plot widget --
        self.analysis_plot = pg.PlotWidget()
        self.scatter_plot = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 150),
            hoverable=True, hoverBrush=pg.mkBrush(255, 0, 0, 200)
        )
        self.analysis_plot.addItem(self.scatter_plot)
        self.analysis_plot.showGrid(x=True, y=True, alpha=0.3)
        self.analysis_plot.getPlotItem().setMenuEnabled(False) # Disable default context menu
        analysis_layout.addWidget(self.analysis_plot)

        self.tabs.addTab(analysis_container, "📈 Analysis")

        # Connect signals
        self.x_axis_combo.currentIndexChanged.connect(self._update_analysis_plot)
        self.y_axis_combo.currentIndexChanged.connect(self._update_analysis_plot)
        self.size_combo.currentIndexChanged.connect(self._update_analysis_plot)
        self.scatter_plot.sigHovered.connect(self._on_scatter_hover)
        self.analysis_plot_text = pg.TextItem(text="", color=(200, 200, 200), anchor=(0,1))
        self.analysis_plot.addItem(self.analysis_plot_text)
        self.analysis_plot_text.hide()


        layout.addWidget(self.tabs)

        # --- Actions with Shortcuts ---
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut(QKeySequence.StandardKey.Refresh)  # F5
        refresh_action.triggered.connect(self.refresh_data)
        self.addAction(refresh_action)

        launch_action = QAction("Launch New Race", self)
        launch_action.setShortcut(QKeySequence("Ctrl+N"))
        launch_action.triggered.connect(self.launch_new_race)
        self.addAction(launch_action)


        # Define table columns
        self.column_keys = [
            ("name", "Experiment Name"),
            ("type", "Type"),
            ("status", "Status"),
            ("model", "Model"),
            ("dataset", "Dataset"),
            ("final_loss", "Final Loss"),
            ("created", "Created"),
            ("race_id", "Race ID"),
        ]
        self.table.setColumnCount(len(self.column_keys))
        self.table.setHorizontalHeaderLabels([label for _, label in self.column_keys])

    def _init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(5000)  # Refresh every 5 seconds

    def refresh_data(self):
        """
        Fetches the latest experiment data, updates filters, and refreshes the table.
        """
        current_selection = self.get_selected_experiment_name()
        # Fetch all data from the manager
        self.all_experiments_data = self.manager.get_experiments_data()

        self._update_filter_options()
        self._apply_filters() # This calls update_table
        self._update_tree_view() # This calls update_tree
        self._update_analysis_view() # This calls the new analysis update method

        self.select_experiment_by_name(current_selection)

    def _update_filter_options(self):
        """
        Updates the dataset filter dropdown with available datasets.
        """
        current_dataset = self.dataset_filter_combo.currentText()
        datasets = {"All"}
        for exp in self.all_experiments_data:
            dataset = exp.get("dataset")
            if dataset and dataset != "N/A":
                datasets.add(dataset)

        self.dataset_filter_combo.blockSignals(True)
        self.dataset_filter_combo.clear()
        self.dataset_filter_combo.addItems(sorted(list(datasets)))
        idx = self.dataset_filter_combo.findText(current_dataset)
        if idx != -1:
            self.dataset_filter_combo.setCurrentIndex(idx)
        else:
            self.dataset_filter_combo.setCurrentIndex(0)
        self.dataset_filter_combo.blockSignals(False)

    def _apply_filters(self):
        """
        Filters the full experiment list based on UI controls and updates the table.
        """
        name_filter = self.name_filter_input.text().lower().strip()
        status_filter = self.status_filter_combo.currentText()
        dataset_filter = self.dataset_filter_combo.currentText()

        filtered_data = self.all_experiments_data

        if name_filter:
            filtered_data = [
                exp for exp in filtered_data
                if name_filter in exp.get("name", "").lower()
            ]
        if status_filter != "All":
            filtered_data = [
                exp for exp in filtered_data
                if exp.get("status") == status_filter
            ]
        if dataset_filter != "All":
            filtered_data = [
                exp for exp in filtered_data
                if exp.get("dataset") == dataset_filter
            ]

        self.experiments_data = filtered_data
        self.update_table()


    def update_table(self):
        """
        Repopulates the QTableWidget with the current (filtered) experiment data.
        """
        self.table.setSortingEnabled(False) # Disable sorting during update for performance
        self.table.clearContents()

        self.table.setRowCount(len(self.experiments_data))
        status_col_idx = next((i for i, (key, _) in enumerate(self.column_keys) if key == "status"), None)
        loss_col_idx = next((i for i, (key, _) in enumerate(self.column_keys) if key == "final_loss"), None)

        for row, exp_data in enumerate(self.experiments_data):
            for col, (key, _) in enumerate(self.column_keys):
                value_str = str(exp_data.get(key, "N/A"))
                if col == loss_col_idx:
                    item = NumericTableWidgetItem(value_str)
                else:
                    item = QTableWidgetItem(value_str)
                self.table.setItem(row, col, item)

            # Color code the status column
            if status_col_idx is not None:
                status_item = self.table.item(row, status_col_idx)
                status_text = status_item.text()
                color = None
                if status_text == STATUS_RUNNING:
                    color = PALETTE["accent_yellow"]
                elif status_text == STATUS_FAILED:
                    color = PALETTE["accent_red"]
                elif status_text == STATUS_COMPLETED:
                    color = PALETTE["accent_green"]
                elif status_text == STATUS_ERROR:
                    color = PALETTE["accent_orange"]

                if color:
                    status_item.setBackground(QColor(color))

                # Add a tooltip for error messages
                error_message = exp_data.get("error_message")
                if error_message:
                    status_item.setToolTip(error_message)


        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )

    def get_selected_experiment_data(self):
        """Returns the full data dict for the selected experiment from the active view."""
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index == 0:  # Table view
            selected_rows = self.table.selectionModel().selectedRows()
            if not selected_rows:
                return None
            name_item = self.table.item(selected_rows[0].row(), 0)
            if not name_item:
                return None
            name = name_item.text()
            return next((exp for exp in self.all_experiments_data if exp['name'] == name), None)

        elif current_tab_index == 1:  # Tree view
            selected_items = self.tree.selectedItems()
            if not selected_items:
                return None
            # Data is stored in the first column of the item
            return selected_items[0].data(0, Qt.ItemDataRole.UserRole)

        return None

    def get_selected_experiment_name(self):
        """Returns the name of the selected experiment."""
        exp_data = self.get_selected_experiment_data()
        return exp_data["name"] if exp_data else None

    def select_experiment_by_name(self, name_to_select):
        """Selects an item in the active view based on the experiment name."""
        if not name_to_select:
            return

        current_tab_index = self.tabs.currentIndex()
        if current_tab_index == 0: # Table View
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).text() == name_to_select:
                    self.table.selectRow(row)
                    return

        elif current_tab_index == 1: # Tree View
            # We need to traverse the tree to find the item
            iterator = QTreeWidgetItemIterator(self.tree)
            while iterator.value():
                item = iterator.value()
                if item.text(0) == name_to_select:
                    self.tree.setCurrentItem(item)
                    return
                iterator += 1

    # --- Actions ---

    def launch_new_race(self):
        launcher = RaceLauncher(self)
        # The launcher will emit a signal that the main app connects to
        # For now, we can handle it directly for simplicity
        if launcher.exec():
            launch_info = launcher.launch_info
            success, result = self.manager.launch_experiment_race(launch_info)
            if success:
                QMessageBox.information(
                    self,
                    "Race Launched",
                    f"Successfully launched race '{launch_info['base_name']}'.\n\n"
                    f"Baseline failures (if any):\n{result or 'None'}",
                )
                self.refresh_data()
                # Automatically open the monitor for the new race
                self.view_race_monitor(launch_info["base_name"])
            else:
                QMessageBox.critical(
                    self, "Race Launch Failed", f"Could not launch race.\n\nReason: {result}"
                )

    def view_race_monitor(self, race_base_name=None):
        exp_data = self.get_selected_experiment_data()
        if not exp_data:
            return

        race_id = exp_data.get("race_id")
        if not race_id or race_id == "N/A":
            QMessageBox.warning(
                self, "Action Failed", "Please select a race participant to view its monitor."
            )
            return

        # Use race_id as the key for tracking monitor windows
        if race_id in self.race_monitors and self.race_monitors[race_id].isVisible():
            self.race_monitors[race_id].activateWindow()
            return

        # Reliably find all participants using the race_id
        participants = [exp for exp in self.experiments_data if exp.get("race_id") == race_id]

        challenger = next((p for p in participants if "_challenger" in p['name']), None)
        if not challenger:
            # Fallback for single experiment view if no challenger found
            challenger = exp_data

        # Baselines are all other participants in the race
        baselines = [p['model'] for p in participants if p['name'] != challenger['name']]

        # The base name is the common prefix
        race_base_name = os.path.commonprefix([p['name'] for p in participants]).rstrip('_')


        if not challenger:
            QMessageBox.critical(self, "Error", f"Could not find a main participant for race ID '{race_id}'.")
            return

        race_info = {
            "base_name": race_base_name,
            "baselines": baselines,
            "dataset": challenger['dataset'],
            "race_id": race_id
        }

        monitor = RaceMonitor(race_info, self.manager)
        monitor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        monitor.show()
        self.race_monitors[race_id] = monitor
        monitor.destroyed.connect(lambda: self.race_monitors.pop(race_id, None))

    def rename_experiment(self):
        old_name = self.get_selected_experiment_name()
        if not old_name:
            return

        dialog = InputDialog(
            self,
            title="Rename Experiment",
            label="New Name:",
            text=old_name
        )
        new_name = dialog.get_text()

        if new_name:
            success, message = self.manager.rename_experiment(old_name, new_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_data()
                self.select_experiment_by_name(new_name)
            else:
                QMessageBox.warning(self, "Rename Failed", message)

    def clone_experiment(self):
        original_name = self.get_selected_experiment_name()
        if not original_name:
            return

        dialog = InputDialog(
            self,
            title="Clone Experiment",
            label="New Name for Clone:",
            text=f"{original_name}-clone"
        )
        new_name = dialog.get_text()

        if new_name:
            success, message = self.manager.clone_experiment(original_name, new_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_data()
                self.select_experiment_by_name(new_name)
            else:
                QMessageBox.warning(self, "Clone Failed", message)

    def archive_experiment(self):
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Archive",
            f"Are you sure you want to archive '{exp_name}'?\n"
            "It can be restored later from the archive view.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.archive_experiment(exp_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_data()
            else:
                QMessageBox.warning(self, "Archive Failed", message)

    def delete_experiment(self):
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            return

        exp_data = self.get_selected_experiment_data()
        if exp_data.get("status") == STATUS_RUNNING:
            QMessageBox.warning(self, "Action Failed", "Cannot delete a running experiment. Please stop it first.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete the results for '{exp_name}'?\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.delete_single_experiment(exp_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_data()
            else:
                QMessageBox.warning(self, "Deletion Failed", message)

    def delete_race(self):
        exp_data = self.get_selected_experiment_data()
        if not exp_data:
            return

        race_id = exp_data.get("race_id")
        if not race_id or race_id == "N/A":
            QMessageBox.warning(self, "Action Failed", "This experiment is not part of a race.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Race Deletion",
            f"Are you sure you want to permanently delete all experiments for race '{race_id}'?\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.delete_race(race_id)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_data()
            else:
                QMessageBox.warning(self, "Deletion Failed", message)

    def stop_experiment(self):
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            f"Are you sure you want to stop the running experiment '{exp_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self.manager.stop_experiment(exp_name)
            if success:
                QMessageBox.information(self, "Success", f"Stop signal sent to '{exp_name}'.")
                self.refresh_data()
            else:
                QMessageBox.warning(self, "Stop Failed", f"Could not stop experiment '{exp_name}'. It may have already finished.")

    def stop_race(self):
        exp_data = self.get_selected_experiment_data()
        if not exp_data:
            return

        race_id = exp_data.get("race_id")
        if not race_id or race_id == "N/A":
            QMessageBox.warning(self, "Action Failed", "This experiment is not part of a race.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Stop Race",
            f"Are you sure you want to stop all running experiments for race '{race_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.stop_race(race_id)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_data()
            else:
                QMessageBox.warning(self, "Stop Failed", message)


    def _create_context_menu(self, selected_data):
        """Creates and returns a context menu based on the selected item."""
        menu = QMenu()

        is_running = selected_data.get("status") == STATUS_RUNNING
        is_race = selected_data.get("race_id", "N/A") != "N/A"

        # --- General Actions ---
        rename_action = menu.addAction("Rename...")
        rename_action.triggered.connect(self.rename_experiment)
        rename_action.setEnabled(not is_running)

        clone_action = menu.addAction("Clone...")
        clone_action.triggered.connect(self.clone_experiment)

        menu.addAction("View Logs").triggered.connect(self.view_experiment_logs)

        menu.addSeparator()

        # --- Stop Actions ---
        added_stop_action = False
        if is_running:
            menu.addAction("Stop Experiment").triggered.connect(self.stop_experiment)
            added_stop_action = True

        if is_race:
            race_id = selected_data.get("race_id")
            participants = [exp for exp in self.all_experiments_data if exp.get("race_id") == race_id]
            is_race_running = any(p.get("status") == STATUS_RUNNING for p in participants)

            if is_race_running:
                menu.addAction("Stop Race").triggered.connect(self.stop_race)
                added_stop_action = True

        if added_stop_action:
            menu.addSeparator()

        # --- View/Delete Race Actions ---
        if is_race:
            menu.addAction("View Race Monitor").triggered.connect(self.view_race_monitor)
            race_id = selected_data.get("race_id")
            participants = [exp for exp in self.all_experiments_data if exp.get("race_id") == race_id]
            is_race_running = any(p.get("status") == STATUS_RUNNING for p in participants)
            delete_race_action = menu.addAction("Delete Race...")
            delete_race_action.triggered.connect(self.delete_race)
            delete_race_action.setEnabled(not is_race_running)
            menu.addSeparator()

        # --- Archive/Delete Actions ---
        archive_action = menu.addAction("Archive")
        archive_action.triggered.connect(self.archive_experiment)
        archive_action.setEnabled(not is_running)

        delete_action = menu.addAction("Delete Permanently")
        delete_action.triggered.connect(self.delete_experiment)
        delete_action.setEnabled(not is_running)

        return menu

    def show_context_menu(self, pos):
        sender = self.sender()
        selected_data = self.get_selected_experiment_data()

        if not selected_data:
            return

        menu = self._create_context_menu(selected_data)

        if sender == self.table:
            menu.exec(self.table.mapToGlobal(pos))
        elif sender == self.tree:
            menu.exec(self.tree.mapToGlobal(pos))


    def handle_double_click(self, item, column=None):
        """Handles double clicks from both table and tree."""
        exp_data = self.get_selected_experiment_data()
        if exp_data and exp_data.get("race_id") != "N/A":
            # view_race_monitor uses get_selected_experiment_data, so it works for both
            self.view_race_monitor()

    def _update_tree_view(self):
        """Populates the tree view with the experiment hierarchy."""
        self.tree.clear()
        graph = self.manager.get_experiment_graph()
        nodes = graph['nodes']
        edges = graph['edges']
        roots = graph['roots']

        # Create a dictionary of children for easy lookup
        children_map = {}
        for parent, child in edges:
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(child)

        def add_children(parent_item, parent_name):
            if parent_name not in children_map:
                return
            # Sort children by creation date for consistent ordering
            sorted_children = sorted(
                children_map[parent_name],
                key=lambda name: nodes.get(name, {}).get("created", ""),
                reverse=True
            )
            for child_name in sorted_children:
                child_data = nodes[child_name]
                child_item = QTreeWidgetItem([
                    child_data.get('name', 'N/A'),
                    child_data.get('status', 'N/A'),
                    str(child_data.get('final_loss', 'N/A')),
                ])
                child_item.setData(0, Qt.ItemDataRole.UserRole, child_data) # Store data
                parent_item.addChild(child_item)
                add_children(child_item, child_name)

        # Sort roots by creation date
        sorted_roots = sorted(
            roots,
            key=lambda name: nodes.get(name, {}).get("created", ""),
            reverse=True
        )
        for root_name in sorted_roots:
            root_data = nodes[root_name]
            root_item = QTreeWidgetItem([
                root_data.get('name', 'N/A'),
                root_data.get('status', 'N/A'),
                str(root_data.get('final_loss', 'N/A')),
            ])
            root_item.setData(0, Qt.ItemDataRole.UserRole, root_data) # Store data
            self.tree.addTopLevelItem(root_item)
            add_children(root_item, root_name)

        self.tree.expandAll()
        for i in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)


    def _update_combo_box(self, combo, keys):
        current_text = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(keys)
        idx = combo.findText(current_text)
        if idx != -1:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _update_analysis_view(self):
        """
        Gathers data for the N-D analysis plot and populates the selectors.
        """
        all_keys = set()
        self.analysis_data_points = []

        for exp in self.all_experiments_data:
            if exp.get("status") != STATUS_COMPLETED:
                continue

            point = {'name': exp['name']}

            config, err = self.manager.load_experiment_config(exp['name'])
            if config and not err:
                flat_config = flatten_dict(config)
                for k, v in flat_config.items():
                    if isinstance(v, (int, float)):
                        all_keys.add(k)
                        point[k] = v

            results, err = self.manager.load_experiment_results(exp['name'])
            if results and not err:
                # Add final loss from summary if available
                if "final_loss" in exp and exp["final_loss"] != "N/A":
                    try:
                        all_keys.add("final_loss")
                        point["final_loss"] = float(exp["final_loss"])
                    except (ValueError, TypeError):
                        pass

            self.analysis_data_points.append(point)

        sorted_keys = sorted(list(all_keys))
        self._update_combo_box(self.x_axis_combo, sorted_keys)
        self._update_combo_box(self.y_axis_combo, sorted_keys)

        size_keys = ["None"] + sorted_keys
        self._update_combo_box(self.size_combo, size_keys)

        self._update_analysis_plot()

    def _update_analysis_plot(self):
        """
        Updates the scatter plot based on the current axis selections.
        """
        x_key = self.x_axis_combo.currentText()
        y_key = self.y_axis_combo.currentText()
        size_key = self.size_combo.currentText()

        if not x_key or not y_key:
            self.scatter_plot.clear()
            return

        points = []
        sizes = []
        for p in self.analysis_data_points:
            if x_key in p and y_key in p:
                size = 12
                if size_key != "None" and size_key in p:
                    size = p[size_key]
                sizes.append(size)
                points.append({'pos': (p[x_key], p[y_key]), 'data': p})

        if not points:
            self.scatter_plot.clear()
            return

        # Simple size normalization
        min_size_val = min(sizes)
        max_size_val = max(sizes)
        for i, p in enumerate(points):
            if max_size_val > min_size_val and size_key != "None":
                norm_size = 5 + 20 * (sizes[i] - min_size_val) / (max_size_val - min_size_val)
                p['size'] = norm_size
            else:
                p['size'] = 12

        self.scatter_plot.setData(points)
        self.analysis_plot.setLabel('bottom', x_key)
        self.analysis_plot.setLabel('left', y_key)
        if size_key != "None":
            self.analysis_plot.getPlotItem().getAxis('left').setLabel(y_key, units=f"(size by {size_key})")


    def _on_scatter_hover(self, _, points):
        if points:
            p = points[0]
            data = p.data()
            pos = p.pos()
            text = f"{data['name']}\n{self.x_axis_combo.currentText()}: {pos[0]:.4g}\n{self.y_axis_combo.currentText()}: {pos[1]:.4g}"

            size_key = self.size_combo.currentText()
            if size_key != "None" and size_key in data:
                text += f"\n{size_key}: {data[size_key]:.4g}"

            self.analysis_plot_text.setText(text)
            self.analysis_plot_text.setPos(pos[0], pos[1])
            self.analysis_plot_text.show()
        else:
            self.analysis_plot_text.hide()

    def view_experiment_logs(self):
        exp_data = self.get_selected_experiment_data()
        if not exp_data:
            return

        exp_name = exp_data["name"]
        if exp_name in self.log_viewers and self.log_viewers[exp_name].isVisible():
            self.log_viewers[exp_name].activateWindow()
            return

        log_viewer = LogViewer(self.manager, exp_name, self)
        log_viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        log_viewer.show()
        self.log_viewers[exp_name] = log_viewer
        log_viewer.destroyed.connect(lambda: self.log_viewers.pop(exp_name, None))


    def open_archive_browser(self):
        if self.archive_browser is None or not self.archive_browser.isVisible():
            self.archive_browser = ArchiveBrowser(self.manager, self)
            # When the archive browser is closed, it will emit the finished signal.
            # We can use this to refresh the main dashboard if any changes were made.
            self.archive_browser.finished.connect(self.refresh_data)
            self.archive_browser.show()
        else:
            self.archive_browser.activateWindow()

    def closeEvent(self, event):
        # Clean up any open monitor windows
        for monitor in list(self.race_monitors.values()):
            monitor.close()
        # Clean up any open log viewers
        for viewer in list(self.log_viewers.values()):
            viewer.close()
        # Clean up the archive browser if it's open
        if self.archive_browser:
            self.archive_browser.close()
        super().closeEvent(event)
