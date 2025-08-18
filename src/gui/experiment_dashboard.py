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
    QFormLayout,
    QDialogButtonBox,
    QAbstractItemView,
    QMenu,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QKeySequence

from .experiment_manager import ExperimentManager
from .race_launcher import RaceLauncher
from .race_monitor import RaceMonitor
from .reusable_dialogs import InputDialog
from .constants import STATUS_RUNNING, STATUS_FAILED


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
        self.experiments_data = []
        self.race_monitors = {}  # To track open race monitor windows

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
        self.launch_race_button = QPushButton("🚀 Launch New Race")
        self.launch_race_button.clicked.connect(self.launch_new_race)
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.clicked.connect(self.refresh_data)
        action_layout.addWidget(self.launch_race_button)
        action_layout.addStretch()
        action_layout.addWidget(self.refresh_button)
        layout.addLayout(action_layout)

        # --- Experiment Table ---
        self.table = QTableWidget()
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.doubleClicked.connect(self.handle_double_click)
        layout.addWidget(self.table)

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
        Fetches the latest experiment data and updates the table.
        """
        current_selection = self.get_selected_experiment_name()
        self.experiments_data = sorted(
            self.manager.get_experiments_data(),
            key=lambda x: x.get("created", ""),
            reverse=True,
        )
        self.update_table()
        self.select_experiment_by_name(current_selection)

    def update_table(self):
        """
        Repopulates the QTableWidget with the current experiment data.
        """
        self.table.setRowCount(len(self.experiments_data))
        for row, exp_data in enumerate(self.experiments_data):
            for col, (key, _) in enumerate(self.column_keys):
                item = QTableWidgetItem(str(exp_data.get(key, "N/A")))
                self.table.setItem(row, col, item)

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
        """Returns the full data dict for the selected experiment."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        selected_row_index = selected_rows[0].row()
        return self.experiments_data[selected_row_index]

    def get_selected_experiment_name(self):
        """Returns the name of the selected experiment."""
        exp_data = self.get_selected_experiment_data()
        return exp_data["name"] if exp_data else None

    def select_experiment_by_name(self, name_to_select):
        """Selects a row in the table based on the experiment name."""
        if not name_to_select:
            return
        for row, exp_data in enumerate(self.experiments_data):
            if exp_data["name"] == name_to_select:
                self.table.selectRow(row)
                break

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


    def _create_context_menu(self, selected_data):
        """Creates and returns a context menu based on the selected item."""
        menu = QMenu()

        # Actions available for any selection
        rename_action = menu.addAction("Rename...")
        rename_action.triggered.connect(self.rename_experiment)
        clone_action = menu.addAction("Clone...")
        clone_action.triggered.connect(self.clone_experiment)
        menu.addSeparator()

        # Actions specific to experiment type
        if selected_data.get("race_id") != "N/A":
            view_monitor_action = menu.addAction("View Race Monitor")
            view_monitor_action.triggered.connect(self.view_race_monitor)

            delete_race_action = menu.addAction("Delete Race...")
            delete_race_action.triggered.connect(self.delete_race)
            menu.addSeparator()

        # Actions available for non-running experiments
        is_running = selected_data.get("status") == STATUS_RUNNING

        archive_action = menu.addAction("Archive")
        archive_action.triggered.connect(self.archive_experiment)
        archive_action.setEnabled(not is_running)

        delete_action = menu.addAction("Delete Permanently")
        delete_action.triggered.connect(self.delete_experiment)
        delete_action.setEnabled(not is_running)

        return menu

    def show_context_menu(self, pos):
        selected_data = self.get_selected_experiment_data()
        if not selected_data:
            return

        menu = self._create_context_menu(selected_data)
        menu.exec(self.table.mapToGlobal(pos))

    def handle_double_click(self, model_index):
        if not model_index.isValid():
            return

        exp_data = self.experiments_data[model_index.row()]
        if exp_data.get("race_id") != "N/A":
            self.view_race_monitor()
        # Could add other double-click actions here, e.g., view logs for single experiments

    def closeEvent(self, event):
        # Clean up any open monitor windows
        for monitor in list(self.race_monitors.values()):
            monitor.close()
        super().closeEvent(event)
