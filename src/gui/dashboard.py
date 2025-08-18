from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHeaderView,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class DashboardWindow(QMainWindow):
    """
    The main dashboard window for managing and viewing all experiment races.
    """
    # Signal to request launching the race launcher dialog
    launch_new_race = pyqtSignal()
    # Signal to request monitoring an existing race
    monitor_race = pyqtSignal(dict)


    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Mission Control Dashboard")
        self.setGeometry(100, 100, 1000, 700)

        self._init_ui()
        self.populate_experiments_table()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Toolbar ---
        toolbar_layout = QHBoxLayout()
        self.launch_button = QPushButton("🚀 Launch New Race")
        self.launch_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.monitor_button = QPushButton("👀 Monitor Selected Race")
        self.monitor_button.setEnabled(False) # Enabled on selection
        self.refresh_button = QPushButton("🔄 Refresh")
        self.delete_button = QPushButton("🗑️ Delete Selected Race")
        self.delete_button.setEnabled(False) # Enabled on selection
        toolbar_layout.addWidget(self.launch_button)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.monitor_button)
        toolbar_layout.addWidget(self.delete_button)
        toolbar_layout.addWidget(self.refresh_button)
        layout.addLayout(toolbar_layout)

        # --- Experiments Table ---
        self.experiments_table = QTableWidget()
        self.experiments_table.setColumnCount(5)
        self.experiments_table.setHorizontalHeaderLabels(
            ["Race Name", "Status", "Challenger", "Baselines", "Created At"]
        )
        self.experiments_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.experiments_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.experiments_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.experiments_table.verticalHeader().setVisible(False)
        self.experiments_table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.experiments_table)

        # --- Connect signals ---
        self.refresh_button.clicked.connect(self.populate_experiments_table)
        self.launch_button.clicked.connect(self.launch_new_race.emit)
        self.monitor_button.clicked.connect(self._on_monitor_clicked)
        self.delete_button.clicked.connect(self._on_delete_clicked)


    def populate_experiments_table(self):
        """
        Fetches all race data from the ExperimentManager and populates the table.
        """
        self.experiments_table.setSortingEnabled(False)
        self.experiments_table.setRowCount(0) # Clear table
        all_races = self.manager.get_all_races_summary()
        for race in all_races:
            self.add_race_to_table(race)
        self.experiments_table.setSortingEnabled(True)


    def add_race_to_table(self, race_data):
        row_position = self.experiments_table.rowCount()
        self.experiments_table.insertRow(row_position)

        name_item = QTableWidgetItem(race_data.get("base_name", "N/A"))
        # Store the raw race_id in the item for later retrieval
        name_item.setData(Qt.ItemDataRole.UserRole, race_data.get("race_id"))

        self.experiments_table.setItem(row_position, 0, name_item)
        self.experiments_table.setItem(
            row_position, 1, QTableWidgetItem(race_data.get("status", "N/A"))
        )
        self.experiments_table.setItem(
            row_position, 2, QTableWidgetItem(race_data.get("challenger", "N/A"))
        )
        self.experiments_table.setItem(
            row_position, 3, QTableWidgetItem(", ".join(race_data.get("baselines", [])))
        )
        self.experiments_table.setItem(
            row_position, 4, QTableWidgetItem(race_data.get("created_at", "N/A"))
        )


    def _on_selection_changed(self):
        """
        Enables or disables buttons based on the current table selection.
        """
        selected_rows = self.experiments_table.selectionModel().selectedRows()
        is_selection_valid = len(selected_rows) == 1

        self.delete_button.setEnabled(is_selection_valid)

        if is_selection_valid:
            status_item = self.experiments_table.item(selected_rows[0].row(), 1)
            is_running = status_item and status_item.text() == "Running"
            self.monitor_button.setEnabled(is_running)
        else:
            self.monitor_button.setEnabled(False)

    def _on_monitor_clicked(self):
        selected_rows = self.experiments_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        race_id_item = self.experiments_table.item(selected_rows[0].row(), 0)
        race_id = race_id_item.data(Qt.ItemDataRole.UserRole)

        launch_info = self.manager.get_race_info(race_id)
        if launch_info:
            self.monitor_race.emit(launch_info)
        else:
            QMessageBox.critical(
                self, "Error", f"Could not reconstruct information for race {race_id}."
            )

    def _on_delete_clicked(self):
        selected_rows = self.experiments_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        race_id_item = self.experiments_table.item(selected_rows[0].row(), 0)
        race_id = race_id_item.data(Qt.ItemDataRole.UserRole)
        base_name = race_id_item.text()

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete all experiments for race '{base_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.delete_race(race_id)
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.warning(self, "Error", message)
            self.populate_experiments_table()
