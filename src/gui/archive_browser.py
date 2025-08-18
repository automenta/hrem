from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QAbstractItemView,
)
from .experiment_manager import ExperimentManager


class ArchiveBrowser(QDialog):
    """
    A dialog for viewing and managing archived experiments.
    """

    def __init__(self, manager: ExperimentManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Archived Experiments")
        self.setGeometry(150, 150, 800, 500)
        self.setModal(False) # Allow interaction with the main window

        # --- UI ---
        self._init_ui()

        # --- Initial Load ---
        self.refresh_table()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

        # Define table columns
        self.column_keys = [
            ("name", "Experiment Name"),
            ("model", "Model"),
            ("dataset", "Dataset"),
            ("created", "Archived Date"),
        ]
        self.table.setColumnCount(len(self.column_keys))
        self.table.setHorizontalHeaderLabels([label for _, label in self.column_keys])

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.restore_button = QPushButton("Restore")
        self.restore_button.clicked.connect(self.restore_selected)
        button_layout.addWidget(self.restore_button)

        self.delete_button = QPushButton("Delete Permanently")
        self.delete_button.clicked.connect(self.delete_selected)
        button_layout.addWidget(self.delete_button)

        button_layout.addStretch()

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_table)
        button_layout.addWidget(self.refresh_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept) # `accept` closes the dialog
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def refresh_table(self):
        """
        Fetches the latest archived experiment data and repopulates the table.
        """
        self.table.setSortingEnabled(False)
        self.table.clearContents()

        archived_data = self.manager.get_archived_experiments_data()
        self.table.setRowCount(len(archived_data))

        for row, exp_data in enumerate(archived_data):
            for col, (key, _) in enumerate(self.column_keys):
                value_str = str(exp_data.get(key, "N/A"))
                item = QTableWidgetItem(value_str)
                self.table.setItem(row, col, item)

        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    def get_selected_experiment_name(self) -> str | None:
        """Returns the name of the selected experiment in the table."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select an experiment first.")
            return None
        # Name is in the first column
        return self.table.item(selected_rows[0].row(), 0).text()

    def restore_selected(self):
        """
        Restores the selected experiment from the archive.
        """
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            return

        success, message = self.manager.restore_experiment(exp_name)
        if success:
            QMessageBox.information(self, "Success", message)
            self.refresh_table()
        else:
            QMessageBox.critical(self, "Restore Failed", message)

    def delete_selected(self):
        """
        Permanently deletes the selected experiment from the archive.
        """
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Permanent Deletion",
            f"Are you sure you want to permanently delete the archived experiment '{exp_name}'?\n\n"
            "This action CANNOT be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.delete_experiment_permanently(exp_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_table()
            else:
                QMessageBox.critical(self, "Deletion Failed", message)
