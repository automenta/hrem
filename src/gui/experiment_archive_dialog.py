import os
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QHeaderView,
)

from .experiment_manager import ExperimentManager


class ExperimentArchiveDialog(QDialog):
    """
    A dialog for managing archived experiments.
    """

    def __init__(self, manager: ExperimentManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Archived Experiments")
        self.setMinimumSize(600, 400)

        self._init_ui()
        self.populate_archive_table()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.archive_table = QTableWidget()
        self.archive_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.archive_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.archive_table.verticalHeader().setVisible(False)
        header = self.archive_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.archive_table)

        button_layout = QHBoxLayout()
        self.restore_button = QPushButton("Restore")
        self.restore_button.clicked.connect(self.restore_selected)
        self.delete_button = QPushButton("Delete Permanently")
        self.delete_button.clicked.connect(self.delete_selected)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        button_layout.addWidget(self.restore_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

    def populate_archive_table(self):
        self.archive_table.setRowCount(0)
        archived_data = self.manager.get_archived_experiments_data()

        headers = ["Name", "Model", "Dataset", "Archived Date"]
        self.archive_table.setColumnCount(len(headers))
        self.archive_table.setHorizontalHeaderLabels(headers)

        for row, data in enumerate(archived_data):
            self.archive_table.insertRow(row)
            self.archive_table.setItem(row, 0, QTableWidgetItem(data["name"]))
            self.archive_table.setItem(row, 1, QTableWidgetItem(data["model"]))
            self.archive_table.setItem(row, 2, QTableWidgetItem(data["dataset"]))
            self.archive_table.setItem(row, 3, QTableWidgetItem(data["created"]))

    def get_selected_experiment_name(self):
        selected_rows = self.archive_table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        return self.archive_table.item(selected_rows[0].row(), 0).text()

    def restore_selected(self):
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            QMessageBox.warning(
                self, "No Selection", "Please select an experiment to restore."
            )
            return

        success, message = self.manager.restore_experiment(exp_name)
        if success:
            QMessageBox.information(self, "Success", message)
            self.populate_archive_table()
        else:
            QMessageBox.warning(self, "Error", message)

    def delete_selected(self):
        exp_name = self.get_selected_experiment_name()
        if not exp_name:
            QMessageBox.warning(
                self, "No Selection", "Please select an experiment to delete."
            )
            return

        reply = QMessageBox.question(
            self,
            "Delete Permanently",
            f"Are you sure you want to permanently delete '{exp_name}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.delete_experiment_permanently(exp_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.populate_archive_table()
            else:
                QMessageBox.warning(self, "Error", message)
