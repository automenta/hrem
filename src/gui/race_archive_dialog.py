from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QHBoxLayout,
    QLabel,
)
from PyQt6.QtCore import Qt


class RaceArchiveDialog(QDialog):
    """
    A dialog to display a summary of completed races (Hall of Fame).
    """

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Race History - Hall of Fame")
        self.setMinimumSize(800, 500)

        self._init_ui()
        self._populate_data()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Completed Race History")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        self.races_table = QTableWidget()
        self.races_table.setColumnCount(4)
        self.races_table.setHorizontalHeaderLabels(
            ["Race Name", "Winner", "Date Completed", "Participants"]
        )
        self.races_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.races_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.races_table.verticalHeader().setVisible(False)
        self.races_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.races_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.races_table.setSortingEnabled(True)
        layout.addWidget(self.races_table)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

    def _populate_data(self):
        self.races_table.setSortingEnabled(False)
        completed_races = self.manager.get_completed_races()
        self.races_table.setRowCount(len(completed_races))

        for row, race in enumerate(completed_races):
            self.races_table.setItem(row, 0, QTableWidgetItem(race["race_id"]))
            self.races_table.setItem(row, 1, QTableWidgetItem(race["winner"]))
            self.races_table.setItem(row, 2, QTableWidgetItem(race["completed_at"]))
            self.races_table.setItem(
                row, 3, QTableWidgetItem(", ".join(race["participants"]))
            )

        self.races_table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
