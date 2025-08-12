import json
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal


class SearchConfigEditor(QWidget):
    """
    A widget for editing the 'search' section of a hyperparameter search config.
    """

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        search_group = QGroupBox("Hyperparameter Search Settings")
        group_layout = QVBoxLayout(search_group)
        layout.addWidget(search_group)

        # --- General Search Settings ---
        form_layout = QFormLayout()
        self.n_trials_input = QLineEdit("20")
        self.n_trials_input.textChanged.connect(self.config_changed)
        form_layout.addRow("Number of Trials:", self.n_trials_input)

        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(["TPESampler", "RandomSampler"])
        self.sampler_combo.currentTextChanged.connect(self.config_changed)
        form_layout.addRow("Sampler:", self.sampler_combo)
        group_layout.addLayout(form_layout)

        # --- Search Space Table ---
        search_space_group = QGroupBox("Search Space")
        search_space_layout = QVBoxLayout(search_space_group)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(
            ["Parameter Path", "Type", "Distribution (JSON)"]
        )
        self.table.itemChanged.connect(self.config_changed)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        search_space_layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Parameter")
        remove_button = QPushButton("Remove Selected")
        add_button.clicked.connect(self._add_row)
        remove_button.clicked.connect(self._remove_row)
        button_layout.addStretch()
        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)
        search_space_layout.addLayout(button_layout)

        group_layout.addWidget(search_space_group)

        # Add a default row for convenience
        self._add_row("training.learning_rate", "float", "[1e-5, 1e-1]")

    def _add_row(self, path="", type="float", distribution=""):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        # Parameter Path
        self.table.setItem(row_position, 0, QTableWidgetItem(path))

        # Type ComboBox
        type_combo = QComboBox()
        type_combo.addItems(["float", "int", "categorical"])
        if type in ["float", "int", "categorical"]:
            type_combo.setCurrentText(type)
        self.table.setCellWidget(row_position, 1, type_combo)

        # Distribution
        self.table.setItem(row_position, 2, QTableWidgetItem(distribution))

    def _remove_row(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def set_config(self, search_config: dict):
        """
        Populates the editor with an existing search configuration.
        """
        if not search_config:
            return

        self.n_trials_input.setText(str(search_config.get("n_trials", 20)))
        self.sampler_combo.setCurrentText(search_config.get("sampler", "TPESampler"))

        # Clear existing rows
        self.table.setRowCount(0)

        params = search_config.get("params", {})
        for path, details in params.items():
            param_type = details.get("type")
            distribution = details.get("distribution")
            dist_str = json.dumps(distribution)
            self._add_row(path, param_type, dist_str)

    def get_config(self) -> dict:
        """
        Constructs the 'search' dictionary from the UI elements.
        """
        params = {}
        for row in range(self.table.rowCount()):
            path_item = self.table.item(row, 0)
            type_widget = self.table.cellWidget(row, 1)
            dist_item = self.table.item(row, 2)

            if not (path_item and type_widget and dist_item):
                continue

            path = path_item.text()
            param_type = type_widget.currentText()
            dist_str = dist_item.text()

            if not path:
                continue

            try:
                # Use JSON to parse the distribution which is safer than eval
                distribution = json.loads(dist_str)
            except json.JSONDecodeError:
                # If JSON fails, for categorical it might just be a list of strings
                # This is a simplification. A real implementation might need more robust parsing.
                if param_type == "categorical":
                    distribution = [s.strip() for s in dist_str.strip("[]").split(",")]
                else:
                    continue  # Skip malformed rows

            params[path] = {
                "type": param_type,
                "distribution": distribution,
            }

        return {
            "n_trials": int(self.n_trials_input.text()),
            "sampler": self.sampler_combo.currentText(),
            "params": params,
        }
