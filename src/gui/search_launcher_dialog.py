import json
import os
from PyQt6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTreeView,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QDoubleValidator,
    QCheckBox,
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from .constants import CONFIGS_DIR


class SearchLauncherDialog(QDialog):
    """
    A dialog for interactively creating and launching a hyperparameter search.
    """

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Launch Advanced Hyperparameter Search")
        self.setMinimumSize(800, 600)

        self.base_config = {}
        self.config_model = QStandardItemModel()

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(self)
        layout.addWidget(splitter)

        # --- Left Panel (Config Tree) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.load_button = QPushButton("Load Base Config...")
        self.load_button.clicked.connect(self._load_config)
        self.config_tree = QTreeView()
        self.config_tree.setHeaderHidden(True)
        self.config_tree.setModel(self.config_model)
        self.add_param_button = QPushButton(">> Add to Search >>")
        self.add_param_button.clicked.connect(self._add_param_to_search)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.config_tree)
        left_layout.addWidget(self.add_param_button)
        left_panel.setLayout(left_layout)

        # --- Right Panel (Search Setup) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Search Parameters"))
        self._create_search_table()
        right_layout.addWidget(self.search_params_table)
        right_panel.setLayout(right_layout)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 500])

        # --- Global Search Settings ---
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Search Name:"))
        self.search_name_input = QLineEdit()
        settings_layout.addWidget(self.search_name_input)
        settings_layout.addWidget(QLabel("Trials:"))
        self.trials_input = QLineEdit("20")
        self.trials_input.setValidator(QDoubleValidator())
        settings_layout.addWidget(self.trials_input)
        layout.addLayout(settings_layout)

        # --- Dialog Buttons ---
        button_box = QHBoxLayout()
        self.launch_button = QPushButton("Launch Search")
        self.launch_button.clicked.connect(self._on_accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(self.cancel_button)
        button_box.addWidget(self.launch_button)
        layout.addLayout(button_box)

    def _create_search_table(self):
        self.search_params_table = QTableWidget()
        self.search_params_table.setColumnCount(4)
        self.search_params_table.setHorizontalHeaderLabels(
            ["Parameter", "Type", "Distribution", "Options"]
        )
        self.search_params_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.search_params_table.verticalHeader().setVisible(False)

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Base Config", CONFIGS_DIR, "JSON files (*.json)"
        )
        if path:
            try:
                with open(path, "r") as f:
                    self.base_config = json.load(f)
                self.config_model.clear()
                self._populate_tree(
                    self.base_config, self.config_model.invisibleRootItem()
                )
                self.search_name_input.setText(f"search_{os.path.basename(path)}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load config: {e}")

    def _populate_tree(self, data, parent_item):
        if isinstance(data, dict):
            for key, value in data.items():
                item = QStandardItem(key)
                parent_item.appendRow(item)
                self._populate_tree(value, item)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                item = QStandardItem(f"[{i}]")
                parent_item.appendRow(item)
                self._populate_tree(value, item)

    def _get_selected_param_path(self):
        indexes = self.config_tree.selectedIndexes()
        if not indexes:
            return None

        index = indexes[0]
        path = []
        while index.isValid():
            path.insert(0, index.data())
            index = index.parent()
        return ".".join(path)

    def _add_param_to_search(self):
        path = self._get_selected_param_path()
        if not path:
            QMessageBox.warning(self, "Error", "No parameter selected from the tree.")
            return

        # Check if already added
        for row in range(self.search_params_table.rowCount()):
            if self.search_params_table.item(row, 0).text() == path:
                return

        row_pos = self.search_params_table.rowCount()
        self.search_params_table.insertRow(row_pos)
        self.search_params_table.setItem(row_pos, 0, QTableWidgetItem(path))

        # Add type selector
        type_combo = QComboBox()
        type_combo.addItems(["float", "int", "categorical"])
        self.search_params_table.setCellWidget(row_pos, 1, type_combo)
        type_combo.currentTextChanged.connect(lambda: self._update_row_widgets(row_pos))

        # Initial update
        self._update_row_widgets(row_pos)

    def _update_row_widgets(self, row):
        type_combo = self.search_params_table.cellWidget(row, 1)
        param_type = type_combo.currentText()

        # Distribution cell
        dist_widget = QWidget()
        dist_layout = QHBoxLayout(dist_widget)
        dist_layout.setContentsMargins(2, 2, 2, 2)

        if param_type == "float":
            dist_layout.addWidget(QLabel("Min:"))
            dist_layout.addWidget(QLineEdit())
            dist_layout.addWidget(QLabel("Max:"))
            dist_layout.addWidget(QLineEdit())
            log_check = QCheckBox("Log")
            dist_layout.addWidget(log_check)
        elif param_type == "int":
            dist_layout.addWidget(QLabel("Low:"))
            dist_layout.addWidget(QLineEdit())
            dist_layout.addWidget(QLabel("High:"))
            dist_layout.addWidget(QLineEdit())
        elif param_type == "categorical":
            dist_layout.addWidget(QLabel("Choices:"))
            dist_layout.addWidget(QLineEdit())  # Comma-separated

        self.search_params_table.setCellWidget(row, 2, dist_widget)

        # Options cell (for remove button)
        opts_widget = QWidget()
        opts_layout = QHBoxLayout(opts_widget)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.search_params_table.removeRow(row))
        opts_layout.addWidget(remove_button)
        opts_layout.addStretch()
        self.search_params_table.setCellWidget(row, 3, opts_widget)

    def _on_accept(self):
        # This is where we will construct the final config and launch
        # For now, just a placeholder
        search_name = self.search_name_input.text().strip()
        if not search_name:
            QMessageBox.warning(
                self, "Validation Error", "Search name cannot be empty."
            )
            return

        final_config = self.base_config.copy()
        search_params = {}

        for row in range(self.search_params_table.rowCount()):
            path = self.search_params_table.item(row, 0).text()
            type_combo = self.search_params_table.cellWidget(row, 1)
            dist_widget = self.search_params_table.cellWidget(row, 2)

            param_type = type_combo.currentText()
            param_info = {"type": param_type, "args": [], "kwargs": {}}

            layout = dist_widget.layout()
            if param_type == "float":
                min_val = float(layout.itemAt(1).widget().text())
                max_val = float(layout.itemAt(3).widget().text())
                is_log = layout.itemAt(4).widget().isChecked()
                param_info["args"] = [min_val, max_val]
                if is_log:
                    param_info["kwargs"]["log"] = True
            elif param_type == "int":
                low_val = int(layout.itemAt(1).widget().text())
                high_val = int(layout.itemAt(3).widget().text())
                param_info["args"] = [low_val, high_val]
            elif param_type == "categorical":
                choices_str = layout.itemAt(1).widget().text()
                # need to handle types correctly, e.g., "true" -> True
                param_info["args"] = [c.strip() for c in choices_str.split(",")]

            search_params[path] = param_info

        final_config["search"] = {
            "direction": "minimize",  # placeholder
            "metric": "test_loss",  # placeholder
            "n_trials": int(self.trials_input.text()),
            "params": search_params,
        }
        final_config["experiment_name"] = search_name

        # Launching logic
        success, msg = self.manager.launch_experiment_from_config(
            final_config, search_name
        )
        if success:
            self.accept()
        else:
            QMessageBox.critical(self, "Launch Failed", msg)
