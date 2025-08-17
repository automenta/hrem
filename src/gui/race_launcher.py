import os
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QMessageBox,
    QWidget,
    QGroupBox,
    QComboBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from .config_editor import ConfigEditor
from .constants import CONFIGS_DIR, BASE_MODELS_DIR, BASE_DATASETS_DIR

class RaceLauncher(QDialog):
    """
    A dialog for launching a new race (challenge).
    """
    # Signal to emit the launch info dictionary
    launch_info_ready = pyqtSignal(dict)

import json

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New Race")
        self.setMinimumWidth(600)
        self.setMinimumHeight(600)

        self.challenger_config_path = None
        self.challenger_config_data = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Name Input ---
        name_layout = QHBoxLayout()
        name_label = QLabel("Race Name:")
        name_layout.addWidget(name_label)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., my_model_vs_baselines")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # --- Task Selection ---
        task_group = QGroupBox("Task / Dataset")
        task_layout = QHBoxLayout(task_group)
        task_layout.addWidget(QLabel("Task:"))
        self.dataset_selector = QComboBox()
        task_layout.addWidget(self.dataset_selector)
        available_datasets = self._get_available_config_names(BASE_DATASETS_DIR)
        self.dataset_selector.addItems(available_datasets)
        layout.addWidget(task_group)

        # --- Challenger Selection ---
        challenger_group = QGroupBox("Challenger Model")
        challenger_layout = QVBoxLayout(challenger_group)
        challenger_file_layout = QHBoxLayout()
        self.challenger_label = QLabel("No file selected...")
        self.select_challenger_button = QPushButton("Select Config...")
        self.select_challenger_button.clicked.connect(self._select_challenger)
        self.edit_challenger_button = QPushButton("View/Edit...")
        self.edit_challenger_button.clicked.connect(self._edit_challenger)
        self.edit_challenger_button.setEnabled(False) # Disabled until a file is chosen
        challenger_file_layout.addWidget(self.challenger_label)
        challenger_file_layout.addStretch()
        challenger_file_layout.addWidget(self.select_challenger_button)
        challenger_file_layout.addWidget(self.edit_challenger_button)
        challenger_layout.addLayout(challenger_file_layout)
        layout.addWidget(challenger_group)

        # --- Baselines Selection ---
        baseline_group = QGroupBox("Race Against Baselines")
        baseline_layout = QVBoxLayout(baseline_group)
        self.baseline_list = QListWidget()
        baseline_layout.addWidget(self.baseline_list)
        layout.addWidget(baseline_group)

        available_baselines = self._get_available_config_names(BASE_MODELS_DIR)
        for baseline_name in available_baselines:
            item = QListWidgetItem(baseline_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.baseline_list.addItem(item)

        # --- Notes ---
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout(notes_group)
        self.notes_editor = QTextEdit()
        self.notes_editor.setPlaceholderText("Add any notes about this race...")
        notes_layout.addWidget(self.notes_editor)
        layout.addWidget(notes_group)

        # --- Dialog Buttons ---
        button_box = QHBoxLayout()
        self.launch_button = QPushButton("Launch Race")
        self.launch_button.clicked.connect(self._on_accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(self.cancel_button)
        button_box.addWidget(self.launch_button)
        layout.addLayout(button_box)

    def _select_challenger(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Challenger Config", CONFIGS_DIR, "JSON files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                self.challenger_config_data = json.load(f)

            self.challenger_config_path = path
            self.challenger_label.setText(os.path.basename(path))
            self.edit_challenger_button.setEnabled(True)

        except (json.JSONDecodeError, IOError) as e:
            QMessageBox.critical(
                self, "Error Reading Config", f"Could not read or parse the config file:\n{e}"
            )
            self.challenger_config_path = None
            self.challenger_config_data = None
            self.challenger_label.setText("No file selected...")
            self.edit_challenger_button.setEnabled(False)

    def _edit_challenger(self):
        if not self.challenger_config_data:
            return

        editor = ConfigEditor(self.challenger_config_data, self, is_read_only=False)
        editor.setWindowTitle("Edit Challenger Configuration")
        if editor.exec():
            self.challenger_config_data = editor.get_config()
            # Indicate that the config has been modified
            if self.challenger_config_path:
                self.challenger_label.setText(f"{os.path.basename(self.challenger_config_path)}* (modified)")
            else:
                self.challenger_label.setText("Custom Config* (modified)")

    def _get_available_config_names(self, directory):
        """Scans a directory for available JSON configuration files."""
        names = []
        if not os.path.isdir(directory):
            return names
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                names.append(os.path.splitext(filename)[0])
        names.sort()
        return names

    def _on_accept(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Race Name cannot be empty.")
            return

        if not self.challenger_config_data:
            QMessageBox.warning(
                self,
                "Validation Error",
                "You must select a challenger model config.",
            )
            return

        selected_dataset = self.dataset_selector.currentText()
        if not selected_dataset:
            QMessageBox.warning(
                self, "Validation Error", "You must select a task/dataset."
            )
            return

        baselines = []
        for i in range(self.baseline_list.count()):
            item = self.baseline_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                baselines.append(item.text())

        if not baselines:
            QMessageBox.warning(
                self,
                "Validation Error",
                "You must select at least one baseline model to race against.",
            )
            return

        launch_info = {
            "type": "Challenge",
            "base_name": name,
            "challenger_config": self.challenger_config_data, # Pass the dict directly
            "baselines": baselines,
            "dataset": selected_dataset,
        }

        notes = self.notes_editor.toPlainText().strip()
        if notes:
             launch_info["notes"] = notes

        self.launch_info_ready.emit(launch_info)
        self.accept()
