import os
import json
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QGroupBox,
    QPushButton, QHBoxLayout, QListWidget, QListWidgetItem, QComboBox,
    QDialogButtonBox, QMessageBox, QFileDialog, QInputDialog, QLabel, QTextEdit
)
from PyQt6.QtCore import Qt

from .config_editor import ConfigEditor
from .constants import CONFIGS_DIR, BASE_MODELS_DIR, BASE_DATASETS_DIR

class RaceLauncherDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New Race")
        self.setMinimumSize(700, 600)

        self.challenger_config_data = None
        self.launch_info = None

        self._init_ui()
        self._populate_lists()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Race Details ---
        details_group = QGroupBox("Race Details")
        details_layout = QFormLayout(details_group)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., my_transformer_vs_lstm")
        self.notes_editor = QTextEdit()
        self.notes_editor.setPlaceholderText("Describe the goal of this race...")
        self.notes_editor.setFixedHeight(80)
        details_layout.addRow("Race Name:", self.name_input)
        details_layout.addRow("Notes:", self.notes_editor)
        main_layout.addWidget(details_group)

        # --- Challenger ---
        challenger_group = QGroupBox("Challenger Model")
        challenger_layout = QVBoxLayout(challenger_group)
        self.challenger_label = QLabel("No challenger configuration loaded.")
        challenger_layout.addWidget(self.challenger_label)

        challenger_buttons = QHBoxLayout()
        new_button = QPushButton("New from Template...")
        select_button = QPushButton("Select from File...")
        self.edit_button = QPushButton("Edit Config...")
        self.edit_button.setEnabled(False)
        challenger_buttons.addWidget(new_button)
        challenger_buttons.addWidget(select_button)
        challenger_buttons.addWidget(self.edit_button)
        challenger_buttons.addStretch()
        challenger_layout.addLayout(challenger_buttons)
        main_layout.addWidget(challenger_group)

        # --- Baselines ---
        baselines_group = QGroupBox("Baselines")
        baselines_layout = QVBoxLayout(baselines_group)
        self.baseline_list = QListWidget()
        self.baseline_list.setToolTip("Select one or more standard baseline models.")
        baselines_layout.addWidget(self.baseline_list)
        main_layout.addWidget(baselines_group)

        # --- Task & Training ---
        task_group = QGroupBox("Task and Training Overrides")
        task_layout = QFormLayout(task_group)
        self.dataset_selector = QComboBox()
        self.epochs_input = QLineEdit()
        self.batch_size_input = QLineEdit()
        self.lr_input = QLineEdit()
        task_layout.addRow("Task / Dataset:", self.dataset_selector)
        task_layout.addRow("Epochs Override:", self.epochs_input)
        task_layout.addRow("Batch Size Override:", self.batch_size_input)
        task_layout.addRow("Learning Rate Override:", self.lr_input)
        main_layout.addWidget(task_group)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

        # --- Connections ---
        new_button.clicked.connect(self.create_new_challenger)
        select_button.clicked.connect(self.select_challenger)
        self.edit_button.clicked.connect(self.edit_challenger)
        self.button_box.accepted.connect(self.on_accept)
        self.button_box.rejected.connect(self.reject)

    def _populate_lists(self):
        # Populate baselines
        available_baselines = self._get_available_config_names(BASE_MODELS_DIR)
        for name in available_baselines:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.baseline_list.addItem(item)

        # Populate datasets
        available_datasets = self._get_available_config_names(BASE_DATASETS_DIR)
        self.dataset_selector.addItems([""] + available_datasets)

    def _get_available_config_names(self, directory):
        names = []
        if not os.path.isdir(directory):
            return names
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                names.append(os.path.splitext(filename)[0])
        names.sort()
        return names

    def create_new_challenger(self):
        baselines = self._get_available_config_names(BASE_MODELS_DIR)
        if not baselines:
            QMessageBox.warning(self, "No Templates", "No baseline model configs found.")
            return

        baseline_name, ok = QInputDialog.getItem(self, "Create New Challenger", "Select a model template:", baselines, 0, False)
        if ok and baseline_name:
            template_path = os.path.join(BASE_MODELS_DIR, f"{baseline_name}.json")
            try:
                with open(template_path, "r") as f:
                    model_config_data = json.load(f)

                # Create a full, new config from the model template
                new_config = {
                    "experiment_name": "new_challenger_config",
                    "model": model_config_data,
                    "dataset": {"name": "", "params": {}},
                    "training": {
                        "epochs": 20,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    }
                }

                editor = ConfigEditor(new_config, self) # Edit the full config
                editor.setWindowTitle(f"New Challenger (from {baseline_name})")

                if editor.exec():
                    self.challenger_config_data = editor.get_config()
                    self.challenger_label.setText("Unsaved Custom Challenger*")
                    self.edit_button.setEnabled(True)
                    # Update the dataset selector to reflect the new config
                    dataset_name = self.challenger_config_data.get("dataset", {}).get("name")
                    if dataset_name:
                        idx = self.dataset_selector.findText(dataset_name)
                        if idx != -1:
                            self.dataset_selector.setCurrentIndex(idx)

            except (json.JSONDecodeError, IOError) as e:
                QMessageBox.critical(self, "Error Reading Template", f"Could not read template: {e}")

    def select_challenger(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Challenger Config", CONFIGS_DIR, "JSON files (*.json)")
        if path:
            try:
                with open(path, "r") as f:
                    # Load the entire configuration.
                    self.challenger_config_data = json.load(f)
                self.challenger_label.setText(os.path.basename(path))
                self.edit_button.setEnabled(True)
                # Update the dataset selector to reflect the loaded config
                dataset_name = self.challenger_config_data.get("dataset", {}).get("name")
                if dataset_name:
                    idx = self.dataset_selector.findText(dataset_name)
                    if idx != -1:
                        self.dataset_selector.setCurrentIndex(idx)

            except (json.JSONDecodeError, IOError) as e:
                QMessageBox.critical(self, "Error Reading Config", f"Could not read config: {e}")
                self.challenger_config_data = None
                self.challenger_label.setText("No challenger configuration loaded.")
                self.edit_button.setEnabled(False)

    def edit_challenger(self):
        if not self.challenger_config_data:
            return

        editor = ConfigEditor(
            self.challenger_config_data,
            self
        )
        editor.setWindowTitle("Edit Challenger Config")

        if editor.exec():
            self.challenger_config_data = editor.get_config()
            self.challenger_label.setText(f"{self.challenger_label.text().split('*')[0]}* (modified)")
            # Update the dataset selector to reflect the edited config
            dataset_name = self.challenger_config_data.get("dataset", {}).get("name")
            if dataset_name:
                idx = self.dataset_selector.findText(dataset_name)
                if idx != -1:
                    self.dataset_selector.setCurrentIndex(idx)

    def on_accept(self):
        # --- Validation ---
        race_name = self.name_input.text().strip()
        if not race_name:
            QMessageBox.warning(self, "Validation Error", "Race Name cannot be empty.")
            return

        if not self.challenger_config_data:
            QMessageBox.warning(self, "Validation Error", "A challenger model must be configured.")
            return

        dataset = self.dataset_selector.currentText()
        if not dataset:
            QMessageBox.warning(self, "Validation Error", "A dataset must be selected for the race.")
            return

        selected_baselines = [self.baseline_list.item(i).text() for i in range(self.baseline_list.count()) if self.baseline_list.item(i).checkState() == Qt.CheckState.Checked]

        # --- Training Overrides ---
        training_overrides = {}
        try:
            if self.epochs_input.text().strip(): training_overrides["epochs"] = int(self.epochs_input.text())
            if self.batch_size_input.text().strip(): training_overrides["batch_size"] = int(self.batch_size_input.text())
            if self.lr_input.text().strip(): training_overrides["learning_rate"] = float(self.lr_input.text())
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", f"Invalid number in training overrides: {e}")
            return

        # --- Final Challenger Config ---
        # The user can select a dataset that overrides what's in the challenger config file.
        final_challenger_config = self.challenger_config_data.copy()
        final_challenger_config["dataset"] = {"name": dataset, "params": {}}

        # --- Build Launch Info ---
        self.launch_info = {
            "base_name": race_name,
            "notes": self.notes_editor.toPlainText().strip(),
            "challenger_config": final_challenger_config,
            "standard_baselines": selected_baselines,
            "training_overrides": training_overrides
        }

        self.accept()

    def get_launch_info(self):
        return self.launch_info
