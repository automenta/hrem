import os
import re
import json
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
    QInputDialog,
    QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal

from .config_editor import ConfigEditor
from .constants import (
    CONFIGS_DIR,
    BASE_MODELS_DIR,
    BASE_DATASETS_DIR,
    TRAINING_PROFILES_DIR,
)

class RaceLauncher(QDialog):
    """
    A dialog for launching a new race (challenge).
    """
    # Signal to emit the launch info dictionary
    launch_info_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New Race")
        self.setMinimumWidth(600)
        self.setMinimumHeight(600)

        self.challenger_config_path = None
        self.challenger_config_data = None
        self.launch_info = None # Will be populated on accept
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
        self.new_challenger_button = QPushButton("New...")
        self.new_challenger_button.setToolTip("Create a new challenger config from a baseline template")
        self.new_challenger_button.clicked.connect(self._create_new_challenger)
        self.select_challenger_button = QPushButton("Select...")
        self.select_challenger_button.setToolTip("Select an existing challenger config file")
        self.select_challenger_button.clicked.connect(self._select_challenger)
        self.edit_challenger_button = QPushButton("View/Edit...")
        self.edit_challenger_button.clicked.connect(self._edit_challenger)
        self.edit_challenger_button.setEnabled(False) # Disabled until a file is chosen
        challenger_file_layout.addWidget(self.challenger_label)
        challenger_file_layout.addStretch()
        challenger_file_layout.addWidget(self.new_challenger_button)
        challenger_file_layout.addWidget(self.select_challenger_button)
        challenger_file_layout.addWidget(self.edit_challenger_button)
        challenger_layout.addLayout(challenger_file_layout)
        layout.addWidget(challenger_group)

        # --- Baselines Selection ---
        baseline_group = QGroupBox("Race Against Baselines")
        baseline_layout = QVBoxLayout(baseline_group)
        self.baseline_list = QListWidget()
        self.baseline_list.setToolTip("Select pre-defined baseline models to race against.")
        baseline_layout.addWidget(self.baseline_list)

        # Add buttons for dynamic baselines
        baseline_button_layout = QHBoxLayout()
        self.add_dynamic_baseline_button = QPushButton("Edit & Add Baseline...")
        self.add_dynamic_baseline_button.setToolTip("Create a new baseline for this race by editing a template.")
        self.add_dynamic_baseline_button.clicked.connect(self._add_dynamic_baseline)
        baseline_button_layout.addStretch()
        baseline_button_layout.addWidget(self.add_dynamic_baseline_button)
        baseline_layout.addLayout(baseline_button_layout)

        layout.addWidget(baseline_group)

        # --- Initialize baselines ---
        self.dynamic_baselines = {} # Will store name: config_dict
        available_baselines = self._get_available_config_names(BASE_MODELS_DIR)
        for baseline_name in available_baselines:
            item = QListWidgetItem(baseline_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.baseline_list.addItem(item)

        # --- Training Overrides ---
        override_group = QGroupBox("Training Overrides (Optional)")
        override_layout = QVBoxLayout(override_group)
        override_form_layout = QFormLayout()

        self.epochs_input = QLineEdit()
        self.epochs_input.setPlaceholderText("e.g., 50")
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setPlaceholderText("e.g., 64")
        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText("e.g., 0.001")

        override_form_layout.addRow("Epochs:", self.epochs_input)
        override_form_layout.addRow("Batch Size:", self.batch_size_input)
        override_form_layout.addRow("Learning Rate:", self.lr_input)
        override_layout.addLayout(override_form_layout)

        profile_button_layout = QHBoxLayout()
        self.save_profile_button = QPushButton("Save Profile...")
        self.save_profile_button.setToolTip("Save the current override settings as a profile for later use.")
        self.save_profile_button.clicked.connect(self._save_training_profile)
        self.load_profile_button = QPushButton("Load Profile...")
        self.load_profile_button.setToolTip("Load override settings from a profile.")
        self.load_profile_button.clicked.connect(self._load_training_profile)
        profile_button_layout.addStretch()
        profile_button_layout.addWidget(self.load_profile_button)
        profile_button_layout.addWidget(self.save_profile_button)
        override_layout.addLayout(profile_button_layout)
        layout.addWidget(override_group)


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

    def _create_new_challenger(self):
        baselines = self._get_available_config_names(BASE_MODELS_DIR)
        if not baselines:
            QMessageBox.warning(self, "No Templates", "No baseline model configs found to use as templates.")
            return

        baseline_name, ok = QInputDialog.getItem(
            self, "Create New Challenger", "Select a template:", baselines, 0, False
        )

        if ok and baseline_name:
            template_path = os.path.join(BASE_MODELS_DIR, f"{baseline_name}.json")
            try:
                with open(template_path, "r") as f:
                    config_data = json.load(f)

                # Open the editor with this template data
                editor = ConfigEditor(config_data, self, is_read_only=False)
                editor.setWindowTitle(f"New Challenger (from {baseline_name})")
                if editor.exec():
                    self.challenger_config_data = editor.get_config()
                    self.challenger_config_path = None # It's an unsaved, custom config
                    self.challenger_label.setText("Unsaved Custom Challenger*")
                    self.edit_challenger_button.setEnabled(True)

            except (json.JSONDecodeError, IOError) as e:
                QMessageBox.critical(
                    self, "Error Reading Template", f"Could not read or parse the template file:\n{e}"
                )


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

    def _save_training_profile(self):
        os.makedirs(TRAINING_PROFILES_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Profile", TRAINING_PROFILES_DIR, "JSON files (*.json)"
        )
        if not path:
            return

        profile_data = {
            "epochs": self.epochs_input.text(),
            "batch_size": self.batch_size_input.text(),
            "learning_rate": self.lr_input.text(),
        }

        try:
            with open(path, "w") as f:
                json.dump(profile_data, f, indent=4)
            QMessageBox.information(self, "Success", f"Profile saved to {os.path.basename(path)}")
        except IOError as e:
            QMessageBox.critical(self, "Error", f"Could not save profile:\n{e}")

    def _load_training_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Training Profile", TRAINING_PROFILES_DIR, "JSON files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                profile_data = json.load(f)

            self.epochs_input.setText(profile_data.get("epochs", ""))
            self.batch_size_input.setText(profile_data.get("batch_size", ""))
            self.lr_input.setText(profile_data.get("learning_rate", ""))
            QMessageBox.information(self, "Success", f"Profile loaded from {os.path.basename(path)}")

        except (json.JSONDecodeError, IOError) as e:
            QMessageBox.critical(self, "Error", f"Could not load profile:\n{e}")

    def _add_dynamic_baseline(self):
        baselines = self._get_available_config_names(BASE_MODELS_DIR)
        if not baselines:
            QMessageBox.warning(self, "No Templates", "No baseline model configs found to use as templates.")
            return

        baseline_name, ok = QInputDialog.getItem(
            self, "Create Dynamic Baseline", "Select a template:", baselines, 0, False
        )

        if ok and baseline_name:
            template_path = os.path.join(BASE_MODELS_DIR, f"{baseline_name}.json")
            try:
                with open(template_path, "r") as f:
                    config_data = json.load(f)

                editor = ConfigEditor(config_data, self, is_read_only=False)
                editor.setWindowTitle(f"New Baseline (from {baseline_name})")
                if editor.exec():
                    new_config = editor.get_config()
                    # Create a unique name for the dynamic baseline
                    dynamic_name = f"{baseline_name}_dynamic_{len(self.dynamic_baselines) + 1}"
                    self.dynamic_baselines[dynamic_name] = new_config

                    # Add it to the list widget and check it by default
                    item = QListWidgetItem(dynamic_name)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)
                    item.setToolTip("This is a dynamically generated baseline for this race.")
                    self.baseline_list.addItem(item)

            except (json.JSONDecodeError, IOError) as e:
                QMessageBox.critical(
                    self, "Error Reading Template", f"Could not read or parse the template file:\n{e}"
                )


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

        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            QMessageBox.warning(
                self,
                "Validation Error",
                "Race Name can only contain letters, numbers, underscores, and hyphens.",
            )
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

        standard_baselines = []
        dynamic_baselines = []
        for i in range(self.baseline_list.count()):
            item = self.baseline_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                baseline_name = item.text()
                if baseline_name in self.dynamic_baselines:
                    dynamic_baselines.append(self.dynamic_baselines[baseline_name])
                else:
                    standard_baselines.append(baseline_name)

        if not standard_baselines and not dynamic_baselines:
            QMessageBox.warning(
                self,
                "Validation Error",
                "You must select at least one baseline model to race against.",
            )
            return

        # --- Collect Training Overrides ---
        training_overrides = {}
        try:
            epochs = self.epochs_input.text().strip()
            if epochs:
                training_overrides["epochs"] = int(epochs)

            batch_size = self.batch_size_input.text().strip()
            if batch_size:
                training_overrides["batch_size"] = int(batch_size)

            lr = self.lr_input.text().strip()
            if lr:
                training_overrides["learning_rate"] = float(lr)

        except ValueError as e:
            QMessageBox.warning(
                self,
                "Validation Error",
                f"Invalid number in training overrides:\n{e}",
            )
            return

        launch_info = {
            "type": "Challenge",
            "base_name": name,
            "challenger_config": self.challenger_config_data,  # Pass the dict directly
            "standard_baselines": standard_baselines,
            "dynamic_baselines": dynamic_baselines,
            "dataset": selected_dataset,
        }
        if training_overrides:
            launch_info["training_overrides"] = training_overrides

        notes = self.notes_editor.toPlainText().strip()
        if notes:
            launch_info["notes"] = notes

        self.launch_info = launch_info
        self.launch_info_ready.emit(launch_info)
        self.accept()
