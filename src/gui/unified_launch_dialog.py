import os
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
    QTabWidget,
    QComboBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QStackedWidget,
)
from PyQt6.QtCore import Qt

from .config_editor import ConfigEditor
from .search_config_editor import SearchConfigEditor
from .constants import CONFIGS_DIR, BASE_MODELS_DIR, BASE_DATASETS_DIR


class UnifiedLaunchDialog(QDialog):
    """
    A unified dialog for launching any type of run:
    - Single Experiment
    - Hyperparameter Search
    - Challenge (Race)
    """

    def __init__(self, config=None, exp_name=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New Run")
        self.setMinimumWidth(700)
        self.setMinimumHeight(700)

        self.launch_info = None

        self._init_ui()

        if config:
            # This part might need refinement depending on how we re-launch
            # For now, it primarily supports re-launching single/search runs
            self.single_run_editor.set_config(config)
            self.search_run_editor.set_config(config)
            if "search" in config:
                self.search_config_editor.set_config(config["search"])
            if config.get("notes"):
                self.notes_editor.setText(config["notes"])

        if exp_name:
            self.name_input.setText(exp_name)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Run Type Selection ---
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Run Type:"))
        self.run_type_selector = QComboBox()
        self.run_type_selector.addItems(
            ["Single Experiment", "Hyperparameter Search", "Challenge"]
        )
        type_layout.addWidget(self.run_type_selector)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # --- Name Input (Common to all types) ---
        name_layout = QHBoxLayout()
        self.name_label = QLabel("Experiment Name:")
        name_layout.addWidget(self.name_label)
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # --- Stacked Widget for different run type UIs ---
        self.run_type_stack = QStackedWidget()
        self._create_single_run_panel()
        self._create_search_panel()
        self._create_challenge_panel()
        layout.addWidget(self.run_type_stack)

        self.run_type_selector.currentIndexChanged.connect(self._on_run_type_changed)

        # --- Notes (Common to all types) ---
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout(notes_group)
        self.notes_editor = QTextEdit()
        self.notes_editor.setPlaceholderText("Add any notes about this run...")
        notes_layout.addWidget(self.notes_editor)
        layout.addWidget(notes_group)

        # --- Dialog Buttons (Common to all types) ---
        button_box = QHBoxLayout()
        self.launch_button = QPushButton("Launch")
        self.launch_button.clicked.connect(self._on_accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(self.cancel_button)
        button_box.addWidget(self.launch_button)
        layout.addLayout(button_box)

        self._on_run_type_changed(0)  # Set initial state

    def _on_run_type_changed(self, index):
        self.run_type_stack.setCurrentIndex(index)
        run_type = self.run_type_selector.currentText()
        if run_type == "Single Experiment":
            self.name_label.setText("Experiment Name:")
            self.name_input.setPlaceholderText("e.g., my_awesome_experiment")
            self.launch_button.setText("Launch Experiment")
        elif run_type == "Hyperparameter Search":
            self.name_label.setText("Search Name:")
            self.name_input.setPlaceholderText("e.g., my_hyperparam_search")
            self.launch_button.setText("Launch Search")
        elif run_type == "Challenge":
            self.name_label.setText("Challenge Name:")
            self.name_input.setPlaceholderText("e.g., my_model_vs_baselines")
            self.launch_button.setText("Launch Challenge")

    def _create_single_run_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 5, 0, 0)
        self.single_run_editor = ConfigEditor()
        layout.addWidget(self.single_run_editor)
        self.run_type_stack.addWidget(panel)

    def _create_search_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 5, 0, 0)
        base_config_group = QGroupBox("Base Configuration")
        base_config_layout = QVBoxLayout(base_config_group)
        self.search_run_editor = ConfigEditor()
        base_config_layout.addWidget(self.search_run_editor)
        layout.addWidget(base_config_group)
        self.search_config_editor = SearchConfigEditor()
        layout.addWidget(self.search_config_editor)
        self.run_type_stack.addWidget(panel)

    def _create_challenge_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 5, 0, 0)

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
        self.challenger_button = QPushButton("Select Config...")
        self.challenger_button.clicked.connect(self._select_challenger)
        challenger_file_layout.addWidget(self.challenger_label)
        challenger_file_layout.addStretch()
        challenger_file_layout.addWidget(self.challenger_button)
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

        # --- Advanced Options ---
        self.advanced_options_checkbox = QCheckBox("Advanced Options (Training Overrides)")
        self.advanced_options_group = QGroupBox("Training Overrides")
        advanced_options_layout = QVBoxLayout(self.advanced_options_group)
        self.challenge_training_editor = ConfigEditor(
            visible_sections=["training"],
        )
        advanced_options_layout.addWidget(self.challenge_training_editor)

        self.advanced_options_group.setVisible(False)
        self.advanced_options_checkbox.toggled.connect(
            self.advanced_options_group.setVisible
        )

        layout.addWidget(self.advanced_options_checkbox)
        layout.addWidget(self.advanced_options_group)

        layout.addStretch()
        self.run_type_stack.addWidget(panel)

    def _select_challenger(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Challenger Config", CONFIGS_DIR, "JSON files (*.json)"
        )
        if path:
            self.challenger_config_path = path
            self.challenger_label.setText(os.path.basename(path))

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
            QMessageBox.warning(self, "Validation Error", "Name cannot be empty.")
            return

        notes = self.notes_editor.toPlainText().strip()
        run_type = self.run_type_selector.currentText()

        if run_type == "Single Experiment":
            config = self.single_run_editor.get_config()
            config["experiment_name"] = name
            if notes:
                config["notes"] = notes
            self.launch_info = {"config": config, "name": name, "type": "Single Run"}

        elif run_type == "Hyperparameter Search":
            config = self.search_run_editor.get_config()
            search_config = self.search_config_editor.get_config()
            config["experiment_name"] = name
            config["search"] = search_config
            if notes:
                config["notes"] = notes
            self.launch_info = {
                "config": config,
                "name": name,
                "type": "Hyperparameter Search",
            }

        elif run_type == "Challenge":
            if not hasattr(self, "challenger_config_path") or not self.challenger_config_path:
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

            self.launch_info = {
                "type": "Challenge",
                "base_name": name,
                "challenger_config": self.challenger_config_path,
                "baselines": baselines,
                "dataset": selected_dataset,
            }
            if notes:
                 self.launch_info["notes"] = notes

            if self.advanced_options_checkbox.isChecked():
                overrides = self.challenge_training_editor.get_config()
                if overrides.get("training"):
                    self.launch_info["training_overrides"] = overrides["training"]

        self.accept()

    def get_launch_info(self):
        return self.launch_info
