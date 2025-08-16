import json
import os
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QMessageBox,
    QStackedWidget,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QGroupBox,
)
from PyQt6.QtCore import Qt

from .constants import CONFIGS_DIR, BASE_MODELS_DIR
from .config_editor import ConfigEditor
from .search_config_editor import SearchConfigEditor


class UnifiedLaunchDialog(QDialog):
    """
    A dialog for launching a new experiment, challenge, or hyperparameter search.
    """

    def __init__(self, config=None, exp_name=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New...")
        self.setMinimumWidth(700)
        self.setMinimumHeight(750)

        self.launch_info = None
        self.challenger_config_path = None

        self._init_ui()
        self._on_run_type_changed(self.run_type_selector.currentText())

        if config:
            # If it's a search config, switch to that tab first
            if "search" in config:
                self.run_type_selector.setCurrentText("Hyperparameter Search")
                self.search_config_editor.set_config(config["search"])

            # Set the base config in both editors that have a ConfigEditor
            self.single_run_editor.set_config(config)
            self.search_run_editor.set_config(config)

            # Set notes if they exist
            if config.get("notes"):
                self.notes_editor.setText(config["notes"])

        if exp_name:
            self.exp_name_input.setText(exp_name)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Experiment Name ---
        name_layout = QHBoxLayout()
        self.name_label = QLabel("Name:")
        name_layout.addWidget(self.name_label)
        self.exp_name_input = QLineEdit()
        name_layout.addWidget(self.exp_name_input)
        layout.addLayout(name_layout)

        # --- Run Type ---
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Run Type:"))
        self.run_type_selector = QComboBox()
        self.run_type_selector.addItems(
            ["Single Run", "Challenge", "Hyperparameter Search"]
        )
        self.run_type_selector.currentTextChanged.connect(self._on_run_type_changed)
        type_layout.addWidget(self.run_type_selector)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # --- Main Panel ---
        self.main_panel = QStackedWidget()
        layout.addWidget(self.main_panel)

        self._create_single_run_panel()
        self._create_challenge_panel()
        self._create_search_panel()

        # --- Notes ---
        self.notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout(self.notes_group)
        self.notes_editor = QTextEdit()
        self.notes_editor.setPlaceholderText("Add any notes about this run...")
        notes_layout.addWidget(self.notes_editor)
        layout.addWidget(self.notes_group)

        # --- Dialog Buttons ---
        button_box = QHBoxLayout()
        self.launch_button = QPushButton("Launch")
        self.launch_button.clicked.connect(self._on_accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(self.cancel_button)
        button_box.addWidget(self.launch_button)
        layout.addLayout(button_box)

    def _create_single_run_panel(self):
        """Creates the panel for single runs, which use a form-based editor."""
        self.single_run_panel = QWidget()
        layout = QVBoxLayout(self.single_run_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.single_run_editor = ConfigEditor()
        self.single_run_editor.config_changed.connect(self._update_json_preview)
        layout.addWidget(self.single_run_editor)

        self.main_panel.addWidget(self.single_run_panel)

    def _create_search_panel(self):
        """Creates the panel for hyperparameter searches."""
        self.search_panel = QWidget()
        layout = QVBoxLayout(self.search_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Editor for the base configuration (model, dataset, training defaults)
        base_config_group = QGroupBox("Base Configuration")
        base_config_layout = QVBoxLayout(base_config_group)
        self.search_run_editor = ConfigEditor()
        self.search_run_editor.config_changed.connect(self._update_json_preview)
        base_config_layout.addWidget(self.search_run_editor)
        layout.addWidget(base_config_group)

        # Editor for the search-specific parameters
        self.search_config_editor = SearchConfigEditor()
        self.search_config_editor.config_changed.connect(self._update_json_preview)
        layout.addWidget(self.search_config_editor)

        self.main_panel.addWidget(self.search_panel)

    def _update_json_preview(self):
        run_type = self.run_type_selector.currentText()
        config = {}
        if run_type == "Single Run":
            config = self.single_run_editor.get_config()
        elif run_type == "Hyperparameter Search":
            config = self.search_run_editor.get_config()
            search_config = self.search_config_editor.get_config()
            config["search"] = search_config

        if run_type == "Challenge":
            self.json_preview.clear()
        else:
            self.json_preview.setText(json.dumps(config, indent=4))

    def _create_challenge_panel(self):
        """Creates the panel for launching a challenge."""
        self.challenge_panel = QWidget()
        layout = QVBoxLayout(self.challenge_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Challenger Selection
        challenger_group = QGroupBox("Challenger Model")
        challenger_group_layout = QVBoxLayout(challenger_group)
        challenger_layout = QHBoxLayout()
        self.challenger_label = QLabel("No file selected...")
        self.challenger_button = QPushButton("Select Config...")
        self.challenger_button.clicked.connect(self._select_challenger)
        challenger_layout.addWidget(self.challenger_label)
        challenger_layout.addStretch()
        challenger_layout.addWidget(self.challenger_button)
        challenger_group_layout.addLayout(challenger_layout)
        layout.addWidget(challenger_group)

        # Baselines Selection
        baseline_group = QGroupBox("Race Against Baselines")
        baseline_group_layout = QVBoxLayout(baseline_group)
        self.baseline_list = QListWidget()
        baseline_group_layout.addWidget(self.baseline_list)
        layout.addWidget(baseline_group)

        # Populate baselines
        available_baselines = self._get_available_baselines()
        for baseline_name in available_baselines:
            item = QListWidgetItem(baseline_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.baseline_list.addItem(item)

        layout.addStretch()
        self.main_panel.addWidget(self.challenge_panel)

    def _on_run_type_changed(self, run_type):
        """Switches the visible panel and updates UI elements based on the selected run type."""
        self.name_label.setText("Name:")
        if run_type == "Challenge":
            self.main_panel.setCurrentWidget(self.challenge_panel)
            self.exp_name_input.setPlaceholderText("e.g., my_model_vs_baselines")
            self.setWindowTitle("Launch New Challenge")
            self.launch_button.setText("Launch Challenge")
            self.name_label.setText("Challenge Base Name:")
        elif run_type == "Single Run":
            self.main_panel.setCurrentWidget(self.single_run_panel)
            self.exp_name_input.setPlaceholderText("e.g., my_experiment_name")
            self.setWindowTitle("Launch New Experiment")
            self.launch_button.setText("Launch")
        elif run_type == "Hyperparameter Search":
            self.main_panel.setCurrentWidget(self.search_panel)
            self.exp_name_input.setPlaceholderText("e.g., my_search_name")
            self.setWindowTitle("Launch New Hyperparameter Search")
            self.launch_button.setText("Launch Search")

    def _select_challenger(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Challenger Config", CONFIGS_DIR, "JSON files (*.json)"
        )
        if path:
            self.challenger_config_path = path
            self.challenger_label.setText(os.path.basename(path))

    def _get_available_baselines(self):
        """Scans the base models directory for available baseline JSON files."""
        baselines = []
        if not os.path.isdir(BASE_MODELS_DIR):
            return baselines
        for filename in os.listdir(BASE_MODELS_DIR):
            if filename.endswith(".json"):
                baselines.append(os.path.splitext(filename)[0])
        baselines.sort()
        return baselines

    def _on_accept(self):
        """Validates the inputs based on the run type and accepts the dialog."""
        run_type = self.run_type_selector.currentText()
        name = self.exp_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Name cannot be empty.")
            return

        notes = self.notes_editor.toPlainText().strip()

        if run_type == "Challenge":
            if not self.challenger_config_path:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "You must select a challenger model config.",
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
                "notes": notes,
            }
        elif run_type == "Single Run":
            config = self.single_run_editor.get_config()
            config["experiment_name"] = name
            if notes:
                config["notes"] = notes
            self.launch_info = {"config": config, "name": name, "type": run_type}

        elif run_type == "Hyperparameter Search":
            config = self.search_run_editor.get_config()
            search_config = self.search_config_editor.get_config()

            config["experiment_name"] = name
            config["search"] = search_config
            if notes:
                config["notes"] = notes

            self.launch_info = {"config": config, "name": name, "type": run_type}

        self.accept()

    def get_launch_info(self):
        """Returns the information needed to launch the experiment."""
        return self.launch_info
