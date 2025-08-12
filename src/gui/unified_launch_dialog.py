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
    QFileDialog,
)
from PyQt6.QtCore import Qt

# Assuming constants are in a reachable path.
# If not, these might need to be defined here or passed in.
# For now, let's define them locally for robustness.
from .constants import CONFIGS_DIR, BASE_MODELS_DIR


class UnifiedLaunchDialog(QDialog):
    """
    A dialog for launching a new experiment, challenge, or hyperparameter search.
    """
    def __init__(self, config=None, exp_name=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New...")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.launch_info = None
        self.challenger_config_path = None

        self._init_ui()
        self._on_run_type_changed(self.run_type_selector.currentText())

        if config:
            self.config_editor.setText(json.dumps(config, indent=4))
        if exp_name:
            self.exp_name_input.setText(exp_name)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Experiment Name ---
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.exp_name_input = QLineEdit()
        name_layout.addWidget(self.exp_name_input)
        layout.addLayout(name_layout)

        # --- Run Type ---
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Run Type:"))
        self.run_type_selector = QComboBox()
        self.run_type_selector.addItems(["Single Run", "Challenge", "Hyperparameter Search"])
        self.run_type_selector.currentTextChanged.connect(self._on_run_type_changed)
        type_layout.addWidget(self.run_type_selector)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # --- Main Panel ---
        self.main_panel = QStackedWidget()
        layout.addWidget(self.main_panel)

        self._create_standard_panel()
        self._create_challenge_panel()

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

    def _create_standard_panel(self):
        """Creates the panel for single runs and searches, which use a JSON editor."""
        self.standard_panel = QWidget()
        layout = QVBoxLayout(self.standard_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Configuration (JSON):"))
        self.config_editor = QTextEdit()
        self.config_editor.setFontFamily("monospace")
        layout.addWidget(self.config_editor)
        self.main_panel.addWidget(self.standard_panel)

    def _create_challenge_panel(self):
        """Creates the panel for launching a challenge."""
        self.challenge_panel = QWidget()
        layout = QVBoxLayout(self.challenge_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Challenger Selection
        layout.addWidget(QLabel("Challenger Model Config:"))
        challenger_layout = QHBoxLayout()
        self.challenger_label = QLabel("No file selected...")
        self.challenger_button = QPushButton("Select...")
        self.challenger_button.clicked.connect(self._select_challenger)
        challenger_layout.addWidget(self.challenger_label)
        challenger_layout.addStretch()
        challenger_layout.addWidget(self.challenger_button)
        layout.addLayout(challenger_layout)

        # Baselines Selection
        layout.addWidget(QLabel("Baseline Models:"))
        self.baseline_list = QListWidget()
        baseline_button_layout = QHBoxLayout()
        self.add_baseline_button = QPushButton("Add Baseline...")
        self.remove_baseline_button = QPushButton("Remove Selected")
        self.add_baseline_button.clicked.connect(self._add_baseline)
        self.remove_baseline_button.clicked.connect(self._remove_baseline)
        baseline_button_layout.addStretch()
        baseline_button_layout.addWidget(self.add_baseline_button)
        baseline_button_layout.addWidget(self.remove_baseline_button)
        layout.addWidget(self.baseline_list)
        layout.addLayout(baseline_button_layout)
        layout.addStretch()
        self.main_panel.addWidget(self.challenge_panel)

    def _on_run_type_changed(self, run_type):
        """Switches the visible panel based on the selected run type."""
        if run_type == "Challenge":
            self.main_panel.setCurrentWidget(self.challenge_panel)
            self.exp_name_input.setPlaceholderText("e.g., my_model_vs_baselines")
            self.setWindowTitle("Launch New Challenge")
            self.launch_button.setText("Launch Challenge")
        else:
            self.main_panel.setCurrentWidget(self.standard_panel)
            if run_type == "Single Run":
                self.exp_name_input.setPlaceholderText("e.g., my_experiment_name")
                self.setWindowTitle("Launch New Experiment")
                self.launch_button.setText("Launch")
            else: # Hyperparameter Search
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

    def _add_baseline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Baseline Model", BASE_MODELS_DIR, "JSON files (*.json)"
        )
        if path:
            model_name = os.path.splitext(os.path.basename(path))[0]
            if not self.baseline_list.findItems(model_name, Qt.MatchFlag.MatchExactly):
                self.baseline_list.addItem(model_name)

    def _remove_baseline(self):
        for item in self.baseline_list.selectedItems():
            self.baseline_list.takeItem(self.baseline_list.row(item))

    def _on_accept(self):
        """
        Validates the inputs based on the run type and accepts the dialog.
        """
        run_type = self.run_type_selector.currentText()
        name = self.exp_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Name cannot be empty.")
            return

        if run_type == "Challenge":
            if not self.challenger_config_path:
                QMessageBox.warning(self, "Validation Error", "You must select a challenger model config.")
                return
            if self.baseline_list.count() == 0:
                QMessageBox.warning(self, "Validation Error", "You must select at least one baseline model.")
                return

            baselines = [self.baseline_list.item(i).text() for i in range(self.baseline_list.count())]
            self.launch_info = {
                "type": "Challenge",
                "base_name": name,
                "challenger_config": self.challenger_config_path,
                "baselines": baselines,
            }
        else: # Single Run or Hyperparameter Search
            try:
                config_text = self.config_editor.toPlainText()
                if not config_text.strip():
                    QMessageBox.warning(self, "Validation Error", "Configuration cannot be empty.")
                    return
                config = json.loads(config_text)
            except json.JSONDecodeError as e:
                QMessageBox.warning(self, "Validation Error", f"Invalid JSON in configuration: {e}")
                return

            # Add experiment name to config, as this is expected by the runner
            config['experiment_name'] = name

            self.launch_info = {
                "config": config,
                "name": name,
                "type": run_type,
            }

        self.accept()

    def get_launch_info(self):
        """
        Returns the information needed to launch the experiment.
        """
        return self.launch_info
