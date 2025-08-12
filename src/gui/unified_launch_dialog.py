import json
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
)

class UnifiedLaunchDialog(QDialog):
    """
    A dialog for launching a new experiment or hyperparameter search.
    Allows editing the configuration as a raw JSON text.
    """
    def __init__(self, config=None, exp_name=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New Experiment")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.launch_info = None

        self._init_ui()

        if config:
            self.config_editor.setText(json.dumps(config, indent=4))
        if exp_name:
            self.exp_name_input.setText(exp_name)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Experiment Name ---
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Experiment Name:"))
        self.exp_name_input = QLineEdit()
        name_layout.addWidget(self.exp_name_input)
        layout.addLayout(name_layout)

        # --- Run Type ---
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Run Type:"))
        self.run_type_selector = QComboBox()
        self.run_type_selector.addItems(["Single Run", "Hyperparameter Search"])
        type_layout.addWidget(self.run_type_selector)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # --- Config Editor ---
        layout.addWidget(QLabel("Configuration (JSON):"))
        self.config_editor = QTextEdit()
        self.config_editor.setFontFamily("monospace")
        layout.addWidget(self.config_editor)

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

    def _on_accept(self):
        """
        Validates the config and name before accepting the dialog.
        """
        exp_name = self.exp_name_input.text().strip()
        if not exp_name:
            QMessageBox.warning(self, "Validation Error", "Experiment name cannot be empty.")
            return

        try:
            config_text = self.config_editor.toPlainText()
            config = json.loads(config_text)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Validation Error", f"Invalid JSON in configuration: {e}")
            return

        # Add experiment name to config, as this is expected by the runner
        config['experiment_name'] = exp_name

        self.launch_info = {
            "config": config,
            "name": exp_name,
            "type": self.run_type_selector.currentText(),
        }
        self.accept()

    def get_launch_info(self):
        """
        Returns the information needed to launch the experiment.
        """
        return self.launch_info
