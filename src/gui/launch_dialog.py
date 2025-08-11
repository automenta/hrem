from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
)
from PyQt6.QtCore import Qt

from .constants import CONFIGS_DIR

class LaunchExperimentDialog(QDialog):
    """
    A dialog for launching a new experiment, with options for paired baseline runs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Launch New Experiment Race")
        self.setMinimumWidth(500)

        self.challenger_config_path = ""

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Challenger Selection ---
        challenger_layout = QHBoxLayout()
        self.config_path_label = QLabel("Challenger Config: (None)")
        self.select_config_button = QPushButton("Select...")
        self.select_config_button.clicked.connect(self._select_challenger_config)
        challenger_layout.addWidget(self.config_path_label)
        challenger_layout.addWidget(self.select_config_button)
        layout.addLayout(challenger_layout)

        # --- Experiment Name ---
        layout.addWidget(QLabel("Base Name for Experiment Race:"))
        self.exp_name_input = QLineEdit()
        layout.addWidget(self.exp_name_input)

        # --- Baseline Selection ---
        layout.addWidget(QLabel("Select Baseline Models to Race Against:"))
        self.baseline_list_widget = QListWidget()
        # In a real implementation, this would be populated dynamically
        baselines = ["lstm", "transformer", "mlp"]
        for baseline in baselines:
            item = QListWidgetItem(baseline)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.baseline_list_widget.addItem(item)
        layout.addWidget(self.baseline_list_widget)

        # --- Dialog Buttons ---
        button_box = QHBoxLayout()
        self.launch_button = QPushButton("Launch")
        self.launch_button.clicked.connect(self.accept)
        self.launch_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(self.cancel_button)
        button_box.addWidget(self.launch_button)
        layout.addLayout(button_box)

    def _select_challenger_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Challenger Config", CONFIGS_DIR, "Config files (*.json *.conf)"
        )
        if path:
            self.challenger_config_path = path
            # Set a default experiment name based on the file
            base_name = path.split('/')[-1].replace('.json', '').replace('.conf', '')
            self.config_path_label.setText(f"Challenger Config: {base_name}")
            self.exp_name_input.setText(base_name)
            self.launch_button.setEnabled(True)

    def get_launch_info(self):
        """
        Returns the information needed to launch the experiment race.
        """
        checked_baselines = []
        for i in range(self.baseline_list_widget.count()):
            item = self.baseline_list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_baselines.append(item.text())

        return {
            "challenger_config": self.challenger_config_path,
            "base_name": self.exp_name_input.text(),
            "baselines": checked_baselines,
        }
