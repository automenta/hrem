import os
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QListWidget,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from .constants import CONFIGS_DIR, BASE_MODELS_DIR


class ChallengeLauncherDialog(QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Launch New Baseline Challenge")
        self.setMinimumWidth(500)

        self.challenger_config_path = None
        self.baseline_model_names = []
        self.launch_info = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Challenge Name
        layout.addWidget(QLabel("Challenge Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., my_model_vs_baselines")
        layout.addWidget(self.name_input)

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

        # Dialog Buttons
        button_box = QHBoxLayout()
        self.launch_button = QPushButton("Launch Challenge")
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
        if path:
            self.challenger_config_path = path
            self.challenger_label.setText(os.path.basename(path))

    def _add_baseline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Baseline Model", BASE_MODELS_DIR, "JSON files (*.json)"
        )
        if path:
            model_name = os.path.splitext(os.path.basename(path))[0]
            # Check if the item already exists
            if not self.baseline_list.findItems(model_name, Qt.MatchFlag.MatchExactly):
                self.baseline_list.addItem(model_name)

    def _remove_baseline(self):
        for item in self.baseline_list.selectedItems():
            self.baseline_list.takeItem(self.baseline_list.row(item))

    def _on_accept(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Challenge name cannot be empty.")
            return
        if not self.challenger_config_path:
            QMessageBox.warning(self, "Validation Error", "You must select a challenger model config.")
            return
        if self.baseline_list.count() == 0:
            QMessageBox.warning(self, "Validation Error", "You must select at least one baseline model.")
            return

        baselines = [self.baseline_list.item(i).text() for i in range(self.baseline_list.count())]

        self.launch_info = {
            "base_name": name,
            "challenger_config": self.challenger_config_path,
            "baselines": baselines,
        }
        self.accept()

    def get_launch_info(self):
        return self.launch_info
