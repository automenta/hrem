from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QLabel, QLineEdit, QFormLayout,
    QTextEdit, QGroupBox, QListWidget, QListWidgetItem, QComboBox,
    QPushButton, QHBoxLayout, QFileDialog, QInputDialog, QMessageBox
)
from PyQt6.QtCore import Qt
import os
import json

from .config_editor import ConfigEditor
from .constants import CONFIGS_DIR, BASE_MODELS_DIR, BASE_DATASETS_DIR, TRAINING_PROFILES_DIR

class RaceWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Race Creation Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(700, 600)

        self.challenger_config_data = None
        self.launch_info = None

        self.addPage(IntroPage())
        self.addPage(RaceDetailsPage())
        self.addPage(ChallengerPage())
        self.addPage(BaselinesPage())
        self.addPage(TrainingPage())
        self.addPage(SummaryPage())

    def accept(self):
        # This method is called when the user clicks "Finish"
        race_name = self.field("raceName")
        notes = self.page(1).notes_editor.toPlainText().strip()

        challenger_config = self.challenger_config_data

        baselines_page = self.page(3)
        selected_baselines = []
        for i in range(baselines_page.baseline_list.count()):
            item = baselines_page.baseline_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_baselines.append(item.text())

        dataset = self.field("datasetName")

        training_overrides = {}
        try:
            epochs = self.page(4).epochs_input.text().strip()
            if epochs: training_overrides["epochs"] = int(epochs)
            batch_size = self.page(4).batch_size_input.text().strip()
            if batch_size: training_overrides["batch_size"] = int(batch_size)
            lr = self.page(4).lr_input.text().strip()
            if lr: training_overrides["learning_rate"] = float(lr)
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", f"Invalid number in training overrides: {e}")
            # Don't close the wizard
            return


        self.launch_info = {
            "type": "Challenge",
            "base_name": race_name,
            "challenger_config": challenger_config,
            "standard_baselines": selected_baselines,
            "dynamic_baselines": [], # Simplified: not using dynamic baselines from the wizard
            "dataset": dataset,
        }
        if training_overrides:
            self.launch_info["training_overrides"] = training_overrides
        if notes:
            self.launch_info["notes"] = notes

        super().accept()

    def get_launch_info(self):
        return self.launch_info

class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome to the Race Creation Wizard")
        self.setSubTitle("This wizard will guide you through setting up a new experiment race.")

        layout = QVBoxLayout(self)
        label = QLabel(
            "A 'race' allows you to compare a 'challenger' model configuration against one or more 'baseline' models on a specific dataset.\n\n"
            "This wizard will help you:\n"
            "1. Name and describe your race.\n"
            "2. Define your challenger model.\n"
            "3. Select baseline models to compete against.\n"
            "4. Configure training settings.\n"
            "5. Review and launch the race.\n\n"
            "Click 'Next' to begin."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.setLayout(layout)

class RaceDetailsPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Race Details")
        self.setSubTitle("Provide a name and optional notes for your race.")

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., my_transformer_vs_lstm")
        self.name_input.setToolTip("Enter a unique name for the race (e.g., 'transformer_v3_vs_lstm').\nAllowed characters: letters, numbers, underscore, hyphen.")
        form_layout.addRow("Race Name:", self.name_input)

        self.notes_editor = QTextEdit()
        self.notes_editor.setPlaceholderText("Describe the goal of this race, what you're testing, etc.")
        self.notes_editor.setToolTip("Add any notes about this race.\nThis is for your own reference and will be saved with the race results.")
        form_layout.addRow("Notes:", self.notes_editor)

        layout.addLayout(form_layout)
        self.setLayout(layout)

        self.registerField("raceName*", self.name_input)

class ChallengerPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Challenger Model")
        self.setSubTitle("Select, create, or edit the configuration for your main 'challenger' model.")

        layout = QVBoxLayout(self)
        self.challenger_label = QLabel("No challenger configuration loaded.")
        self.edit_button = QPushButton("View/Edit...")
        self.edit_button.setEnabled(False)
        self.edit_button.setToolTip("View or edit the currently loaded challenger configuration.")

        button_layout = QHBoxLayout()
        new_button = QPushButton("New from Template...")
        new_button.setToolTip("Create a new challenger config by editing a copy of a standard baseline template.")
        new_button.clicked.connect(self.create_new_challenger)
        select_button = QPushButton("Select from File...")
        select_button.setToolTip("Select an existing challenger model configuration from a .json file.")
        select_button.clicked.connect(self.select_challenger)
        self.edit_button.clicked.connect(self.edit_challenger)

        button_layout.addWidget(new_button)
        button_layout.addWidget(select_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addStretch()

        layout.addWidget(self.challenger_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # This will be used to store the path of the loaded config
        self.challenger_config_path = None

    def isComplete(self):
        return self.wizard().challenger_config_data is not None

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

                editor = ConfigEditor(config_data, self, is_read_only=False)
                editor.setWindowTitle(f"New Challenger (from {baseline_name})")
                if editor.exec():
                    self.wizard().challenger_config_data = editor.get_config()
                    self.challenger_config_path = None
                    self.challenger_label.setText("Unsaved Custom Challenger*")
                    self.edit_button.setEnabled(True)
                    self.completeChanged.emit()

            except (json.JSONDecodeError, IOError) as e:
                QMessageBox.critical(
                    self, "Error Reading Template", f"Could not read or parse the template file:\n{e}"
                )

    def select_challenger(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Challenger Config", CONFIGS_DIR, "JSON files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                self.wizard().challenger_config_data = json.load(f)

            self.challenger_config_path = path
            self.challenger_label.setText(os.path.basename(path))
            self.edit_button.setEnabled(True)
            self.completeChanged.emit()

        except (json.JSONDecodeError, IOError) as e:
            QMessageBox.critical(
                self, "Error Reading Config", f"Could not read or parse the config file:\n{e}"
            )
            self.wizard().challenger_config_data = None
            self.challenger_config_path = None
            self.challenger_label.setText("No file selected...")
            self.edit_button.setEnabled(False)
            self.completeChanged.emit()

    def edit_challenger(self):
        if not self.wizard().challenger_config_data:
            return

        editor = ConfigEditor(self.wizard().challenger_config_data, self, is_read_only=False)
        editor.setWindowTitle("Edit Challenger Configuration")
        if editor.exec():
            self.wizard().challenger_config_data = editor.get_config()
            if self.challenger_config_path:
                self.challenger_label.setText(f"{os.path.basename(self.challenger_config_path)}* (modified)")
            else:
                self.challenger_label.setText("Custom Config* (modified)")
            self.completeChanged.emit()

class BaselinesPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select Baselines")
        self.setSubTitle("Choose the baseline models to race your challenger against.")

        layout = QVBoxLayout(self)
        self.baseline_list = QListWidget()
        self.baseline_list.setToolTip("Select one or more standard baseline models to race against your challenger.")

        available_baselines = self._get_available_config_names(BASE_MODELS_DIR)
        for name in available_baselines:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.baseline_list.addItem(item)

        layout.addWidget(self.baseline_list)
        self.setLayout(layout)

    def _get_available_config_names(self, directory):
        names = []
        if not os.path.isdir(directory):
            return names
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                names.append(os.path.splitext(filename)[0])
        names.sort()
        return names

class TrainingPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Training Configuration")
        self.setSubTitle("Select a dataset and optionally override training parameters for all participants in the race.")

        layout = QVBoxLayout(self)

        # Dataset selection
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout(dataset_group)
        self.dataset_selector = QComboBox()
        self.dataset_selector.setToolTip("Select the dataset/task for this race.")
        available_datasets = self._get_available_config_names(BASE_DATASETS_DIR)
        self.dataset_selector.addItems([""] + available_datasets) # Add empty option
        dataset_layout.addRow("Task / Dataset:", self.dataset_selector)
        layout.addWidget(dataset_group)

        # Training Overrides
        override_group = QGroupBox("Training Overrides (Optional)")
        override_layout = QFormLayout(override_group)
        self.epochs_input = QLineEdit()
        self.epochs_input.setToolTip("Override the number of training epochs for ALL participants in this race.\nLeave blank to use the value from each model's config file.")
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setToolTip("Override the batch size for ALL participants in this race.\nLeave blank to use the value from each model's config file.")
        self.lr_input = QLineEdit()
        self.lr_input.setToolTip("Override the learning rate for ALL participants in this race.\nLeave blank to use the value from each model's config file.")
        override_layout.addRow("Epochs:", self.epochs_input)
        override_layout.addRow("Batch Size:", self.batch_size_input)
        override_layout.addRow("Learning Rate:", self.lr_input)
        layout.addWidget(override_group)

        self.setLayout(layout)
        self.registerField("datasetName*", self.dataset_selector, "currentText", self.dataset_selector.currentIndexChanged)

    def _get_available_config_names(self, directory):
        names = []
        if not os.path.isdir(directory):
            return names
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                names.append(os.path.splitext(filename)[0])
        names.sort()
        return names

class SummaryPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Summary & Launch")
        self.setSubTitle("Review the race details before launching.")

        layout = QVBoxLayout(self)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        self.setLayout(layout)

    def initializePage(self):
        # This is called every time the page is shown
        # We'll build the summary text here.
        race_name = self.field("raceName")
        notes = self.wizard().page(1).notes_editor.toPlainText().strip()

        challenger_label = self.wizard().page(2).challenger_label.text()

        baselines_page = self.wizard().page(3)
        selected_baselines = []
        for i in range(baselines_page.baseline_list.count()):
            item = baselines_page.baseline_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_baselines.append(item.text())

        dataset = self.field("datasetName")
        epochs = self.wizard().page(4).epochs_input.text()
        batch_size = self.wizard().page(4).batch_size_input.text()
        lr = self.wizard().page(4).lr_input.text()

        summary = (
            f"<b>Race Name:</b> {race_name}<br><br>"
            f"<b>Challenger:</b> {challenger_label}<br><br>"
            f"<b>Baselines:</b><br>- " + "<br>- ".join(selected_baselines or ["None selected"]) + "<br><br>"
            f"<b>Dataset:</b> {dataset}<br><br>"
            f"<b>Training Overrides:</b><br>"
            f"- Epochs: {epochs or 'Default'}<br>"
            f"- Batch Size: {batch_size or 'Default'}<br>"
            f"- Learning Rate: {lr or 'Default'}<br><br>"
            f"<b>Notes:</b><br>{notes or 'None'}"
        )
        self.summary_text.setHtml(summary)
