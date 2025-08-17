import inspect
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QFormLayout,
    QGroupBox,
    QCheckBox,
    QDialogButtonBox,
)
from PyQt6.QtCore import pyqtSignal

from src.factories import MODEL_REGISTRY, DATASET_REGISTRY


class ConfigEditor(QDialog):
    """
    A dialog for viewing and editing experiment configurations.
    """
    def __init__(self, config: dict, parent=None, is_read_only=False):
        super().__init__(parent)
        self.is_read_only = is_read_only
        self.setWindowTitle("Configuration Editor")
        self.setMinimumWidth(500)

        self.visible_sections = ["model", "dataset", "training"]
        self.config = config

        self._init_ui()
        self._populate_dropdowns()
        self.set_config(self.config)

        if self.is_read_only:
            self._set_read_only()

        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        # No margins needed as it's the main layout for a dialog
        # main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Model Selection ---
        self.model_group = QGroupBox("Model")
        model_group_layout = QVBoxLayout(self.model_group)
        model_form_layout = QFormLayout()
        self.model_combo = QComboBox()
        model_form_layout.addRow(QLabel("Name:"), self.model_combo)
        model_group_layout.addLayout(model_form_layout)
        self.model_params_layout = QFormLayout()
        model_group_layout.addLayout(self.model_params_layout)
        main_layout.addWidget(self.model_group)

        # --- Dataset Selection ---
        self.dataset_group = QGroupBox("Dataset")
        dataset_group_layout = QVBoxLayout(self.dataset_group)
        dataset_form_layout = QFormLayout()
        self.dataset_combo = QComboBox()
        dataset_form_layout.addRow(QLabel("Name:"), self.dataset_combo)
        dataset_group_layout.addLayout(dataset_form_layout)
        self.dataset_params_layout = QFormLayout()
        dataset_group_layout.addLayout(self.dataset_params_layout)
        main_layout.addWidget(self.dataset_group)

        # --- Training Parameters ---
        self.training_group = QGroupBox("Training")
        self.training_params_layout = QFormLayout(self.training_group)
        self.training_params_layout.addRow("Epochs:", QLineEdit("20"))
        self.training_params_layout.addRow("Batch Size:", QLineEdit("32"))
        self.training_params_layout.addRow("Learning Rate:", QLineEdit("0.001"))
        main_layout.addWidget(self.training_group)

        main_layout.addStretch()

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self.model_group.setVisible("model" in self.visible_sections)
        self.dataset_group.setVisible("dataset" in self.visible_sections)
        self.training_group.setVisible("training" in self.visible_sections)

    def _populate_dropdowns(self):
        self.model_combo.addItems(sorted(MODEL_REGISTRY.keys()))
        self.dataset_combo.addItems(sorted(DATASET_REGISTRY.keys()))

    def _on_model_changed(self, model_name):
        model_class = MODEL_REGISTRY.get(model_name)
        self._update_params_layout(self.model_params_layout, model_class)

    def _on_dataset_changed(self, dataset_name):
        dataset_class = DATASET_REGISTRY.get(dataset_name)
        self._update_params_layout(self.dataset_params_layout, dataset_class)

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _update_params_layout(self, layout, component_class):
        self._clear_layout(layout)
        if not component_class:
            return

        sig = inspect.signature(component_class.__init__)
        params = sig.parameters

        for name, param in params.items():
            if name in ["self", "args", "kwargs", "config_dict"]:
                continue

            widget = None
            default_value = (
                param.default if param.default is not inspect.Parameter.empty else ""
            )

            # Try to infer widget type from default value type or annotation
            param_type = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else type(default_value)
            )

            if param_type is bool:
                widget = QCheckBox()
                widget.setChecked(bool(default_value))
            else:  # Default to QLineEdit
                widget = QLineEdit(str(default_value))

            if widget:
                # Use docstring for tooltip
                doc = inspect.getdoc(component_class.__init__)
                if doc:
                    # Simple parsing to find param docstring (this is brittle)
                    for line in doc.split("\n"):
                        if line.strip().startswith(f":param {name}"):
                            widget.setToolTip(line.split(":", 2)[-1].strip())
                            break

                layout.addRow(f"{name}:", widget)

    def get_config(self) -> dict:
        config = {
            "model": {"name": self.model_combo.currentText(), "params": {}},
            "dataset": {"name": self.dataset_combo.currentText(), "params": {}},
            "training": {},
        }

        self._get_params_from_layout(
            self.model_params_layout, config["model"]["params"]
        )
        self._get_params_from_layout(
            self.dataset_params_layout, config["dataset"]["params"]
        )
        self._get_params_from_layout(self.training_params_layout, config["training"])

        return config

    def _get_params_from_layout(self, layout, config_dict):
        for i in range(0, layout.count(), 2):
            label_item = layout.itemAt(i)
            field_item = layout.itemAt(i + 1)
            if (
                label_item
                and label_item.widget()
                and field_item
                and field_item.widget()
            ):
                label_widget = label_item.widget()
                field_widget = field_item.widget()
                key = label_widget.text().replace(":", "").lower().replace(" ", "_")
                config_dict[key] = self._get_widget_value(field_widget)

    def _get_widget_value(self, widget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QLineEdit):
            text = widget.text()
            # Try to convert to number if possible
            try:
                return int(text)
            except ValueError:
                try:
                    return float(text)
                except ValueError:
                    # Handle lists/tuples
                    if text.startswith("(") and text.endswith(")"):
                        return tuple(map(int, text[1:-1].split(",")))
                    if text.startswith("[") and text.endswith("]"):
                        return list(map(int, text[1:-1].split(",")))
                    return text  # It's just a string
        return None

    def set_config(self, config: dict):
        # Set model
        model_config = config.get("model", {})
        if model_config.get("name"):
            self.model_combo.setCurrentText(model_config["name"])
            self._update_params_layout(
                self.model_params_layout, MODEL_REGISTRY.get(model_config["name"])
            )
            self._set_layout_values(
                self.model_params_layout, model_config.get("params", {})
            )

        # Set dataset
        dataset_config = config.get("dataset", {})
        if dataset_config.get("name"):
            self.dataset_combo.setCurrentText(dataset_config["name"])
            self._update_params_layout(
                self.dataset_params_layout, DATASET_REGISTRY.get(dataset_config["name"])
            )
            self._set_layout_values(
                self.dataset_params_layout, dataset_config.get("params", {})
            )

        # Set training params
        self._set_layout_values(self.training_params_layout, config.get("training", {}))

    def _set_layout_values(self, layout, params: dict):
        for i in range(0, layout.count(), 2):
            label_item = layout.itemAt(i)
            field_item = layout.itemAt(i + 1)
            if (
                label_item
                and label_item.widget()
                and field_item
                and field_item.widget()
            ):
                label_widget = label_item.widget()
                field_widget = field_item.widget()
                key = label_widget.text().replace(":", "").lower().replace(" ", "_")
                if key in params:
                    self._set_widget_value(field_widget, params[key])

    def _set_widget_value(self, widget, value):
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))

    def _set_read_only(self):
        """Disables all input widgets in the dialog."""
        self.model_combo.setEnabled(False)
        self.dataset_combo.setEnabled(False)

        # Disable all child QLineEdit and QCheckBox widgets
        for group in [self.model_group, self.dataset_group, self.training_group]:
            for widget in group.findChildren((QLineEdit, QCheckBox)):
                widget.setEnabled(False)

        # Change buttons to just a "Close" button
        self.button_box.clear()
        self.button_box.addButton(QDialogButtonBox.StandardButton.Close)
