from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox


class InputDialog(QDialog):
    """
    A reusable dialog to get a single line of text input from the user.
    """
    def __init__(self, parent=None, title="Input", label="Value:", text=""):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QFormLayout(self)

        self.input_field = QLineEdit(text)
        layout.addRow(label, self.input_field)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.input_field.selectAll()

    def get_text(self):
        """
        Returns the text entered by the user if the dialog was accepted.
        Otherwise, returns None.
        """
        if self.exec():
            return self.input_field.text().strip()
        return None
