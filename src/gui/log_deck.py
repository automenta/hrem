from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLabel,
    QPushButton,
    QSizePolicy,
)
from PyQt6.QtCore import Qt


class LogViewWidget(QWidget):
    """
    A widget that displays the logs for a single experiment.
    """

    def __init__(self, exp_name, log_content, parent=None):
        super().__init__(parent)
        self.exp_name = exp_name

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        title_layout = QHBoxLayout()
        title_label = QLabel(f"<b>{exp_name}</b>")
        self.close_button = QPushButton("x")
        self.close_button.setFixedSize(20, 20)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.close_button)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFontFamily("monospace")
        self.log_display.setText(log_content)
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

        main_layout.addLayout(title_layout)
        main_layout.addWidget(self.log_display)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class LogDeckWindow(QWidget):
    """
    A non-modal window for displaying multiple experiment logs side-by-side.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Deck")
        self.setWindowFlags(Qt.WindowType.Window)  # Make it a separate window
        self.setGeometry(150, 150, 1000, 600)

        self.active_logs = {}  # {exp_name: widget}

        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)

    def add_log_view(self, exp_name, log_content):
        """
        Adds a new log view to the deck, or updates an existing one.
        """
        if exp_name in self.active_logs:
            # If log view already exists, just bring it to front (or update content)
            # For now, we just ignore if it's already open.
            # A more advanced implementation could update it.
            return

        log_widget = LogViewWidget(exp_name, log_content, self)
        log_widget.close_button.clicked.connect(lambda: self.remove_log_view(exp_name))

        self.main_layout.addWidget(log_widget)
        self.active_logs[exp_name] = log_widget

    def remove_log_view(self, exp_name):
        """
        Removes a log view from the deck.
        """
        if exp_name in self.active_logs:
            widget = self.active_logs.pop(exp_name)
            widget.setParent(None)
            widget.deleteLater()

    def update_log_content(self, exp_name, log_content):
        """
        Updates the content of an existing log view.
        """
        if exp_name in self.active_logs:
            self.active_logs[exp_name].log_display.setText(log_content)
            # Auto-scroll to bottom
            scrollbar = self.active_logs[exp_name].log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def showEvent(self, event):
        """Ensure the window is raised and activated when shown."""
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
