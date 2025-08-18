from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTextEdit,
    QPushButton,
    QHBoxLayout,
    QCheckBox,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtCore import QTimer, Qt
from .experiment_manager import ExperimentManager
from .constants import STATUS_RUNNING

class LogViewer(QDialog):
    """
    A dialog for viewing the live logs of an experiment.
    """
    def __init__(self, manager: ExperimentManager, exp_name: str, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.exp_name = exp_name
        self.is_running = False

        self.setWindowTitle(f"Logs for '{self.exp_name}'")
        self.setGeometry(200, 200, 800, 600)

        # --- UI ---
        self._init_ui()
        self._init_timer()

        # --- Initial Load ---
        self.refresh_logs()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        font = self.log_display.font()
        font.setFamily("Courier")
        font.setPointSize(10)
        self.log_display.setFont(font)
        layout.addWidget(self.log_display)

        button_layout = QHBoxLayout()
        self.auto_refresh_checkbox = QCheckBox("Auto-Refresh")
        self.auto_refresh_checkbox.stateChanged.connect(self._toggle_timer)
        button_layout.addWidget(self.auto_refresh_checkbox)

        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_logs)
        button_layout.addWidget(self.refresh_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_logs)
        # Timer is started/stopped by the checkbox

    def _toggle_timer(self, state):
        if state == Qt.CheckState.Checked.value:
            if self.is_running:
                self.timer.start(2000) # Refresh every 2 seconds
        else:
            self.timer.stop()

    def refresh_logs(self):
        """
        Fetches the latest logs from the manager and updates the display.
        """
        log_content = self.manager.get_log_contents(self.exp_name)
        statuses = self.manager.get_experiment_statuses()
        status = statuses.get(self.exp_name, "N/A")
        self.is_running = (status == STATUS_RUNNING)

        if not self.is_running:
            self.timer.stop()
            self.auto_refresh_checkbox.setChecked(False)
            self.auto_refresh_checkbox.setEnabled(False)
            self.auto_refresh_checkbox.setToolTip("Auto-refresh is disabled for finished experiments.")
        else:
            self.auto_refresh_checkbox.setEnabled(True)
            self.auto_refresh_checkbox.setToolTip("Enable to automatically refresh logs every 2 seconds.")


        current_scrollbar = self.log_display.verticalScrollBar()
        scroll_at_bottom = current_scrollbar.value() >= current_scrollbar.maximum()

        self.log_display.setPlainText(log_content)

        if scroll_at_bottom:
            current_scrollbar.setValue(current_scrollbar.maximum())

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
