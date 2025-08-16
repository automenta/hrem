import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QMessageBox,
    QHBoxLayout,
)
from src.gui.challenge_view import ChallengeView
from src.gui.unified_launch_dialog import UnifiedLaunchDialog
from src.gui.experiment_manager import ExperimentManager
from src.gui.race_archive_dialog import RaceArchiveDialog
from src.gui.file_watcher import ResultsPathWatcher
from watchdog.observers import Observer


class RaceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HREM Race GUI")
        self.setGeometry(100, 100, 900, 700)
        self.manager = ExperimentManager()
        self._init_ui()
        self._init_watcher()
        self.challenge_view.refresh() # Initial load

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Top Button Panel ---
        button_panel = QWidget()
        button_layout = QHBoxLayout(button_panel)
        button_layout.setContentsMargins(0, 0, 0, 0) # Remove padding

        self.launch_button = QPushButton("Launch New Challenge...")
        self.launch_button.clicked.connect(self.launch_new_challenge)
        button_layout.addWidget(self.launch_button)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_view)
        button_layout.addWidget(self.refresh_button)

        button_layout.addStretch() # Pushes buttons to the left

        self.archive_button = QPushButton("Race History")
        self.archive_button.clicked.connect(self.open_race_archive)
        button_layout.addWidget(self.archive_button)

        layout.addWidget(button_panel)

        self.challenge_view = ChallengeView(self.manager)
        layout.addWidget(self.challenge_view)

    def _init_watcher(self):
        self.observer = Observer()
        self.watcher = ResultsPathWatcher()
        self.watcher.directory_changed.connect(self.challenge_view.refresh)
        # Ensure the results directory exists before observing
        os.makedirs(self.manager.RESULTS_DIR, exist_ok=True)
        self.observer.schedule(self.watcher, self.manager.RESULTS_DIR, recursive=True)
        self.observer.start()

    def refresh_view(self):
        """Manually trigger a refresh of the challenge view."""
        self.challenge_view.refresh()

    def open_race_archive(self):
        dialog = RaceArchiveDialog(self.manager, self)
        dialog.exec()

    def launch_new_challenge(self):
        dialog = UnifiedLaunchDialog(parent=self)
        # Force the dialog into 'Challenge' mode
        dialog.run_type_selector.setCurrentText("Challenge")
        dialog.run_type_selector.setEnabled(False) # Prevent user from changing it

        if dialog.exec():
            launch_info = dialog.get_launch_info()
            if not launch_info:
                return

            success, message = self.manager.launch_experiment_race(launch_info)
            if not success:
                QMessageBox.warning(self, "Launch Failed", message)
            # No timer needed, the file watcher will trigger the refresh automatically

    def closeEvent(self, event):
        self.observer.stop()
        self.observer.join()
        super().closeEvent(event)


def main():
    """
    The main entry point for the Race GUI application.
    """
    app = QApplication(sys.argv)
    gui = RaceGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
