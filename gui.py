import sys
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog
from src.gui.dashboard import DashboardWindow
from src.gui.race_launcher import RaceLauncher
from src.gui.race_monitor import RaceMonitor
from src.gui.experiment_manager import ExperimentManager
from src.gui.workers import RaceLaunchWorker

class RaceGUIApplication(QApplication):
    """
    The main application class for the Race GUI.
    It manages the windows and the experiment manager.
    """
    def __init__(self, argv):
        super().__init__(argv)
        self.manager = ExperimentManager()

        self.dashboard = DashboardWindow(self.manager)
        self.dashboard.launch_new_race.connect(self._show_launcher)
        self.dashboard.monitor_race.connect(self._show_monitor)

        self.launcher = None
        self.monitors = {} # Track open monitors by race base_name
        self.worker_thread = None
        self.worker = None
        self.progress_dialog = None

        self.dashboard.show()

    def _show_launcher(self):
        # Make sure only one launcher is open at a time
        if self.launcher is None or not self.launcher.isVisible():
            self.launcher = RaceLauncher()
            self.launcher.launch_info_ready.connect(self._start_race_launch)
            self.launcher.show()
        self.launcher.activateWindow()

    def _start_race_launch(self, launch_info):
        if self.launcher:
            self.launcher.hide()

        self.progress_dialog = QProgressDialog(
            "Launching race...", "Cancel", 0, 0, self.dashboard
        )
        self.progress_dialog.setWindowTitle("Please Wait")
        self.progress_dialog.setModal(True)
        self.progress_dialog.show()

        self.worker_thread = QThread()
        self.worker = RaceLaunchWorker(self.manager, launch_info)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_launch_progress)
        self.worker.error.connect(self._on_launch_error)
        self.worker.success.connect(
            lambda failures: self._on_launch_success(launch_info, failures)
        )
        self.worker.success.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.worker.deleteLater)

        self.progress_dialog.canceled.connect(self.worker.cancel)
        self.progress_dialog.canceled.connect(self.worker_thread.quit)

        self.worker_thread.start()

    def _on_launch_progress(self, message):
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)

    def _on_launch_error(self, message):
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(
            self.dashboard, "Race Launch Failed", f"Could not launch the race.\n\nReason: {message}"
        )
        self.dashboard.populate_experiments_table() # Refresh dashboard view

    def _on_launch_success(self, launch_info, baseline_failures):
        if self.progress_dialog:
            self.progress_dialog.close()

        self.dashboard.populate_experiments_table() # Refresh dashboard view
        self._show_monitor(launch_info)

        if baseline_failures:
            # Since the monitor might not be the active window, show the warning on the dashboard
            failed_baselines_str = "\n".join(baseline_failures)
            QMessageBox.warning(
                self.dashboard,
                "Baseline Launch Failures",
                "The race has started, but some baselines failed to launch:\n\n"
                f"{failed_baselines_str}",
            )

    def _show_monitor(self, launch_info):
        base_name = launch_info["base_name"]
        if base_name in self.monitors and self.monitors[base_name].isVisible():
            self.monitors[base_name].activateWindow()
        else:
            monitor = RaceMonitor(launch_info, self.manager)
            self.monitors[base_name] = monitor
            monitor.show()

def main():
    """
    The main entry point for the Race GUI application.
    """
    app = RaceGUIApplication(sys.argv)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
