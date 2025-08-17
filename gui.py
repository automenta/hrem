import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from src.gui.race_launcher import RaceLauncher
from src.gui.race_monitor import RaceMonitor
from src.gui.experiment_manager import ExperimentManager

class RaceGUIApplication(QApplication):
    """
    The main application class for the Race GUI.
    It manages the windows and the experiment manager.
    """
    def __init__(self, argv):
        super().__init__(argv)
        self.manager = ExperimentManager()
        self.launcher = RaceLauncher()
        self.launcher.launch_info_ready.connect(self.start_race)
        self.monitor = None
        self.launcher.show()

    def start_race(self, launch_info):
        success, result = self.manager.launch_experiment_race(launch_info)
        if success:
            self.launcher.hide()
            # `result` is a list of baseline failure messages
            self.monitor = RaceMonitor(launch_info, self.manager)
            self.monitor.show()
            if result:
                failed_baselines_str = "\n".join(result)
                QMessageBox.warning(
                    self.monitor,
                    "Baseline Launch Failures",
                    "The race has started, but some baselines failed to launch:\n\n"
                    f"{failed_baselines_str}",
                )
        else:
            # `result` is an error message string
            QMessageBox.critical(
                self.launcher,
                "Race Launch Failed",
                f"Could not launch the race.\n\nReason: {result}",
            )

def main():
    """
    The main entry point for the Race GUI application.
    """
    app = RaceGUIApplication(sys.argv)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
