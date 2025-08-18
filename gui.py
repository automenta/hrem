import sys
from PyQt6.QtWidgets import QApplication
from src.gui.experiment_manager import ExperimentManager
from src.gui.experiment_dashboard import ExperimentDashboard

class RaceGUIApplication(QApplication):
    """
    The main application class for the Race GUI.
    It initializes the experiment manager and the main dashboard window.
    """
    def __init__(self, argv):
        super().__init__(argv)
        self.manager = ExperimentManager()
        self.dashboard = ExperimentDashboard(self.manager)
        self.dashboard.show()

def main():
    """
    The main entry point for the Race GUI application.
    """
    app = RaceGUIApplication(sys.argv)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
