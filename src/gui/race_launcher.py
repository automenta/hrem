from PyQt6.QtWidgets import QDialog
from .race_launcher_dialog import RaceLauncherDialog

class RaceLauncher(QDialog):
    """
    A dialog for launching a new race (challenge).
    This class now acts as a wrapper around the RaceLauncherDialog.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # This dialog is a proxy for the actual launcher dialog.
        # It doesn't need its own UI.
        self.launch_info = None

        dialog = RaceLauncherDialog(self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.launch_info = dialog.get_launch_info()
            self.accept()
        else:
            self.reject()
