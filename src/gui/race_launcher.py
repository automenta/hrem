from PyQt6.QtWidgets import QDialog
from .race_wizard import RaceWizard

class RaceLauncher(QDialog):
    """
    A dialog for launching a new race (challenge).
    This class now acts as a wrapper around the RaceWizard.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # This dialog is now just a proxy for the wizard.
        # It doesn't need its own UI.
        self.launch_info = None

        wizard = RaceWizard(self)
        result = wizard.exec()

        if result == QDialog.DialogCode.Accepted:
            self.launch_info = wizard.get_launch_info()
            # The parent dialog's exec() will return Accepted
            self.accept()
        else:
            # The parent dialog's exec() will return Rejected
            self.reject()
