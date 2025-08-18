from PyQt6.QtCore import QObject, pyqtSignal

class RaceLaunchWorker(QObject):
    """
    A worker for running the race launch process in a separate thread.
    """
    # Signal to update the progress dialog
    progress = pyqtSignal(str)
    # Signal for critical errors during launch
    error = pyqtSignal(str)
    # Signal for successful launch, carrying the list of baseline failures
    success = pyqtSignal(list)

    def __init__(self, manager, launch_info):
        super().__init__()
        self.manager = manager
        self.launch_info = launch_info
        self.is_cancelled = False

    def run(self):
        """
        Executes the race launch and emits signals.
        """
        try:
            for status_type, data in self.manager.launch_experiment_race(self.launch_info):
                if self.is_cancelled:
                    # TODO: Add logic to stop launching and clean up
                    break

                if status_type == 'progress':
                    self.progress.emit(data)
                elif status_type == 'error':
                    self.error.emit(data)
                    return # Stop on critical error
                elif status_type == 'success':
                    self.success.emit(data)

        except Exception as e:
            self.error.emit(f"An unexpected error occurred in the launch thread: {e}")

    def cancel(self):
        self.is_cancelled = True
