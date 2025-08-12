from PyQt6.QtCore import QObject, pyqtSignal
from watchdog.events import FileSystemEventHandler

class ResultsPathWatcher(QObject, FileSystemEventHandler):
    """
    A file system watcher that emits a signal when the results directory changes.
    """
    directory_changed = pyqtSignal()

    def on_any_event(self, event):
        # This is a broad catch-all. For more specific actions, one could use
        # on_created, on_deleted, on_modified, on_moved.
        # We ignore directory modified events as they are noisy.
        if event.is_directory and event.event_type == 'modified':
            return
        self.directory_changed.emit()
