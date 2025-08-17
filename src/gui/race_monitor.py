from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSplitter
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from .experiment_manager import ExperimentManager

class RaceMonitor(QMainWindow):
    """
    The main window for monitoring a race.
    """
    def __init__(self, race_info: dict, manager: ExperimentManager, parent=None):
        super().__init__(parent)
        self.race_info = race_info
        self.manager = manager
        self.setWindowTitle(f"Race Monitor: {self.race_info['base_name']}")
        self.setGeometry(150, 150, 1200, 800)
        self.plot_widgets = {}
        self.participant_names = []

        self._init_ui()
        self._start_monitoring()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Title
        title = QLabel(f"<h2>Race: {self.race_info['base_name']}</h2>")
        main_layout.addWidget(title)

        # Main splitter for plots
        self.plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.plot_splitter, 1) # Give it stretch factor

        # Create a plot for the challenger and each baseline
        challenger_name = f"{self.race_info['base_name']}_challenger"
        self.participant_names.append(challenger_name)
        self._create_participant_plot(challenger_name, is_challenger=True)

        for baseline_model in self.race_info['baselines']:
            baseline_name = f"{self.race_info['base_name']}_{baseline_model}"
            self.participant_names.append(baseline_name)
            self._create_participant_plot(baseline_name)

        # Stop button
        self.stop_button = QPushButton("Stop Race")
        self.stop_button.clicked.connect(self._stop_race)
        main_layout.addWidget(self.stop_button)

        # Placeholder for summary
        self.summary_label = QLabel("Summary of the race will be displayed here.")
        main_layout.addWidget(self.summary_label)

    def _create_participant_plot(self, participant_name: str, is_challenger: bool = False):
        container = QWidget()
        layout = QVBoxLayout(container)

        title_text = f"<b>{participant_name}</b>"
        if is_challenger:
            title_text += " (Challenger)"

        title = QLabel(title_text)
        layout.addWidget(title)

        plot_widget = pg.PlotWidget()
        plot_widget.addLegend()
        self.plot_widgets[participant_name] = plot_widget
        layout.addWidget(plot_widget)

        self.plot_splitter.addWidget(container)

    def _start_monitoring(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_race_status)
        self.timer.start(2000)  # Update every 2 seconds

    def _update_race_status(self):
        self.manager.update_log_files()
        for name in self.participant_names:
            results_data, _ = self.manager.load_experiment_results(name)
            if results_data:
                self.update_plot(name, results_data)

        # Here you could add logic to update the summary label
        # For now, we'll just keep it simple.

    def update_plot(self, experiment_name: str, results_data: dict):
        if experiment_name not in self.plot_widgets:
            return

        plot_widget = self.plot_widgets[experiment_name]
        plot_widget.clear()

        train_loss = results_data.get("train_loss", [])
        test_loss = results_data.get("test_loss", [])

        if train_loss:
            plot_widget.plot(train_loss, pen='b', name="Train Loss")
        if test_loss:
            plot_widget.plot(test_loss, pen='r', name="Test Loss")

    def _stop_race(self):
        for name in self.participant_names:
            self.manager.stop_experiment(name, force=True)
        self.timer.stop()
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Race Stopped")

    def closeEvent(self, event):
        self._stop_race()
        super().closeEvent(event)
