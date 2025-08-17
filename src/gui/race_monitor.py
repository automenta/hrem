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
            style = "color: #008000;"  # Green
        else:
            style = "color: #000080;"  # Navy

        title = QLabel(f"<span style='{style}'>{title_text}</span>")
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

        self._update_summary()

    def _update_summary(self):
        """Analyzes the current race data and updates the summary label."""
        challenger_name = self.participant_names[0]
        baseline_names = self.participant_names[1:]

        # 1. Get challenger's latest performance
        challenger_results, _ = self.manager.load_experiment_results(challenger_name)
        if not challenger_results or "test_loss" not in challenger_results or not challenger_results["test_loss"]:
            self.summary_label.setText("<i>Waiting for challenger data...</i>")
            return
        challenger_loss = challenger_results["test_loss"][-1]

        # 2. Find the best baseline
        best_baseline_name = None
        best_baseline_loss = float('inf')

        for name in baseline_names:
            results, _ = self.manager.load_experiment_results(name)
            if results and "test_loss" in results and results["test_loss"]:
                current_loss = results["test_loss"][-1]
                if current_loss < best_baseline_loss:
                    best_baseline_loss = current_loss
                    best_baseline_name = name

        # 3. Compare and generate summary text
        summary_html = "<h3>Race Summary</h3>"
        if best_baseline_name is None:
            summary_html += f"Challenger is running. No baseline data available yet.<br>"
            summary_html += f"<b>{challenger_name}</b> Test Loss: {challenger_loss:.4f}"
        else:
            if challenger_loss < best_baseline_loss:
                diff = best_baseline_loss - challenger_loss
                summary_html += (f"<b>Winning:</b> Challenger (<b>{challenger_name}</b>) is leading.<br>"
                                 f"It is <b>{diff:.4f}</b> points ahead of the best baseline ({best_baseline_name}).")
            elif best_baseline_loss < challenger_loss:
                diff = challenger_loss - best_baseline_loss
                summary_html += (f"<b>Losing:</b> Challenger (<b>{challenger_name}</b>) is trailing.<br>"
                                 f"It is <b>{diff:.4f}</b> points behind the best baseline (<b>{best_baseline_name}</b>).")
            else:
                summary_html += "<b>Tied:</b> The challenger and best baseline are currently tied."

            summary_html += (f"<br><br><b>Challenger Loss:</b> {challenger_loss:.4f}<br>"
                             f"<b>Best Baseline Loss:</b> {best_baseline_loss:.4f}")

        self.summary_label.setText(summary_html)


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
