from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QMessageBox, QDialog, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from .experiment_manager import ExperimentManager
from .config_editor import ConfigEditor
from .constants import RESULTS_DIR, STATUS_RUNNING
import os

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
        self.participant_widgets = {}  # Will store {name: {'container': QWidget, 'plot': pg.PlotWidget, 'error_label': QLabel}}
        self.participant_names = []
        self.plot_data_errors = {} # Will store {name: "error message"}

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

        # Summary Table
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(4)
        self.summary_table.setHorizontalHeaderLabels(["Participant", "Status", "Latest Test Loss", "Notes"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        main_layout.addWidget(self.summary_table)

        # Stop button
        self.stop_button = QPushButton("Stop Race")
        self.stop_button.clicked.connect(self._stop_race)
        main_layout.addWidget(self.stop_button)

    def _create_participant_plot(self, participant_name: str, is_challenger: bool = False):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        title_text = f"<b>{participant_name}</b>"
        if is_challenger:
            title_text += " (Challenger)"
            style = "color: #008000;"  # Green
        else:
            style = "color: #000080;"  # Navy

        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel(f"<span style='{style}'>{title_text}</span>"))
        title_layout.addStretch()

        view_config_button = QPushButton("View Config")
        view_config_button.setToolTip("View the configuration for this experiment")
        view_config_button.clicked.connect(lambda: self._view_config(participant_name))
        title_layout.addWidget(view_config_button)

        view_log_button = QPushButton("View Log")
        view_log_button.setToolTip("View the output log for this experiment")
        view_log_button.clicked.connect(lambda: self._view_log(participant_name))
        title_layout.addWidget(view_log_button)

        layout.addLayout(title_layout)

        plot_widget = pg.PlotWidget()
        plot_widget.addLegend()
        layout.addWidget(plot_widget)

        error_label = QLabel("An error occurred while loading data.")
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        error_label.setStyleSheet("color: red; background-color: #ffe0e0; border: 1px solid red;")
        error_label.setWordWrap(True)
        error_label.hide() # Initially hidden
        layout.addWidget(error_label)

        self.participant_widgets[participant_name] = {
            "container": container,
            "plot": plot_widget,
            "error_label": error_label
        }

        self.plot_splitter.addWidget(container)

    def _start_monitoring(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_race_status)
        self.timer.start(2000)  # Update every 2 seconds

    def _update_race_status(self):
        self.manager.update_log_files()
        statuses = self.manager.get_statuses()

        for name in self.participant_names:
            results_data, error_msg = self.manager.load_experiment_results(name)

            if error_msg:
                self.plot_data_errors[name] = error_msg
            else:
                self.plot_data_errors.pop(name, None) # Clear any previous error

            self.update_plot(name, results_data)

        self._update_summary(statuses)

        # Check if the race is finished
        is_race_running = any(statuses.get(name) == STATUS_RUNNING for name in self.participant_names)

        if not is_race_running:
            self.timer.stop()
            self.stop_button.setText("Close")
            try:
                self.stop_button.clicked.disconnect(self._stop_race)
            except TypeError:
                pass # Already disconnected
            try:
                self.stop_button.clicked.disconnect(self.close)
            except TypeError:
                pass
            self.stop_button.clicked.connect(self.close)
            self.stop_button.setEnabled(True)


    def _update_summary(self, statuses: dict):
        """Analyzes the current race data and updates the summary table."""
        self.summary_table.setRowCount(len(self.participant_names))

        all_losses = {}
        for i, name in enumerate(self.participant_names):
            # Get status
            status = statuses.get(name, "Unknown")

            # Get latest loss
            loss_str = "N/A"
            results, err = self.manager.load_experiment_results(name)
            if err:
                loss_str = "Error"
            elif results and "test_loss" in results and results["test_loss"]:
                latest_loss = results["test_loss"][-1]
                all_losses[name] = latest_loss
                loss_str = f"{latest_loss:.4f}"

            # Determine notes (leader, etc.)
            notes = ""
            if i == 0: # Challenger
                notes = "Challenger"

            self.summary_table.setItem(i, 0, QTableWidgetItem(name))
            self.summary_table.setItem(i, 1, QTableWidgetItem(status))
            self.summary_table.setItem(i, 2, QTableWidgetItem(loss_str))
            self.summary_table.setItem(i, 3, QTableWidgetItem(notes))

        # Add "Winning" / "Losing" note
        if self.participant_names[0] in all_losses and len(all_losses) > 1:
            challenger_loss = all_losses[self.participant_names[0]]
            best_baseline_loss = min(v for k, v in all_losses.items() if k != self.participant_names[0])

            challenger_row = 0 # Challenger is always first
            notes_item = self.summary_table.item(challenger_row, 3)
            current_notes = notes_item.text()

            if challenger_loss < best_baseline_loss:
                notes_item.setText(f"{current_notes} (Winning)")
            elif challenger_loss > best_baseline_loss:
                notes_item.setText(f"{current_notes} (Losing)")
            else:
                notes_item.setText(f"{current_notes} (Tied)")


    def update_plot(self, experiment_name: str, results_data: dict | None):
        if experiment_name not in self.participant_widgets:
            return

        widgets = self.participant_widgets[experiment_name]
        plot_widget = widgets["plot"]
        error_label = widgets["error_label"]

        error_msg = self.plot_data_errors.get(experiment_name)

        if error_msg:
            plot_widget.hide()
            error_label.setText(f"Data Error:\n{error_msg}")
            error_label.show()
            return

        # If we reach here, there is no error for this plot
        error_label.hide()
        plot_widget.show()
        plot_widget.clear()

        if not results_data:
            return # No data to plot yet

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
        # Only stop processes if they are still running
        if self.timer.isActive():
            self._stop_race()
        super().closeEvent(event)

    def _view_config(self, experiment_name: str):
        config, err = self.manager.load_experiment_config(experiment_name)
        if err:
            QMessageBox.warning(self, "Error", f"Could not load config for {experiment_name}:\n{err}")
            return

        editor = ConfigEditor(config, self, is_read_only=True)
        editor.setWindowTitle(f"Config: {experiment_name}")
        editor.exec()

    def _view_log(self, experiment_name: str):
        log_content = self.manager.get_log_contents(experiment_name)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Log: {experiment_name}")
        dialog.setGeometry(250, 250, 800, 600)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(log_content)
        layout.addWidget(text_edit)

        dialog.exec()
