from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QMessageBox, QDialog, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QListWidget, QListWidgetItem, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

import json
import csv
from datetime import datetime
from PyQt6.QtWidgets import QInputDialog, QFileDialog
from .experiment_manager import ExperimentManager
from .config_editor import ConfigEditor
from .constants import RESULTS_DIR, STATUS_RUNNING, RACES_DIR
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
        self.available_metrics = []
        self.selected_metrics = ["test_loss"] # Default metric

        self._init_participants()
        self._init_ui()
        self._start_monitoring()

    def _init_participants(self):
        """Determine participant names and available metrics."""
        challenger_name = f"{self.race_info['base_name']}_challenger"
        self.participant_names.append(challenger_name)

        for baseline_model in self.race_info['baselines']:
            baseline_name = f"{self.race_info['base_name']}_baseline_{baseline_model}"
            self.participant_names.append(baseline_name)

        self.available_metrics = self.manager.get_available_metrics_for_race(self.participant_names)
        if not self.available_metrics:
            self.available_metrics = ["test_loss", "train_loss"] # Default fallback

        if not self.selected_metrics:
            if "test_loss" in self.available_metrics:
                self.selected_metrics = ["test_loss"]
            elif self.available_metrics:
                self.selected_metrics = [self.available_metrics[0]]


    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Top Bar: Title and Metric Selector ---
        top_bar_layout = QHBoxLayout()
        title = QLabel(f"<h2>Race: {self.race_info['base_name']}</h2>")
        top_bar_layout.addWidget(title)
        top_bar_layout.addStretch()

        top_bar_layout.addWidget(QLabel("Plot Metrics:"))
        self.select_metrics_button = QPushButton("Select Metrics...")
        self.select_metrics_button.clicked.connect(self._open_metric_selector)
        top_bar_layout.addWidget(self.select_metrics_button)
        main_layout.addLayout(top_bar_layout)


        # Main splitter for plots
        self.plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.plot_splitter, 1) # Give it stretch factor

        # Create a plot for the challenger and each baseline
        self._create_participant_plot(self.participant_names[0], is_challenger=True)
        for participant_name in self.participant_names[1:]:
            self._create_participant_plot(participant_name)

        # Hyperparameter Table
        self.hparam_table = QTableWidget()
        self.hparam_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.hparam_table.verticalHeader().setVisible(False)
        main_layout.addWidget(self.hparam_table)

        # Summary Table
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(4)
        self.summary_table.setHorizontalHeaderLabels(["Participant", "Status", "Latest Test Loss", "Notes"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        main_layout.addWidget(self.summary_table)

        # --- Analysis Toolbar (initially hidden) ---
        self.analysis_toolbar = QWidget()
        analysis_layout = QHBoxLayout(self.analysis_toolbar)
        analysis_layout.setContentsMargins(0, 0, 0, 0)

        self.add_conclusion_button = QPushButton("Add Conclusion")
        self.add_conclusion_button.clicked.connect(self._add_conclusion)
        self.save_plot_button = QPushButton("Save Summary Plot")
        self.save_plot_button.clicked.connect(self._save_summary_plot)
        self.export_summary_button = QPushButton("Export Summary CSV")
        self.export_summary_button.clicked.connect(self._export_summary_data)

        analysis_layout.addWidget(self.add_conclusion_button)
        analysis_layout.addWidget(self.save_plot_button)
        analysis_layout.addWidget(self.export_summary_button)
        analysis_layout.addStretch()
        main_layout.addWidget(self.analysis_toolbar)
        self.analysis_toolbar.hide()


        # Stop button / Close button
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

    def _open_metric_selector(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Metrics to Plot")
        layout = QVBoxLayout(dialog)

        list_widget = QListWidget()
        for metric in self.available_metrics:
            item = QListWidgetItem(metric)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            check_state = Qt.CheckState.Checked if metric in self.selected_metrics else Qt.CheckState.Unchecked
            item.setCheckState(check_state)
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec():
            selected = []
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected.append(item.text())
            self._on_metrics_selected(selected)

    def _on_metrics_selected(self, new_metrics: list[str]):
        """Handle the user selecting new metrics to plot."""
        if not new_metrics: # Don't allow unselecting everything
            QMessageBox.warning(self, "Selection Error", "Please select at least one metric to plot.")
            return
        self.selected_metrics = new_metrics
        # Trigger a full update to redraw plots with the new metric
        self._update_race_status()

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

        self._update_hyperparameter_table()
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
            self.stop_button.clicked.connect(self.close)
            self.stop_button.setEnabled(True)
            self.analysis_toolbar.show()


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
        plot_widget.setLabel('bottom', 'Epoch')
        plot_widget.setLabel('left', 'Metric Value')
        plot_widget.setTitle(f"Metrics vs. Epoch")
        plot_widget.addLegend()

        if not results_data:
            return # No data to plot yet

        # Define a list of colors to cycle through for plotting
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']

        for i, metric_name in enumerate(self.selected_metrics):
            metric_data = results_data.get(metric_name, [])
            if metric_data:
                pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                plot_widget.plot(metric_data, pen=pen, name=metric_name)

    def _update_hyperparameter_table(self):
        """Compares participant configs and displays the differing hyperparameters."""
        all_configs = {}
        for name in self.participant_names:
            config, err = self.manager.load_experiment_config(name)
            if config and not err:
                all_configs[name] = config

        if len(all_configs) < 2:
            self.hparam_table.hide()
            return # No need to show table for one participant

        diff_params = self._get_differentiating_hparams(all_configs)

        if not diff_params:
            self.hparam_table.hide()
            return

        self.hparam_table.show()
        self.hparam_table.setRowCount(len(diff_params))
        self.hparam_table.setColumnCount(len(self.participant_names))

        self.hparam_table.setVerticalHeaderLabels(diff_params)
        self.hparam_table.setHorizontalHeaderLabels(self.participant_names)

        flat_configs = {name: self._flatten_dict(cfg) for name, cfg in all_configs.items()}

        for row, param_key in enumerate(diff_params):
            for col, p_name in enumerate(self.participant_names):
                value = flat_configs.get(p_name, {}).get(param_key, "N/A")
                self.hparam_table.setItem(row, col, QTableWidgetItem(str(value)))

        self.hparam_table.resizeColumnsToContents()
        self.hparam_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Exclude non-essential keys
                if new_key not in ["experiment_name", "race_id", "is_baseline_for", "notes", "parent_experiment"]:
                    items.append((new_key, v))
        return dict(items)

    def _get_differentiating_hparams(self, all_configs: dict[str, dict]) -> list[str]:
        """
        Compares multiple configuration dictionaries and returns a list of keys
        where the values are not all the same.
        """
        if not all_configs or len(all_configs) < 2:
            return []

        # Flatten all configs to handle nested dictionaries
        flat_configs = {name: self._flatten_dict(cfg) for name, cfg in all_configs.items()}

        # Get a set of all keys across all configs
        all_keys = set()
        for cfg in flat_configs.values():
            all_keys.update(cfg.keys())

        diff_keys = []
        config_names = list(flat_configs.keys())

        for key in sorted(list(all_keys)):
            first_val = flat_configs[config_names[0]].get(key)
            is_different = False
            for i in range(1, len(config_names)):
                current_val = flat_configs[config_names[i]].get(key)
                if current_val != first_val:
                    is_different = True
                    break

            if is_different:
                diff_keys.append(key)

        return diff_keys

    def _get_race_summary_path(self):
        """Gets the path for the race's summary file, creating the dir if needed."""
        race_dir = os.path.join(RACES_DIR, self.race_info['base_name'])
        os.makedirs(race_dir, exist_ok=True)
        return os.path.join(race_dir, "race_summary.json")

    def _add_conclusion(self):
        summary_path = self._get_race_summary_path()

        # Load existing summary data if it exists
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
        except (IOError, json.JSONDecodeError):
            summary_data = {"race_name": self.race_info['base_name']}

        current_conclusion = summary_data.get("conclusion", "")
        conclusion, ok = QInputDialog.getMultiLineText(self, "Race Conclusion", "Enter your conclusion for this race:", current_conclusion)

        if ok:
            summary_data["conclusion"] = conclusion
            summary_data["last_updated"] = datetime.now().isoformat()
            try:
                with open(summary_path, 'w') as f:
                    json.dump(summary_data, f, indent=4)
                QMessageBox.information(self, "Success", "Conclusion saved.")
            except IOError as e:
                QMessageBox.critical(self, "Error", f"Could not save conclusion:\n{e}")

    def _save_summary_plot(self):
        race_dir = os.path.join(RACES_DIR, self.race_info['base_name'])
        os.makedirs(race_dir, exist_ok=True)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", race_dir, "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if not path:
            return

        # This is a bit tricky since the plots are in separate widgets.
        # For now, we'll just grab the first plot. A better implementation
        # might combine the plots into a single image.
        try:
            first_participant = self.participant_names[0]
            plot_widget = self.participant_widgets[first_participant]['plot']

            # Use QPixmap to grab the widget's contents
            pixmap = plot_widget.grab()
            if pixmap.save(path):
                QMessageBox.information(self, "Success", f"Plot saved to {os.path.basename(path)}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save plot.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save plot:\n{e}")

    def _export_summary_data(self):
        race_dir = os.path.join(RACES_DIR, self.race_info['base_name'])
        os.makedirs(race_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary", race_dir, "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                header = [self.summary_table.horizontalHeaderItem(i).text() for i in range(self.summary_table.columnCount())]
                writer.writerow(header)

                # Write data rows
                for row in range(self.summary_table.rowCount()):
                    row_data = [self.summary_table.item(row, col).text() for col in range(self.summary_table.columnCount())]
                    writer.writerow(row_data)

            QMessageBox.information(self, "Success", f"Summary data exported to {os.path.basename(path)}")

        except IOError as e:
            QMessageBox.critical(self, "Error", f"Could not export summary:\n{e}")


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
