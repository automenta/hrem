from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QMessageBox, QDialog, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QListWidget, QListWidgetItem, QDialogButtonBox,
    QAbstractItemView
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from pyqtgraph.exporters import ImageExporter
import pyqtgraph as pg

import json
import csv
from datetime import datetime
from .experiment_manager import ExperimentManager
from .config_editor import ConfigEditor
from .constants import RESULTS_DIR, STATUS_RUNNING, RACES_DIR
from .combined_race_plot import CombinedRacePlot
from .utils import flatten_dict
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

        self.participant_names = []
        self.plot_data_errors = {}
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

        # --- Main Splitter: Plot on left, tables on right ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # --- Left side: Combined Plot ---
        self.combined_plot = CombinedRacePlot()
        splitter.addWidget(self.combined_plot)

        # --- Right side: Tables ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        splitter.addWidget(right_pane)

        # Participant Details Table
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(3)
        self.details_table.setHorizontalHeaderLabels(["Participant", "Config", "Log"])
        self.details_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.details_table.verticalHeader().setVisible(False)
        self.details_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.details_table.cellClicked.connect(self._on_details_table_click)
        right_layout.addWidget(self.details_table)

        # Hyperparameter Table
        self.hparam_table = QTableWidget()
        self.hparam_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.hparam_table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.hparam_table)

        # Summary Table
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(4)
        self.summary_table.setHorizontalHeaderLabels(["Participant", "Status", "Latest Test Loss", "Notes"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.summary_table)

        splitter.setSizes([700, 500]) # Initial sizing

        # --- Bottom Bar: Analysis and Controls ---
        bottom_bar = QHBoxLayout()
        main_layout.addLayout(bottom_bar)

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
        self.analysis_toolbar.hide()

        bottom_bar.addWidget(self.analysis_toolbar)
        bottom_bar.addStretch()

        self.stop_button = QPushButton("Stop Race")
        self.stop_button.clicked.connect(self._stop_race)
        bottom_bar.addWidget(self.stop_button)

    def _start_monitoring(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_race_status)
        self.timer.start(2000)  # Update every 2 seconds
        self._update_details_table() # Initial population

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
        if not new_metrics:
            QMessageBox.warning(self, "Selection Error", "Please select at least one metric to plot.")
            return
        self.selected_metrics = new_metrics
        self._update_race_status()

    def _update_race_status(self):
        self.manager.update_log_files()
        statuses = self.manager.get_statuses()

        self.combined_plot.clear_plots()
        title = f"Metrics: {', '.join(self.selected_metrics)}"
        self.combined_plot.set_title(title)

        for name in self.participant_names:
            results_data, error_msg = self.manager.load_experiment_results(name)
            is_challenger = "_challenger" in name

            if error_msg:
                self.plot_data_errors[name] = error_msg
            else:
                self.plot_data_errors.pop(name, None)

            if results_data:
                for metric in self.selected_metrics:
                    self.combined_plot.update_plot(
                        participant_name=name,
                        is_challenger=is_challenger,
                        metric_name=metric,
                        data=results_data.get(metric, [])
                    )

        self._update_hyperparameter_table()
        self._update_summary(statuses)

        is_race_running = any(statuses.get(name) == STATUS_RUNNING for name in self.participant_names)
        if not is_race_running:
            self.timer.stop()
            self.stop_button.setText("Close")
            try:
                self.stop_button.clicked.disconnect(self._stop_race)
            except TypeError: pass
            self.stop_button.clicked.connect(self.close)
            self.stop_button.setEnabled(True)
            self.analysis_toolbar.show()

    def _update_summary(self, statuses: dict):
        self.summary_table.setRowCount(len(self.participant_names))
        all_losses = {}
        for i, name in enumerate(self.participant_names):
            status = statuses.get(name, "Unknown")
            loss_str = "N/A"
            results, err = self.manager.load_experiment_results(name)
            if err:
                loss_str = "Error"
            elif results and "test_loss" in results and results["test_loss"]:
                latest_loss = results["test_loss"][-1]
                all_losses[name] = latest_loss
                loss_str = f"{latest_loss:.4f}"

            notes = "Challenger" if "_challenger" in name else "Baseline"
            self.summary_table.setItem(i, 0, QTableWidgetItem(name))
            self.summary_table.setItem(i, 1, QTableWidgetItem(status))
            self.summary_table.setItem(i, 2, QTableWidgetItem(loss_str))
            self.summary_table.setItem(i, 3, QTableWidgetItem(notes))

        if self.participant_names[0] in all_losses and len(all_losses) > 1:
            challenger_loss = all_losses.get(self.participant_names[0])
            baseline_losses = [v for k, v in all_losses.items() if k != self.participant_names[0]]
            if challenger_loss is not None and baseline_losses:
                best_baseline_loss = min(baseline_losses)
                notes_item = self.summary_table.item(0, 3)
                current_notes = notes_item.text()
                if challenger_loss < best_baseline_loss:
                    notes_item.setText(f"{current_notes} (Winning)")
                elif challenger_loss > best_baseline_loss:
                    notes_item.setText(f"{current_notes} (Losing)")
                else:
                    notes_item.setText(f"{current_notes} (Tied)")

    def _update_details_table(self):
        self.details_table.setRowCount(len(self.participant_names))
        for row, name in enumerate(self.participant_names):
            self.details_table.setItem(row, 0, QTableWidgetItem(name))

            view_config_btn = QPushButton("View Config")
            self.details_table.setCellWidget(row, 1, view_config_btn)

            view_log_btn = QPushButton("View Log")
            self.details_table.setCellWidget(row, 2, view_log_btn)

    def _on_details_table_click(self, row, column):
        participant_name = self.details_table.item(row, 0).text()
        if column == 1: # View Config
            self._view_config(participant_name)
        elif column == 2: # View Log
            self._view_log(participant_name)

    def _update_hyperparameter_table(self):
        all_configs = {}
        for name in self.participant_names:
            config, err = self.manager.load_experiment_config(name)
            if config and not err:
                all_configs[name] = config

        if len(all_configs) < 2:
            self.hparam_table.hide()
            return

        diff_params = self._get_differentiating_hparams(all_configs)
        if not diff_params:
            self.hparam_table.hide()
            return

        self.hparam_table.show()
        self.hparam_table.setRowCount(len(diff_params))
        self.hparam_table.setColumnCount(len(self.participant_names) + 1)
        self.hparam_table.setHorizontalHeaderLabels(["Hyperparameter"] + self.participant_names)

        for row, param_key in enumerate(diff_params):
            self.hparam_table.setItem(row, 0, QTableWidgetItem(param_key))
            flat_configs = {name: self._flatten_dict(cfg) for name, cfg in all_configs.items()}
            for col, p_name in enumerate(self.participant_names, 1):
                value = flat_configs.get(p_name, {}).get(param_key, "N/A")
                self.hparam_table.setItem(row, col, QTableWidgetItem(str(value)))

        self.hparam_table.resizeColumnsToContents()

    def _get_differentiating_hparams(self, all_configs: dict[str, dict]) -> list[str]:
        if not all_configs or len(all_configs) < 2: return []

        # Define keys to ignore during diff
        ignored_keys = {"experiment_name", "race_id", "is_baseline_for", "notes", "parent_experiment"}

        flat_configs = {name: flatten_dict(cfg) for name, cfg in all_configs.items()}

        # Get all unique keys across all configs, excluding ignored ones
        all_keys = set()
        for cfg in flat_configs.values():
            all_keys.update(k for k in cfg.keys() if k not in ignored_keys)

        diff_keys = []
        config_names = list(flat_configs.keys())
        for key in sorted(list(all_keys)):
            first_val = flat_configs[config_names[0]].get(key)
            if any(flat_configs[name].get(key) != first_val for name in config_names[1:]):
                diff_keys.append(key)
        return diff_keys

    def _get_race_summary_path(self):
        race_dir = os.path.join(RACES_DIR, self.race_info['base_name'])
        os.makedirs(race_dir, exist_ok=True)
        return os.path.join(race_dir, "race_summary.json")

    def _add_conclusion(self):
        summary_path = self._get_race_summary_path()
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
        except (IOError, json.JSONDecodeError):
            summary_data = {"race_name": self.race_info['base_name']}
        current_conclusion = summary_data.get("conclusion", "")
        conclusion, ok = QInputDialog.getMultiLineText(self, "Race Conclusion", "Enter your conclusion:", current_conclusion)
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
        if not path: return
        try:
            exporter = ImageExporter(self.combined_plot.getPlotItem())
            exporter.export(path)
            QMessageBox.information(self, "Success", f"Plot saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save plot:\n{e}")

    def _export_summary_data(self):
        race_dir = os.path.join(RACES_DIR, self.race_info['base_name'])
        os.makedirs(race_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary", race_dir, "CSV files (*.csv)"
        )
        if not path: return
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [self.summary_table.horizontalHeaderItem(i).text() for i in range(self.summary_table.columnCount())]
                writer.writerow(header)
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
