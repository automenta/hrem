from PyQt6.QtCore import pyqtSignal, Qt, QObject, QThread, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QTabWidget,
    QTextEdit,
)
import pyqtgraph as pg
from collections import defaultdict


class RaceDataWorker(QObject):
    """
    A worker to load experiment data for a race in the background.
    """
    data_loaded = pyqtSignal(dict)

    def __init__(self, manager, race_id, race_data):
        super().__init__()
        self.manager = manager
        self.race_id = race_id
        self.race_data = race_data

    def load_data(self):
        """
        Loads the data and emits the 'data_loaded' signal.
        """
        challenger = self.race_data.get("challenger")
        baselines = self.race_data.get("baselines", [])
        all_participants = ([challenger] if challenger else []) + baselines

        loaded_data = []
        for p in all_participants:
            if not p:
                continue
            results, error = self.manager.load_experiment_results(p["name"])
            loaded_data.append({
                "participant": p,
                "results": results,
                "error": error
            })

        output = {
            "race_id": self.race_id,
            "loaded_data": loaded_data
        }
        self.data_loaded.emit(output)


class ChallengeView(QWidget):
    """
    A widget to display and analyze baseline challenges (races).
    """

    experiment_selected = pyqtSignal(str)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.races = {}  # To store grouped race data
        self.data_thread = None
        self.data_worker = None
        self._init_ui()

        self.race_list.currentItemChanged.connect(self.display_race_details)

        # Timer for auto-refreshing running experiments
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(5000)  # 5 seconds
        self.refresh_timer.timeout.connect(self._check_and_refresh_running)
        self.refresh_timer.start()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(self)
        layout.addWidget(splitter)

        # --- Left Panel (Race List) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Active Challenges"))
        self.race_list = QListWidget()
        self.race_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.race_list.customContextMenuRequested.connect(self._show_context_menu)
        self.race_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        left_layout.addWidget(self.race_list)
        splitter.addWidget(left_panel)

        # --- Right Panel (Tabbed View) ---
        self.right_tabs = QTabWidget()
        splitter.addWidget(self.right_tabs)

        # -- Plot & Summary Tab --
        plot_summary_widget = QWidget()
        plot_summary_layout = QVBoxLayout(plot_summary_widget)
        self.plot_widget = pg.PlotWidget()
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.itemSelectionChanged.connect(self._on_experiment_selected)
        plot_summary_layout.addWidget(self.plot_widget)
        plot_summary_layout.addWidget(self.results_table)
        self.right_tabs.addTab(plot_summary_widget, "Plot & Summary")

        # -- Config Tab --
        self.config_view = QTextEdit()
        self.config_view.setReadOnly(True)
        self.config_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.right_tabs.addTab(self.config_view, "Configuration")

        # -- Log Tab --
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.right_tabs.addTab(self.log_view, "Logs")

        splitter.setSizes([250, 650])

    def _on_experiment_selected(self):
        """
        When an experiment is selected in the table, load its config and logs.
        """
        self.config_view.clear()
        self.log_view.clear()

        selected_items = self.results_table.selectedItems()
        if not selected_items:
            return

        # The name is in the first column of the selected row
        selected_row = selected_items[0].row()
        exp_name_item = self.results_table.item(selected_row, 0)
        if not exp_name_item:
            return

        exp_name = exp_name_item.text()

        # Load and display config
        config, err = self.manager.load_experiment_config(exp_name)
        if err:
            self.config_view.setText(f"Error loading config: {err}")
        elif config:
            # Pretty-print the JSON config
            import json
            self.config_view.setText(json.dumps(config, indent=4))
        else:
            self.config_view.setText(f"No config file found for {exp_name}.")

        # Load and display logs
        log_content = self.manager.get_log_contents(exp_name)
        if log_content:
            self.log_view.setText(log_content)
        else:
            self.log_view.setText(f"No output.log file found for {exp_name}.")

    def _on_item_double_clicked(self, item):
        """When a race is double-clicked, we can select the challenger in the main experiments tab."""
        race_id = item.text()
        if race_id in self.races:
            challenger = self.races[race_id].get("challenger")
            if challenger:
                self.experiment_selected.emit(challenger["name"])

    def refresh(self):
        """
        Refreshes the view by fetching the latest experiment data and regrouping races.
        """
        current_selection = (
            self.race_list.currentItem().text()
            if self.race_list.currentItem()
            else None
        )

        experiments = self.manager.get_experiments_data()
        self.races = self._group_experiments_by_race(experiments)

        self.race_list.clear()
        for race_id in sorted(self.races.keys()):
            self.race_list.addItem(race_id)

        # Restore selection
        if current_selection:
            items = self.race_list.findItems(
                current_selection, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.race_list.setCurrentItem(items[0])

        self.display_race_details()

    def _group_experiments_by_race(self, experiments: list) -> dict:
        """
        Groups experiments into a dictionary keyed by race_id.
        """
        races = defaultdict(lambda: {"challenger": None, "baselines": []})
        for exp in experiments:
            race_id = exp.get("race_id")
            if race_id and race_id != "N/A":
                if exp.get("is_baseline_for"):
                    races[race_id]["baselines"].append(exp)
                else:  # This is the challenger
                    races[race_id]["challenger"] = exp
        return races

    def display_race_details(self):
        """
        Clears the view and starts a background worker to load details for the selected race.
        """
        # Clear existing views
        self.plot_widget.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.config_view.clear()
        self.log_view.clear()
        self.plot_widget.setTitle("Loading...")

        # Stop any previous worker
        if self.data_thread and self.data_thread.isRunning():
            self.data_thread.quit()
            self.data_thread.wait()

        current_item = self.race_list.currentItem()
        if not current_item:
            self.plot_widget.setTitle("No Challenge Selected")
            return

        race_id = current_item.text()
        race_data = self.races.get(race_id)
        if not race_data:
            self.plot_widget.setTitle("Error: Race data not found.")
            return

        # --- Setup and run background worker ---
        self.data_thread = QThread()
        self.data_worker = RaceDataWorker(self.manager, race_id, race_data)
        self.data_worker.moveToThread(self.data_thread)

        # Connect signals
        self.data_thread.started.connect(self.data_worker.load_data)
        self.data_worker.data_loaded.connect(self._populate_race_details)
        self.data_worker.data_loaded.connect(self.data_thread.quit)
        self.data_thread.finished.connect(self.data_thread.deleteLater)

        self.data_thread.start()

    def _populate_race_details(self, data):
        """
        Populates the UI with data loaded from the background worker.
        """
        race_id = data["race_id"]
        loaded_data = data["loaded_data"]

        # Check if the race is still the one selected
        current_item = self.race_list.currentItem()
        if not current_item or current_item.text() != race_id:
            return # A different race has been selected since we started loading

        self.plot_widget.clear()
        self.plot_widget.setTitle(f"Challenge: {race_id}")
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("left", "Test Loss")
        self.plot_widget.setLabel("bottom", "Step")

        all_participants = [item['participant'] for item in loaded_data]

        # --- Plotting ---
        for i, item in enumerate(loaded_data):
            participant = item["participant"]
            results = item["results"]
            error = item["error"]

            if error:
                print(f"Error loading results for {participant['name']}: {error}")
                continue

            test_loss_data = results.get("test_loss") if results else None
            if not isinstance(test_loss_data, list) or not test_loss_data:
                continue

            hue = i / max(1, len(all_participants))
            color = pg.mkColor(hue=hue, sat=200, val=255)
            pen = pg.mkPen(color, width=2)
            self.plot_widget.plot(test_loss_data, pen=pen, name=participant["name"])

        # --- Table ---
        headers = ["Experiment", "Model", "Status", "Final Loss", "Params"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setRowCount(len(all_participants))

        for row, p in enumerate(all_participants):
            if not p:
                continue
            self.results_table.setItem(row, 0, QTableWidgetItem(p["name"]))
            self.results_table.setItem(row, 1, QTableWidgetItem(p["model"]))
            self.results_table.setItem(row, 2, QTableWidgetItem(p["status"]))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(p["final_loss"])))
            self.results_table.setItem(row, 4, QTableWidgetItem(str(p["params"])))

        self.results_table.resizeColumnsToContents()

    def _check_and_refresh_running(self):
        """
        Checks if the currently viewed race has running experiments and triggers a refresh.
        """
        current_item = self.race_list.currentItem()
        if not current_item:
            return

        race_id = current_item.text()
        race_data = self.races.get(race_id)
        if not race_data:
            return

        # Check if any participant is still running
        challenger = race_data.get("challenger")
        baselines = race_data.get("baselines", [])
        all_participants = ([challenger] if challenger else []) + baselines

        is_running = any(p and p.get("status") == "RUNNING" for p in all_participants)

        if is_running:
            # To avoid refreshing while another load is in progress
            if not self.data_thread or not self.data_thread.isRunning():
                self.display_race_details()


    def _show_context_menu(self, pos):
        """
        Shows a context menu for the race list.
        """
        item = self.race_list.itemAt(pos)
        if not item:
            return

        race_id = item.text()
        menu = self.race_list.createStandardContextMenu()
        menu.addSeparator()

        delete_action = QAction("Delete Race Permanently", self)
        delete_action.triggered.connect(lambda: self._delete_race(race_id))
        menu.addAction(delete_action)

        menu.exec(self.race_list.mapToGlobal(pos))

    def _delete_race(self, race_id):
        """
        Handles the logic for deleting a race after confirmation.
        """
        reply = QMessageBox.warning(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete the race '{race_id}' and all its associated experiments?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.manager.delete_race(race_id)
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)
            self.refresh()
