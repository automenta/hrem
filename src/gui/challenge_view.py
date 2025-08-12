from PyQt6.QtCore import pyqtSignal
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
)
import pyqtgraph as pg
from collections import defaultdict


class ChallengeView(QWidget):
    """
    A widget to display and analyze baseline challenges (races).
    """

    experiment_selected = pyqtSignal(str)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.races = {}  # To store grouped race data
        self._init_ui()

        self.race_list.currentItemChanged.connect(self.display_race_details)

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(self)
        layout.addWidget(splitter)

        # --- Left Panel (Race List) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Active Challenges"))
        self.race_list = QListWidget()
        self.race_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        left_layout.addWidget(self.race_list)
        splitter.addWidget(left_panel)

        # --- Right Panel (Details) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.plot_widget = pg.PlotWidget()
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        right_layout.addWidget(self.plot_widget)
        right_layout.addWidget(self.results_table)
        splitter.addWidget(right_panel)

        splitter.setSizes([200, 600])

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
            self.race_list.currentItem().text() if self.race_list.currentItem() else None
        )

        experiments = self.manager.get_experiments_data()
        self.races = self._group_experiments_by_race(experiments)

        self.race_list.clear()
        for race_id in sorted(self.races.keys()):
            self.race_list.addItem(race_id)

        # Restore selection
        if current_selection:
            items = self.race_list.findItems(current_selection, Qt.MatchFlag.MatchExactly)
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
                else: # This is the challenger
                    races[race_id]["challenger"] = exp
        return races

    def display_race_details(self):
        """
        Displays the plot and table for the currently selected race.
        """
        self.plot_widget.clear()
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)

        current_item = self.race_list.currentItem()
        if not current_item:
            self.plot_widget.setTitle("No Challenge Selected")
            return

        race_id = current_item.text()
        race_data = self.races.get(race_id)
        if not race_data:
            return

        challenger = race_data.get("challenger")
        baselines = race_data.get("baselines", [])
        all_participants = ([challenger] if challenger else []) + baselines

        self.plot_widget.setTitle(f"Challenge: {race_id}")
        self.plot_widget.addLegend()

        # --- Plotting ---
        colors = ["b", "r", "g", "c", "m", "y", "w"]
        for i, participant in enumerate(all_participants):
            if not participant: continue
            results, _ = self.manager.load_experiment_results(participant["name"])
            if results and "test_loss" in results:
                color = colors[i % len(colors)]
                pen = pg.mkPen(color, width=2)
                self.plot_widget.plot(
                    results["test_loss"],
                    pen=pen,
                    name=participant["name"],
                )

        # --- Table ---
        headers = ["Experiment", "Model", "Status", "Final Loss", "Params"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(all_participants))

        for row, p in enumerate(all_participants):
            if not p: continue
            self.results_table.setItem(row, 0, QTableWidgetItem(p["name"]))
            self.results_table.setItem(row, 1, QTableWidgetItem(p["model"]))
            self.results_table.setItem(row, 2, QTableWidgetItem(p["status"]))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(p["final_loss"])))
            self.results_table.setItem(row, 4, QTableWidgetItem(str(p["params"])))

        self.results_table.resizeColumnsToContents()
