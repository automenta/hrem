from PyQt6.QtCore import pyqtSignal, Qt, QObject, QThread, QTimer, QDateTime
from PyQt6.QtGui import QAction, QFont
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
    QFrame,
)
import pyqtgraph as pg
from collections import defaultdict
import random
from PyQt6.QtCore import QDateTime


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


class RaceAnimationManager(QObject):
    """
    Manages the plotting and animation of race data.
    """
    sigPlotClicked = pyqtSignal(str)

    def __init__(self, plot_widget):
        super().__init__()
        self.plot_widget = plot_widget
        self.participants = {}  # name -> { "curve": pg.PlotDataItem, "data": [], "pen": QPen }
        self.leader_pen = pg.mkPen(width=4)
        self.normal_pen_width = 2

    def reset(self):
        """Clears the plot and resets all participant data."""
        self.plot_widget.clear()
        self.participants = {}
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("left", "Test Loss")
        self.plot_widget.setLabel("bottom", "Step")

    def update_data(self, loaded_data):
        """
        Updates the plot with new data, creating or extending lines as needed.
        Returns the current leader's info.
        """
        leader_info = {"name": "N/A", "loss": float('inf'), "step": 0}

        # First, add any new participants
        for i, item in enumerate(loaded_data):
            p_name = item["participant"]["name"]
            if p_name not in self.participants:
                hue = i / max(1, len(loaded_data))
                color = pg.mkColor(hue=hue, sat=200, val=255)
                pen = pg.mkPen(color, width=self.normal_pen_width)
                curve = self.plot_widget.plot(pen=pen, name=p_name)
                curve.sigClicked.connect(lambda c=curve: self._on_curve_clicked(c))
                self.participants[p_name] = {"curve": curve, "data": [], "pen": pen}

        # Now, update all data and find the leader
        for item in loaded_data:
            p_name = item["participant"]["name"]
            results = item.get("results", {})
            test_loss_data = results.get("test_loss", [])

            if not test_loss_data:
                continue

            # Update curve data
            self.participants[p_name]["curve"].setData(test_loss_data)
            self.participants[p_name]["data"] = test_loss_data

            # Check for leader
            current_loss = test_loss_data[-1]
            if current_loss < leader_info["loss"]:
                leader_info = {
                    "name": p_name,
                    "loss": current_loss,
                    "step": len(test_loss_data)
                }

        # Highlight the leader
        self._highlight_leader(leader_info["name"])
        return leader_info

    def _highlight_leader(self, leader_name):
        """Makes the leader's line thicker and others normal."""
        for name, p_info in self.participants.items():
            original_color = p_info["pen"].color()
            if name == leader_name:
                new_pen = pg.mkPen(original_color, width=4)
                p_info["curve"].setPen(new_pen)
                # Bring leader to front
                p_info["curve"].setZValue(1)
            else:
                new_pen = pg.mkPen(original_color, width=self.normal_pen_width)
                p_info["curve"].setPen(new_pen)
                p_info["curve"].setZValue(0)

    def _on_curve_clicked(self, curve):
        """When a curve is clicked, find its name and emit a signal."""
        for name, p_info in self.participants.items():
            if p_info["curve"] == curve:
                self.sigPlotClicked.emit(name)
                return


class RaceControlPanel(QWidget):
    """
    A widget for displaying race commentary and leader status.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Title ---
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label = QLabel("Race Commentary")
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        # --- Leader Status ---
        self.leader_frame = QFrame()
        self.leader_frame.setFrameShape(QFrame.Shape.StyledPanel)
        leader_layout = QVBoxLayout(self.leader_frame)

        leader_title_font = QFont()
        leader_title_font.setBold(True)
        leader_title_font.setPointSize(11)
        leader_label = QLabel("Current Leader")
        leader_label.setFont(leader_title_font)
        leader_layout.addWidget(leader_label)

        self.leader_name_label = QLabel("N/A")
        self.leader_stats_label = QLabel("Loss: N/A | Step: N/A")
        leader_layout.addWidget(self.leader_name_label)
        leader_layout.addWidget(self.leader_stats_label)
        layout.addWidget(self.leader_frame)


        # --- Commentary Box ---
        self.commentary_box = QTextEdit()
        self.commentary_box.setReadOnly(True)
        commentary_font = QFont("Courier New", 10)
        self.commentary_box.setFont(commentary_font)
        layout.addWidget(self.commentary_box)

        layout.setStretchFactor(self.commentary_box, 1) # Make box expand

    def reset_panel(self, race_id):
        self.title_label.setText(f"Race Commentary: {race_id}")
        self.leader_name_label.setText("N/A")
        self.leader_stats_label.setText("Loss: N/A | Step: N/A")
        self.commentary_box.clear()
        self.add_commentary("🏁 And they're off! The race has begun.")

    def update_leader(self, name, loss, step):
        self.leader_name_label.setText(name)
        self.leader_stats_label.setText(f"Loss: {loss:.4f} | Step: {step}")

    def add_commentary(self, text):
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.commentary_box.append(f"[{timestamp}] {text}")


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
        self.animation_manager = RaceAnimationManager(self.plot_widget) # NEW
        self.animation_manager.sigPlotClicked.connect(self.load_config_and_logs) # NEW
        self.race_control_panel = RaceControlPanel() # NEW
        plot_summary_layout.addWidget(self.plot_widget, stretch=3) # Give more space to plot
        plot_summary_layout.addWidget(self.race_control_panel, stretch=1)
        self.right_tabs.addTab(plot_summary_widget, "Race View")


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
        # This method is now disconnected, but we'll keep it for the config/log tabs
        # We need a new way to select an experiment, maybe clicking the plot line
        # For now, this is not connected to anything.
        # A future implementation could involve clicking on a plot line.
        self.config_view.clear()
        self.log_view.clear()

        # The following code is placeholder and not currently active
        # as there is no table to select from.
        # selected_items = self.results_table.selectedItems()
        # if not selected_items:
        #     return
        # exp_name = selected_items[0].text() # Assuming name is the first item
        # self.load_config_and_logs(exp_name)
        pass

    def load_config_and_logs(self, exp_name):
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

        # Switch to the config tab for immediate feedback
        self.right_tabs.setCurrentWidget(self.config_view)


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
        self.animation_manager.reset()
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
        self.race_control_panel.reset_panel(race_id) # Reset commentary
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

        self.plot_widget.setTitle(f"Challenge: {race_id}")

        # --- Animation and Leader Tracking ---
        # This is now handled by the animation manager
        leader_info = self.animation_manager.update_data(loaded_data)

        # --- Update Control Panel ---
        if leader_info["name"] != "N/A":
            last_leader = getattr(self, "_last_leader", None)
            if leader_info["name"] != last_leader:
                self.race_control_panel.add_commentary(f"🏆 {leader_info['name']} has taken the lead!")
                self._last_leader = leader_info["name"]

            self.race_control_panel.update_leader(
                leader_info["name"], leader_info["loss"], leader_info["step"]
            )
            # Add some dynamic commentary
            if random.random() < 0.15: # 15% chance to add a comment
                comment = random.choice([
                    f"Looking strong, {leader_info['name']}!",
                    f"Incredible performance by {leader_info['name']}.",
                    f"{leader_info['name']} is widening the gap!",
                    f"What a run from {leader_info['name']}!",
                ])
                self.race_control_panel.add_commentary(comment)

        # --- Error Reporting ---
        for item in loaded_data:
            if item["error"]:
                self.race_control_panel.add_commentary(f"⚠️ Error for {item['participant']['name']}: {item['error']}")

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
