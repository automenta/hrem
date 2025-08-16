from PyQt6.QtCore import pyqtSignal, Qt, QObject, QThread, QTimer, QDateTime
from PyQt6.QtGui import QAction, QFont, QColor
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
    QGridLayout,
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

        # For hover events
        self.hover_label = pg.TextItem(anchor=(0,1), border='w', fill=(0, 0, 0, 150))
        self.hover_label.hide()
        self.plot_widget.addItem(self.hover_label)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def reset(self):
        """Clears the plot and resets all participant data."""
        self.plot_widget.clear()
        self.participants = {}
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("left", "Test Loss")
        self.plot_widget.setLabel("bottom", "Step")
        self.hover_label.hide()

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

            # Update curve data and store as numpy array for faster processing
            import numpy as np
            x_data = np.arange(len(test_loss_data))
            y_data = np.array(test_loss_data)
            self.participants[p_name]["curve"].setData(x=x_data, y=y_data)
            self.participants[p_name]["data"] = (x_data, y_data)

            # Check for leader
            current_loss = y_data[-1]
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

    def get_participant_color(self, name: str) -> QColor | None:
        """Returns the QColor for a given participant."""
        participant = self.participants.get(name)
        return participant["pen"].color() if participant else None

    def _on_curve_clicked(self, curve):
        """When a curve is clicked, find its name and emit a signal."""
        for name, p_info in self.participants.items():
            if p_info["curve"] == curve:
                self.sigPlotClicked.emit(name)
                return

    def _on_mouse_moved(self, pos):
        """Handle mouse hover events to show data point details."""
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
        found_point = False

        for name, p_info in self.participants.items():
            curve = p_info["curve"]
            # Check if the mouse is near this curve
            if curve.mouseShape().contains(mouse_point):
                x_data, y_data = p_info["data"]
                if len(x_data) == 0:
                    continue

                # Find the closest point on the curve
                distances = (x_data - mouse_point.x())**2 + (y_data - mouse_point.y())**2
                closest_index = distances.argmin()

                step = x_data[closest_index]
                loss = y_data[closest_index]

                self.hover_label.setText(f"{name}\nStep: {step}\nLoss: {loss:.4f}")
                self.hover_label.setPos(mouse_point.x(), mouse_point.y())
                self.hover_label.show()
                found_point = True
                break # Show label for the first curve found

        if not found_point:
            self.hover_label.hide()


class LeaderCard(QFrame):
    """A card to display the current leader's stats."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LeaderCard")
        self._init_ui()

    def _init_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QGridLayout(self)

        # Color swatch
        self.color_swatch = QFrame()
        self.color_swatch.setFixedSize(20, 20)
        layout.addWidget(self.color_swatch, 0, 0, 2, 1)

        # Leader name
        self.name_label = QLabel("N/A")
        font = self.name_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label, 0, 1)

        # "Current Leader" subtitle
        self.subtitle_label = QLabel("Current Leader")
        font = self.subtitle_label.font()
        font.setPointSize(10)
        font.setItalic(True)
        self.subtitle_label.setFont(font)
        self.subtitle_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.subtitle_label, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)

        # Stats
        self.loss_label = self._create_stat_label("Loss")
        self.step_label = self._create_stat_label("Step")
        self.params_label = self._create_stat_label("Params") # Placeholder
        self.time_label = self._create_stat_label("Epoch Time") # Placeholder

        layout.addWidget(self.loss_label, 0, 2)
        layout.addWidget(self.step_label, 1, 2)
        layout.addWidget(self.params_label, 0, 3)
        layout.addWidget(self.time_label, 1, 3)

        layout.setColumnStretch(4, 1) # Push stats to the left

    def _create_stat_label(self, name: str) -> QLabel:
        label = QLabel(f"<b>{name}:</b> N/A")
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    def update_data(self, name: str, stats: dict, color: QColor):
        self.name_label.setText(name)
        self.loss_label.setText(f"<b>Loss:</b> {stats.get('loss', 'N/A'):.4f}")
        self.step_label.setText(f"<b>Step:</b> {stats.get('step', 'N/A')}")
        # Add more stats as they become available
        # self.params_label.setText(f"<b>Params:</b> {stats.get('params', 'N/A')}")
        # self.time_label.setText(f"<b>Epoch Time:</b> {stats.get('epoch_time', 'N/A')}")
        self.color_swatch.setStyleSheet(f"background-color: {color.name()}; border-radius: 10px;")
        self.setStyleSheet(f"""
            #LeaderCard {{
                border: 2px solid {color.name()};
                border-radius: 8px;
                background-color: #2E2E2E;
            }}
        """)

    def reset_card(self):
        self.name_label.setText("N/A")
        self.loss_label.setText("<b>Loss:</b> N/A")
        self.step_label.setText("<b>Step:</b> N/A")
        self.params_label.setText("<b>Params:</b> N/A")
        self.time_label.setText("<b>Epoch Time:</b> N/A")
        neutral_color = QColor("#555")
        self.color_swatch.setStyleSheet(f"background-color: {neutral_color.name()}; border-radius: 10px;")
        self.setStyleSheet(f"""
            #LeaderCard {{
                border: 2px solid #555;
                border-radius: 8px;
                background-color: #2E2E2E;
            }}
        """)


class RaceControlPanel(QWidget):
    """
    A widget for displaying race commentary and leader status.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Use spacing on children

        # --- Leader Card ---
        self.leader_card = LeaderCard()
        layout.addWidget(self.leader_card)

        # --- Title ---
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label = QLabel("Race Commentary")
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("margin-top: 10px;")
        layout.addWidget(self.title_label)

        # --- Commentary Box ---
        self.commentary_box = QTextEdit()
        self.commentary_box.setReadOnly(True)
        commentary_font = QFont("Courier New", 10)
        self.commentary_box.setFont(commentary_font)
        self.commentary_box.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                border: 1px solid #444;
                border-radius: 4px;
                color: #ddd;
            }
        """)
        layout.addWidget(self.commentary_box, stretch=1)

    def reset_panel(self, race_id):
        self.title_label.setText(f"Commentary: {race_id}")
        self.leader_card.reset_card()
        self.commentary_box.clear()
        self.add_commentary("🏁 And they're off! The race has begun.")

    def update_leader(self, name: str, stats: dict, color: QColor):
        self.leader_card.update_data(name, stats, color)

    def add_commentary(self, text):
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.commentary_box.append(f"<font color='#888'>[{timestamp}]</font> {text}")


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
        self._last_race_state = {} # To store previous state for commentary
        self.finish_line = None # To hold the InfiniteLine object
        self._race_winner = None # To track the winner
        self._commentary_cooldowns = defaultdict(int) # To prevent spam
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

        # Reset race-specific state
        self._last_race_state = {}
        self._race_winner = None
        self._commentary_cooldowns.clear()
        if self.finish_line:
            self.plot_widget.removeItem(self.finish_line)
            self.finish_line = None

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

        # --- Add Finish Line (hardcoded for now) ---
        finish_line_value = 0.05
        self.finish_line = pg.InfiniteLine(pos=finish_line_value, angle=0, movable=False,
                                           pen=pg.mkPen('y', style=Qt.PenStyle.DashLine, width=2),
                                           label='Finish Line',
                                           labelOpts={'position': 0.9, 'color': 'y'})
        self.plot_widget.addItem(self.finish_line)


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

    def _generate_commentary(self, current_state, last_state, race_id):
        """The Commentary Engine."""
        if not current_state:
            return

        # Cooldown management
        for key in list(self._commentary_cooldowns.keys()):
            self._commentary_cooldowns[key] -= 1
            if self._commentary_cooldowns[key] <= 0:
                del self._commentary_cooldowns[key]

        # Sort by loss
        sorted_racers = sorted(current_state.values(), key=lambda x: x.get('loss', float('inf')))
        leader = sorted_racers[0]
        leader_name = leader['name']

        # 1. Finish line check
        if self._race_winner is None and leader.get('loss', float('inf')) < self.finish_line.value():
            self._race_winner = leader_name
            self.race_control_panel.add_commentary(f"🎉🎉🎉 <b>{leader_name}</b> has crossed the finish line! VICTORY! 🎉🎉🎉")
            return # Stop other commentary

        # 2. Leader change
        last_leader_name = last_state.get('leader_name')
        if leader_name != last_leader_name:
            challenger_info = self.races.get(race_id, {}).get('challenger', {})
            is_challenger_lead = leader_name == challenger_info.get('name')
            if is_challenger_lead and last_leader_name:
                 self.race_control_panel.add_commentary(f"🚀 Challenger <b>{leader_name}</b> overtakes {last_leader_name} for the lead!")
            else:
                 self.race_control_panel.add_commentary(f"🏆 <b>{leader_name}</b> has taken the lead!")

        # 3. Close race check
        if len(sorted_racers) > 1 and 'close_race' not in self._commentary_cooldowns:
            p1 = sorted_racers[0]
            p2 = sorted_racers[1]
            if p1['loss'] > 0 and abs(p1['loss'] - p2['loss']) / p1['loss'] < 0.05: # 5% difference
                self.race_control_panel.add_commentary(f"🏇 It's neck and neck! <b>{p1['name']}</b> and <b>{p2['name']}</b> are battling for the lead!")
                self._commentary_cooldowns['close_race'] = 3 # Cooldown for 3 updates

        # 4. Stall check
        for name, stats in current_state.items():
            if name in last_state and 'stall_' + name not in self._commentary_cooldowns:
                 # Check if loss hasn't improved in the last update
                 if stats.get('loss') is not None and stats['loss'] == last_state[name].get('loss'):
                     self.race_control_panel.add_commentary(f"🤔 <b>{name}</b> seems to have stalled, making no progress.")
                     self._commentary_cooldowns['stall_' + name] = 5 # Cooldown for 5 updates

        # 5. Random flavor text
        if random.random() < 0.1:
            comment = random.choice([
                f"Looking strong, <b>{leader_name}</b>!",
                f"Incredible performance by <b>{leader_name}</b>.",
                f"<b>{leader_name}</b> is widening the gap!",
            ])
            self.race_control_panel.add_commentary(comment)


    def _populate_race_details(self, data):
        """
        Populates the UI with data loaded from the background worker.
        """
        race_id = data["race_id"]
        loaded_data = data["loaded_data"]

        current_item = self.race_list.currentItem()
        if not current_item or current_item.text() != race_id:
            return

        self.plot_widget.setTitle(f"Challenge: {race_id}")

        leader_info = self.animation_manager.update_data(loaded_data)

        # --- Process data for commentary ---
        current_race_state = {}
        for item in loaded_data:
            name = item["participant"]["name"]
            loss_data = item.get("results", {}).get("test_loss", [])
            if loss_data:
                current_race_state[name] = {
                    "name": name,
                    "loss": loss_data[-1],
                    "step": len(loss_data),
                    "is_challenger": not item["participant"].get("is_baseline_for")
                }
        if leader_info['name'] != 'N/A':
             current_race_state['leader_name'] = leader_info['name']

        # --- Commentary Engine ---
        if not self._race_winner: # Don't generate more comments after a win
            self._generate_commentary(current_race_state, self._last_race_state, race_id)

        # --- Update Control Panel ---
        if leader_info["name"] != "N/A":
            leader_color = self.animation_manager.get_participant_color(leader_info["name"])
            if leader_color:
                self.race_control_panel.update_leader(leader_info["name"], leader_info, leader_color)

        # --- Error Reporting ---
        for item in loaded_data:
            if item["error"]:
                self.race_control_panel.add_commentary(f"⚠️ Error for <b>{item['participant']['name']}</b>: {item['error']}")

        # --- Update state for next iteration ---
        self._last_race_state = current_race_state

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
