import json
import sys
import os

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QTabWidget,
    QTreeView,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QInputDialog,
    QAbstractItemView,
    QMenu,
)
from PyQt6.QtCore import Qt, QTimer, QItemSelectionModel
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction

from .experiment_manager import ExperimentManager
from .archive_dialog import ArchiveManagerDialog
from .trajectory_view import TrajectoryView
from .scatter_plot_view import ScatterPlotView
from .unified_launch_dialog import UnifiedLaunchDialog
from .challenge_view import ChallengeView
from .log_deck import LogDeckWindow
from .constants import (
    INITIAL_SPLITTER_SIZES,
    LAUNCH_DELAY_MS,
    REFRESH_INTERVAL_MS,
    STATUS_RUNNING,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

FILTER_CACHE_FILE = ".filter_cache.txt"

class MainGUI(QMainWindow):
    """
    The main window for the experiment GUI.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HREM Experimentation Platform")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.manager = ExperimentManager()
        self.comparison_list = []
        self.current_train_loss = []
        self.current_test_loss = []
        self.log_deck = LogDeckWindow(self)

        self._init_ui()
        self._load_filter_cache()
        self.refresh_ui()

        # --- Timer for live updates ---
        self.timer = QTimer()
        self.timer.setInterval(REFRESH_INTERVAL_MS)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start()

    def _load_filter_cache(self):
        if os.path.exists(FILTER_CACHE_FILE):
            try:
                with open(FILTER_CACHE_FILE, "r") as f:
                    filter_text = f.read()
                    self.filter_input.setText(filter_text)
            except IOError:
                pass # Ignore errors reading cache

    def _save_filter_cache(self):
        try:
            with open(FILTER_CACHE_FILE, "w") as f:
                f.write(self.filter_input.text())
        except IOError:
            pass # Ignore errors writing cache

    def closeEvent(self, event):
        self._save_filter_cache()
        super().closeEvent(event)

    def _init_ui(self):
        """
        Initializes the UI components.
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._create_experiments_tab()
        self._create_trajectory_tab()
        self._create_analysis_tab()
        self._create_challenges_tab()

    def _create_management_tab(
        self,
        tab_name: str,
        title: str,
        left_content_widget: QWidget,
        selection_changed_fn: callable,
        buttons: list,
        right_panel: QWidget,
    ):
        """
        Creates a standardized management tab with a content widget on the left and a display panel on the right.
        """
        tab = QWidget()
        self.tabs.addTab(tab, tab_name)
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel(title))

        if hasattr(left_content_widget, "selectionModel"):
            left_content_widget.selectionModel().selectionChanged.connect(
                selection_changed_fn
            )

        left_layout.addWidget(left_content_widget)

        button_layout = QHBoxLayout()
        for button in buttons:
            button_layout.addWidget(button)
        left_layout.addLayout(button_layout)
        splitter.addWidget(left_panel)

        # --- Right Panel ---
        splitter.addWidget(right_panel)
        splitter.setSizes(INITIAL_SPLITTER_SIZES)

    def _create_experiments_tab(self):
        """
        Creates the layout and widgets for the 'Mission Control' tab.
        """
        # --- Left Panel Widgets ---
        self.exp_tree = QTreeView()
        self.exp_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.exp_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.exp_tree.setSortingEnabled(True)
        self.exp_tree.header().setStretchLastSection(True)
        self.exp_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.exp_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.exp_tree.customContextMenuRequested.connect(
            self._show_experiment_context_menu
        )

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by name...")
        self.filter_input.textChanged.connect(self.filter_experiments)

        left_content_container = QWidget()
        left_content_layout = QVBoxLayout(left_content_container)
        left_content_layout.setContentsMargins(0, 0, 0, 0)
        left_content_layout.addWidget(self.filter_input)
        left_content_layout.addWidget(self.exp_tree)

        # --- Buttons ---
        self.launch_button = QPushButton("Launch New...")
        self.launch_button.setToolTip("Launch a new experiment, challenge, or search.")
        self.launch_button.clicked.connect(self.launch_new_experiment)
        self.archive_manager_button = QPushButton("Manage Archives...")
        self.archive_manager_button.clicked.connect(self.open_archive_manager)
        self.compare_button = QPushButton("Compare Selected")
        self.compare_button.clicked.connect(self.compare_selected)
        self.clear_comparison_button = QPushButton("Clear Comparison")
        self.clear_comparison_button.clicked.connect(self.clear_comparison)
        self.clear_comparison_button.setEnabled(False)

        # --- Right Panel ---
        right_panel = QSplitter(Qt.Orientation.Vertical)
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()

        metric_selector_layout = QHBoxLayout()
        metric_selector_layout.addWidget(QLabel("Metric:"))
        self.metric_selector = QComboBox()
        self.metric_selector.currentTextChanged.connect(
            self.update_selected_experiment_display
        )
        metric_selector_layout.addWidget(self.metric_selector)
        metric_selector_layout.addStretch()
        plot_layout.addLayout(metric_selector_layout)
        plot_layout.addWidget(self.plot_widget)

        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setFontFamily("monospace")

        self.diff_table = QTableWidget()
        self.diff_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diff_table.verticalHeader().setVisible(False)
        self.diff_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.diff_table.setVisible(False)

        right_panel.addWidget(plot_container)
        right_panel.addWidget(self.config_display)
        right_panel.addWidget(self.diff_table)
        right_panel.setSizes([600, 200, 100])

        self._create_management_tab(
            "Mission Control",
            "Experiments",
            left_content_container,
            self.update_selected_experiment_display,
            [
                self.launch_button,
                self.archive_manager_button,
                self.compare_button,
                self.clear_comparison_button,
            ],
            right_panel,
        )

        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.plot_label = pg.TextItem()
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        self.plot_widget.addItem(self.plot_label, ignoreBounds=True)
        self.v_line.hide()
        self.h_line.hide()
        self.plot_label.hide()
        self.plot_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_plot_hover,
        )

    def _show_experiment_context_menu(self, position):
        menu = QMenu()
        selected_names = self.get_selected_experiment_names()
        num_selected = len(selected_names)
        if num_selected == 0:
            return

        statuses = self.manager.get_experiment_statuses()
        selected_statuses = [statuses.get(name) for name in selected_names]
        are_any_running = any(s == STATUS_RUNNING for s in selected_statuses)
        are_all_stopped = all(s != STATUS_RUNNING for s in selected_statuses)

        view_logs_action = QAction("View Logs", self)
        view_logs_action.triggered.connect(self.view_selected_logs)
        view_logs_action.setEnabled(num_selected == 1)
        menu.addAction(view_logs_action)
        menu.addSeparator()

        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.stop_selected_experiments)
        stop_action.setEnabled(are_any_running)
        menu.addAction(stop_action)

        force_stop_action = QAction("Force Stop", self)
        force_stop_action.triggered.connect(self.force_stop_selected_experiments)
        force_stop_action.setEnabled(are_any_running)
        menu.addAction(force_stop_action)
        menu.addSeparator()

        clone_action = QAction("Clone...", self)
        clone_action.triggered.connect(self.clone_selected_experiment)
        clone_action.setEnabled(num_selected == 1 and are_all_stopped)
        menu.addAction(clone_action)

        rename_action = QAction("Rename...", self)
        rename_action.triggered.connect(self.rename_selected_experiment)
        rename_action.setEnabled(num_selected == 1 and are_all_stopped)
        menu.addAction(rename_action)

        select_parent_action = QAction("Select Parent", self)
        select_parent_action.triggered.connect(self.select_parent_experiment)
        select_parent_action.setEnabled(num_selected == 1)
        menu.addAction(select_parent_action)
        menu.addSeparator()

        archive_action = QAction("Archive...", self)
        archive_action.triggered.connect(self.archive_selected_experiments)
        archive_action.setEnabled(num_selected > 0 and are_all_stopped)
        menu.addAction(archive_action)

        menu.exec(self.exp_tree.viewport().mapToGlobal(position))

    def _create_trajectory_tab(self):
        self.trajectory_tab = QWidget()
        self.tabs.addTab(self.trajectory_tab, "Research Tree")
        layout = QVBoxLayout(self.trajectory_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.trajectory_view = TrajectoryView(self.manager)
        layout.addWidget(self.trajectory_view)

    def _create_analysis_tab(self):
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Analysis")
        layout = QVBoxLayout(self.analysis_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scatter_plot_view = ScatterPlotView(self.manager)
        self.scatter_plot_view.experiment_selected.connect(
            self.select_experiment_by_name
        )
        self.scatter_plot_view.experiments_selected_for_filtering.connect(
            self._filter_experiments_from_analysis
        )
        layout.addWidget(self.scatter_plot_view)

    def _create_challenges_tab(self):
        self.challenge_view = ChallengeView(self.manager)
        self.tabs.addTab(self.challenge_view, "Challenges")
        self.challenge_view.experiment_selected.connect(self.select_experiment_by_name)

    def _filter_experiments_from_analysis(self, names: list):
        if not names:
            return
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Mission Control":
                self.tabs.setCurrentIndex(i)
                break
        filter_text = f"name:{','.join(names)}"
        self.filter_input.setText(filter_text)

    def launch_new_experiment(self):
        default_config = {"model": {}, "dataset": {}, "training": {}}
        dialog = UnifiedLaunchDialog(config=default_config, parent=self)
        if not dialog.exec():
            return
        launch_info = dialog.get_launch_info()
        if not launch_info:
            return
        run_type = launch_info.get("type")
        success, message = False, "An unknown error occurred."
        if run_type == "Challenge":
            if hasattr(self.manager, "launch_experiment_race"):
                success, message = self.manager.launch_experiment_race(launch_info)
            else:
                message = "Functionality to launch challenges is not implemented."
        elif run_type in ["Single Run", "Hyperparameter Search"]:
            success, message = self.manager.launch_experiment_from_config(
                launch_info["config"], launch_info["name"]
            )
        else:
            message = f"Unknown run type '{run_type}'."
        if success:
            QTimer.singleShot(LAUNCH_DELAY_MS, self.refresh_ui)
        else:
            QMessageBox.warning(self, "Launch Failed", message)

    def clone_selected_experiment(self):
        exp_names = self.get_selected_experiment_names()
        if len(exp_names) != 1: return
        original_name = exp_names[0]
        original_config, err = self.manager.load_experiment_config(original_name)
        if err:
            QMessageBox.warning(self, "Clone Failed", f"Could not load config for {original_name}: {err}")
            return
        new_name = f"{original_name}_clone"
        original_config["parent_experiment"] = original_name
        dialog = UnifiedLaunchDialog(config=original_config, exp_name=new_name, parent=self)
        if dialog.exec():
            launch_info = dialog.get_launch_info()
            if launch_info:
                success, message = self.manager.launch_experiment_from_config(
                    launch_info["config"], launch_info["name"]
                )
                if success:
                    QTimer.singleShot(LAUNCH_DELAY_MS, self.refresh_ui)
                    self.select_experiment_by_name(launch_info["name"])
                else:
                    QMessageBox.warning(self, "Launch Failed", message)

    def populate_experiment_list(self):
        self.exp_tree.setSortingEnabled(False)
        current_selection = self.get_selected_experiment_name()
        graph = self.manager.get_experiment_graph()
        self.exp_model = QStandardItemModel()
        self.exp_model.setHorizontalHeaderLabels(
            ["Name", "Type", "Status", "Model", "Dataset", "Final Loss", "Created"]
        )
        self.exp_tree.setModel(self.exp_model)
        items = {}
        for name, data in graph["nodes"].items():
            row = [
                data["name"], data.get("type", "Single"), data["status"],
                data["model"], data["dataset"], data["final_loss"], data["created"],
            ]
            qt_items = [QStandardItem(str(field)) for field in row]
            qt_items[0].setData(data, role=Qt.ItemDataRole.UserRole)
            items[name] = qt_items
        for parent_name, child_name in graph["edges"]:
            if parent_name in items and child_name in items:
                items[parent_name][0].appendRow(items[child_name])
        for root_name in graph["roots"]:
            if root_name in items:
                self.exp_model.appendRow(items[root_name])
        self.exp_tree.expandAll()
        self.exp_tree.setSortingEnabled(True)
        for i in range(self.exp_model.columnCount()):
            self.exp_tree.resizeColumnToContents(i)
        if current_selection:
            self.select_experiment_by_name(current_selection)

    def filter_experiments(self, text):
        text_lower = text.lower()
        name_filters, kv_filters = [], {}
        if text_lower.startswith("name:"):
            kv_filters['name'] = text_lower[5:].split(',')
        else:
            for part in text_lower.split():
                if ":" in part:
                    key, value = part.split(":", 1)
                    kv_filters[key] = value
                else:
                    name_filters.append(part)
        def item_matches(item):
            if not item: return False
            data = item.data(role=Qt.ItemDataRole.UserRole)
            if not data: return False
            item_name_lower = data.get("name", "").lower()
            if any(f not in item_name_lower for f in name_filters): return False
            for key, value in kv_filters.items():
                if key == 'name' and isinstance(value, list):
                    if item_name_lower not in value: return False
                elif value not in str(data.get(key, "")).lower(): return False
            return True
        def recurse(parent_item):
            any_child_is_visible = False
            for r in range(parent_item.rowCount()):
                child_item = parent_item.child(r, 0)
                is_visible = recurse(child_item) or item_matches(child_item)
                self.exp_tree.setRowHidden(r, parent_item.index(), not is_visible)
                if is_visible: any_child_is_visible = True
            return any_child_is_visible
        if hasattr(self, "exp_model"):
            recurse(self.exp_model.invisibleRootItem())

    def get_selected_experiment_names(self):
        if not hasattr(self, "exp_tree"): return []
        selection_model = self.exp_tree.selectionModel()
        if not selection_model: return []
        selected_names = []
        for index in selection_model.selectedRows(column=0):
            item = self.exp_model.itemFromIndex(index)
            if item:
                exp_data = item.data(role=Qt.ItemDataRole.UserRole)
                if exp_data and "name" in exp_data:
                    selected_names.append(exp_data["name"])
        return selected_names

    def get_selected_experiment_name(self):
        names = self.get_selected_experiment_names()
        return names[0] if names else None

    def select_experiment_by_name(self, name_to_select: str):
        if not name_to_select or not hasattr(self, "exp_model"): return
        def find_item_recursively(parent_item):
            for r in range(parent_item.rowCount()):
                item = parent_item.child(r, 0)
                if item and item.data(role=Qt.ItemDataRole.UserRole)["name"] == name_to_select:
                    return item
                found_item = find_item_recursively(item)
                if found_item: return found_item
            return None
        item_to_select = find_item_recursively(self.exp_model.invisibleRootItem())
        if item_to_select:
            self.exp_tree.selectionModel().select(
                item_to_select.index(),
                QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
            )
            self.exp_tree.scrollTo(item_to_select.index())

    def archive_selected_experiments(self):
        exp_names = self.get_selected_experiment_names()
        if not exp_names: return
        reply = QMessageBox.question(
            self, f"Archive {len(exp_names)} Experiments",
            f"Are you sure you want to archive {len(exp_names)} experiments?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for name in exp_names: self.manager.archive_experiment(name)
            self.refresh_ui()

    def open_archive_manager(self):
        dialog = ArchiveManagerDialog(self.manager, self)
        dialog.exec()
        self.refresh_ui()

    def rename_selected_experiment(self):
        exp_names = self.get_selected_experiment_names()
        if len(exp_names) != 1: return
        exp_name = exp_names[0]
        new_name, ok = QInputDialog.getText(self, "Rename Experiment", f"Enter new name for '{exp_name}':", QLineEdit.EchoMode.Normal, exp_name)
        if ok and new_name:
            success, message = self.manager.rename_experiment(exp_name, new_name)
            if success:
                self.refresh_ui()
                self.select_experiment_by_name(new_name)
            else:
                QMessageBox.warning(self, "Error", message)

    def select_parent_experiment(self):
        exp_name = self.get_selected_experiment_name()
        if not exp_name: return
        config, err = self.manager.load_experiment_config(exp_name)
        if err: return
        parent_name = config.get("parent_experiment")
        if parent_name: self.select_experiment_by_name(parent_name)

    def clear_comparison(self):
        self.comparison_list.clear()
        self.update_selected_experiment_display()

    def compare_selected(self):
        selected_names = self.get_selected_experiment_names()
        for name in selected_names:
            if name not in self.comparison_list:
                self.comparison_list.append(name)
        self.update_selected_experiment_display()

    def refresh_ui(self):
        self.manager.update_log_files()
        if self.log_deck.isVisible():
            for exp_name in self.log_deck.active_logs:
                log_content = self.manager.get_log_contents(exp_name)
                self.log_deck.update_log_content(exp_name, log_content)

        self.populate_experiment_list()
        self.update_selected_experiment_display()
        self.trajectory_view.draw_graph()
        self.scatter_plot_view.update_plot()
        if hasattr(self, "challenge_view"): self.challenge_view.refresh()

    def update_selected_experiment_display(self):
        self.plot_widget.clear()
        self.config_display.clear()
        self.diff_table.setRowCount(0)
        self.diff_table.setColumnCount(0)

        selected_names = self.get_selected_experiment_names()
        num_selected = len(selected_names)
        self.clear_comparison_button.setEnabled(len(self.comparison_list) > 0)
        self.compare_button.setEnabled(num_selected > 0)

        if self.comparison_list:
            self._display_comparison()
        elif num_selected == 1:
            exp_name = selected_names[0]
            exp_data = self.manager.get_experiment_graph()["nodes"].get(exp_name)
            if exp_data:
                self.trajectory_view.highlight_node(exp_name)
                self.scatter_plot_view.highlight_point(exp_name)
                if exp_data.get("type") == "Search":
                    self._display_search_summary(exp_name, exp_data)
                else:
                    self._display_single_experiment(exp_name, exp_data)
        else:
            self.plot_widget.setTitle("No experiment selected")
            self.config_display.clear()
            self.diff_table.setVisible(False)
            self.trajectory_view.clear_highlight()
            self.scatter_plot_view.clear_highlight()

    def _display_search_summary(self, exp_name, exp_data):
        self.config_display.setVisible(True)
        self.plot_widget.setVisible(False)
        self.diff_table.setVisible(True)
        self.plot_widget.setTitle(f"Search Summary: {exp_name}")
        graph = self.manager.get_experiment_graph()
        children_names = [edge[1] for edge in graph["edges"] if edge[0] == exp_name]
        trials = [graph["nodes"][name] for name in children_names if name in graph["nodes"]]
        if not trials:
            self.config_display.setText("No trials found for this search yet.")
            self.diff_table.setVisible(False)
            return
        best_trial, best_loss = None, float("inf")
        for trial in trials:
            try:
                loss = float(trial["final_loss"])
                if loss < best_loss: best_loss, best_trial = loss, trial
            except (ValueError, TypeError): continue
        if best_trial:
            best_trial_config, _ = self.manager.load_experiment_config(best_trial["name"])
            summary_text = (f"<b>Best Trial:</b> {best_trial['name']}<br>"
                            f"<b>Best Test Loss:</b> {best_trial['final_loss']}<br><br>"
                            f"<b>Best Parameters:</b><br>")
            search_config, _ = self.manager.load_experiment_config(exp_name)
            tuned_params = search_config.get("search", {}).get("params", {}).keys()
            flat_config = self._flatten_dict(best_trial_config)
            for p in tuned_params:
                summary_text += f"- {p}: {flat_config.get(p, 'N/A')}<br>"
            self.config_display.setHtml(summary_text)
        else:
            self.config_display.setText("No completed trials with valid loss values yet.")
        search_config, _ = self.manager.load_experiment_config(exp_name)
        tuned_params = list(search_config.get("search", {}).get("params", {}).keys())
        headers = ["Trial Name", "Status", "Final Loss"] + tuned_params
        self.diff_table.setColumnCount(len(headers))
        self.diff_table.setHorizontalHeaderLabels(headers)
        self.diff_table.setRowCount(len(trials))
        for row, trial in enumerate(trials):
            self.diff_table.setItem(row, 0, QTableWidgetItem(trial["name"]))
            self.diff_table.setItem(row, 1, QTableWidgetItem(trial["status"]))
            self.diff_table.setItem(row, 2, QTableWidgetItem(str(trial["final_loss"])))
            trial_config, _ = self.manager.load_experiment_config(trial["name"])
            if trial_config:
                flat_config = self._flatten_dict(trial_config)
                for i, key in enumerate(tuned_params):
                    val = flat_config.get(key, "N/A")
                    self.diff_table.setItem(row, 3 + i, QTableWidgetItem(str(val)))
        self.diff_table.resizeColumnsToContents()

    def _display_single_experiment(self, exp_name, exp_data):
        self.diff_table.setVisible(False)
        self.config_display.setVisible(True)
        self.plot_widget.setVisible(True)
        self._update_metric_selector([exp_name])
        self.current_train_loss, self.current_test_loss = [], []
        results_data, res_error = self.manager.load_experiment_results(exp_name)
        if res_error:
            QMessageBox.warning(self, "Result file error", res_error)
        elif results_data:
            metric_base_name = self.metric_selector.currentText() or "loss"
            train_metric, test_metric = f"train_{metric_base_name}", f"test_{metric_base_name}"
            self.current_train_loss = results_data.get(train_metric, [])
            self.current_test_loss = results_data.get(test_metric, [])
            self.plot_widget.setTitle(f"Learning Curves: {exp_name} ({metric_base_name})")
            if self.current_train_loss: self.plot_widget.plot(self.current_train_loss, pen="b", name=f"Train {metric_base_name}")
            if self.current_test_loss: self.plot_widget.plot(self.current_test_loss, pen="r", name=f"Test {metric_base_name}")
            self.plot_widget.addLegend()
        else:
            self.plot_widget.setTitle(f"No results available for: {exp_name}")
        config_data, conf_error = self.manager.load_experiment_config(exp_name)
        if conf_error: self.config_display.setText(conf_error)
        elif config_data: self.config_display.setText(json.dumps(config_data, indent=4))

    def _display_comparison(self):
        self.config_display.setVisible(False)
        self.diff_table.setVisible(True)
        self.plot_widget.setVisible(True)
        self._update_diff_table(self.comparison_list)
        self._update_metric_selector(self.comparison_list)
        self.plot_widget.addLegend()
        self.plot_widget.setTitle(f"Comparing {len(self.comparison_list)} experiments")
        colors = ["b", "r", "g", "c", "m", "y", "w"]
        metric_base_name = self.metric_selector.currentText()
        if not metric_base_name: return
        train_metric, test_metric = f"train_{metric_base_name}", f"test_{metric_base_name}"
        for i, exp_name in enumerate(self.comparison_list):
            results_data, _ = self.manager.load_experiment_results(exp_name)
            if results_data:
                train_vals, test_vals = results_data.get(train_metric, []), results_data.get(test_metric, [])
                color = colors[i % len(colors)]
                if train_vals: self.plot_widget.plot(train_vals, pen=pg.mkPen(color, style=Qt.PenStyle.SolidLine), name=f"{exp_name} Train")
                if test_vals: self.plot_widget.plot(test_vals, pen=pg.mkPen(color, style=Qt.PenStyle.DashLine), name=f"{exp_name} Test")

    def _flatten_dict(self, d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict): items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else: items.append((new_key, v))
        return dict(items)

    def _update_diff_table(self, exp_names):
        self.diff_table.setRowCount(0)
        if len(exp_names) < 2: self.diff_table.setVisible(False); return
        configs = []
        for name in exp_names:
            config_data, _ = self.manager.load_experiment_config(name)
            if config_data: configs.append(self._flatten_dict(config_data))
        if len(configs) < 2: return
        all_keys = set().union(*(c.keys() for c in configs))
        diff_keys = {k for k in all_keys if len({c.get(k) for c in configs}) > 1}
        self.diff_table.setColumnCount(len(exp_names) + 1)
        self.diff_table.setHorizontalHeaderLabels(["Parameter"] + exp_names)
        sorted_diff_keys = sorted(list(diff_keys))
        self.diff_table.setRowCount(len(sorted_diff_keys))
        for row, key in enumerate(sorted_diff_keys):
            self.diff_table.setItem(row, 0, QTableWidgetItem(key))
            for col, config in enumerate(configs):
                self.diff_table.setItem(row, col + 1, QTableWidgetItem(str(config.get(key, "N/A"))))
        self.diff_table.resizeColumnsToContents()

    def _update_metric_selector(self, exp_names):
        self.metric_selector.blockSignals(True)
        self.metric_selector.clear()
        if not exp_names: self.metric_selector.blockSignals(False); return
        common_metrics = None
        for name in exp_names:
            results, _ = self.manager.load_experiment_results(name)
            if results:
                metrics = {k.replace("train_", "").replace("test_", "") for k, v in results.items() if isinstance(v, list)}
                if common_metrics is None: common_metrics = metrics
                else: common_metrics.intersection_update(metrics)
        if common_metrics:
            valid_metrics = [m for m in common_metrics if any(f"train_{m}" in r or f"test_{m}" in r for r, _ in (self.manager.load_experiment_results(n) for n in exp_names))]
            self.metric_selector.addItems(sorted(list(set(valid_metrics))))
        self.metric_selector.blockSignals(False)

    def _on_plot_hover(self, event):
        pos = event[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            if 0 <= x < len(self.current_train_loss):
                index = int(round(x))
                if 0 <= index < len(self.current_train_loss) and 0 <= index < len(self.current_test_loss):
                    train_val, test_val = self.current_train_loss[index], self.current_test_loss[index]
                    self.plot_label.setText(f"Epoch: {index}\nTrain: {train_val:.4f}\nTest: {test_val:.4f}")
                    self.plot_label.setPos(x, y)
                    self.v_line.setPos(x); self.h_line.setPos(y)
                    self.v_line.show(); self.h_line.show(); self.plot_label.show()
                    return
        self.v_line.hide(); self.h_line.hide(); self.plot_label.hide()

    def stop_selected_experiments(self):
        exp_names = self.get_selected_experiment_names()
        if not exp_names: return
        for name in exp_names: self.manager.stop_experiment(name, force=False)
        QTimer.singleShot(500, self.refresh_ui)

    def force_stop_selected_experiments(self):
        exp_names = self.get_selected_experiment_names()
        if not exp_names: return
        for name in exp_names: self.manager.stop_experiment(name, force=True)
        QTimer.singleShot(500, self.refresh_ui)

    def view_selected_logs(self):
        exp_name = self.get_selected_experiment_name()
        if not exp_name: return
        log_content = self.manager.get_log_contents(exp_name)
        self.log_deck.add_log_view(exp_name, log_content)
        self.log_deck.show()
