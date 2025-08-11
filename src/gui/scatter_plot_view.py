import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QMenu,
    QGraphicsLineItem,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QAction

class CustomPlotWidget(pg.PlotWidget):
    """
    A custom PlotWidget that handles context menus for the scatter plot view.
    """
    def __init__(self, parent_view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_view = parent_view

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        filter_action = QAction("Filter selected in Experiments Tab", menu)
        clear_action = QAction("Clear selection", menu)

        filter_action.setEnabled(len(self.parent_view.selected_points) > 0)

        filter_action.triggered.connect(self.parent_view._emit_filter_signal)
        clear_action.triggered.connect(self.parent_view._clear_selection)

        menu.addAction(filter_action)
        menu.addAction(clear_action)
        menu.exec(event.globalPos())

class ScatterPlotView(QWidget):
    """
    A widget for visualizing experiments as a 2D scatter plot.
    """
    experiment_selected = pyqtSignal(str)
    experiments_selected_for_filtering = pyqtSignal(list)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.selected_points = []
        self.all_points_data = [] # To store the raw data for all points

        self._init_ui()
        self._populate_selectors()

        self.scatter_plot.sigClicked.connect(self._on_point_clicked)
        self.roi.sigRegionChangeFinished.connect(self._on_roi_changed)
        self.x_axis_selector.currentTextChanged.connect(self.update_plot)
        self.y_axis_selector.currentTextChanged.connect(self.update_plot)
        self.color_selector.currentTextChanged.connect(self.update_plot)


    def _on_point_clicked(self, plot, points):
        if points:
            exp_name = points[0].data()
            self.experiment_selected.emit(exp_name)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Controls ---
        controls_layout = QHBoxLayout()
        self.x_axis_selector = QComboBox()
        self.y_axis_selector = QComboBox()
        self.color_selector = QComboBox()

        controls_layout.addWidget(QLabel("X-Axis:"))
        controls_layout.addWidget(self.x_axis_selector)
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Y-Axis:"))
        controls_layout.addWidget(self.y_axis_selector)
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Color:"))
        controls_layout.addWidget(self.color_selector)

        # --- Plot ---
        self.plot_widget = CustomPlotWidget(self)
        self.scatter_plot = pg.ScatterPlotItem(
            hoverable=True, hoverPen=pg.mkPen("r", width=2)
        )
        self.plot_widget.addItem(self.scatter_plot)

        # --- ROI for selection ---
        self.roi = pg.RectROI([0, 0], [1, 1], pen=(0, 9))
        self.plot_widget.addItem(self.roi)

        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.plot_widget)

    def _on_roi_changed(self):
        """
        Handles the selection of points when the ROI is moved or resized.
        """
        if not self.all_points_data:
            return

        # Get the bounding rectangle of the ROI in the plot's view coordinates
        roi_rect = self.roi.viewRect()

        self.selected_points = []
        for point in self.all_points_data:
            point_pos = point['pos']
            if roi_rect.contains(*point_pos):
                self.selected_points.append(point)

        self._update_selection_highlighting()

    def _update_selection_highlighting(self):
        """
        Updates the visual appearance of points based on selection.
        """
        selected_names = [p['data'] for p in self.selected_points]

        for p in self.all_points_data:
            if p['data'] in selected_names:
                p['pen'] = pg.mkPen('w', width=2)
                p['size'] = 12
            else:
                p['pen'] = None # Default pen
                p['size'] = 10

        self.scatter_plot.setData(self.all_points_data)

    def _clear_selection(self):
        self.selected_points = []
        self._update_selection_highlighting()

    def _emit_filter_signal(self):
        names = [p['data'] for p in self.selected_points]
        if names:
            self.experiments_selected_for_filtering.emit(names)

    def _populate_selectors(self):
        metrics = self.manager.get_plottable_metrics()
        self.x_axis_selector.addItems(metrics)
        self.y_axis_selector.addItems(metrics)
        self.color_selector.addItems(metrics)

    def _get_metric_value(self, metric_key, exp_data):
        """
        Retrieves a metric value from experiment data, handling nested keys.
        """
        # A helper to access nested dictionary keys
        def get_nested(_dict, keys):
            for key in keys:
                if isinstance(_dict, dict):
                    _dict = _dict.get(key)
                else:
                    return None
            return _dict

        try:
            # Handle top-level results keys
            if metric_key.startswith("results."):
                key = metric_key.replace("results.", "")
                val = exp_data[key]
                return float(val) if val != "N/A" else None

            # Handle nested config keys
            elif metric_key.startswith("config."):
                keys = metric_key.replace("config.", "").split('.')
                config, _ = self.manager.load_experiment_config(exp_data["name"])
                if config:
                    val = get_nested(config, keys)
                    return float(val) if val is not None else None
            return None
        except (ValueError, TypeError, KeyError):
            return None

    def update_plot(self):
        # This can be slow if there are many experiments, so we don't clear the whole plot,
        # we just update the data of the existing scatter plot item.
        self.plot_widget.setLabel("bottom", self.x_axis_selector.currentText())
        self.plot_widget.setLabel("left", self.y_axis_selector.currentText())

        experiments = self.manager.get_experiments_data()
        self.all_points_data = []
        color_values = []

        x_metric = self.x_axis_selector.currentText()
        y_metric = self.y_axis_selector.currentText()
        color_metric = self.color_selector.currentText()

        for exp in experiments:
            x = self._get_metric_value(x_metric, exp)
            y = self._get_metric_value(y_metric, exp)
            color_val = self._get_metric_value(color_metric, exp)

            if x is not None and y is not None:
                self.all_points_data.append({
                    "pos": (x, y),
                    "data": exp["name"],
                    "brush": pg.mkBrush("b"), # Default brush
                    "pen": None,
                    "size": 10
                })
                if color_val is not None:
                    color_values.append(color_val)

        if self.all_points_data and color_values and len(color_values) == len(self.all_points_data):
            min_c = min(color_values)
            max_c = max(color_values)
            cmap = pg.ColorMap(
                pos=[min_c, max_c],
                color=[(0, 0, 255, 255), (255, 0, 0, 255)] # Blue to Red
            )
            for i, p in enumerate(self.all_points_data):
                p["brush"] = cmap.map(color_values[i], 'qcolor')

        self.scatter_plot.setData(self.all_points_data)
        self._update_selection_highlighting()

        # Remove old race lines and color bar before adding new ones
        items_to_remove = [item for item in self.plot_widget.items() if isinstance(item, (QGraphicsLineItem, pg.GradientLegend))]
        for item in items_to_remove:
            self.plot_widget.removeItem(item)

        # Add a color bar
        if color_values:
            grad_legend = pg.GradientLegend((20, 150), (-10, -30))
            grad_legend.setLabels({f"{min_c:.2f}": 0, f"{max_c:.2f}": 1})
            grad_legend.setColorMap(cmap)
            self.plot_widget.addItem(grad_legend)

        # --- Draw lines for races ---
        races = {}
        for i, p in enumerate(self.all_points_data):
            exp_name = p["data"]
            exp_data = next((e for e in experiments if e["name"] == exp_name), None)
            if exp_data and exp_data.get("race_id") and exp_data.get("race_id") != "N/A":
                race_id = exp_data["race_id"]
                if race_id not in races:
                    races[race_id] = []
                races[race_id].append({"point": p, "data": exp_data})

        for race_id, members in races.items():
            if len(members) > 1:
                challenger = next((m for m in members if not m["data"].get("is_baseline_for")), None)
                if not challenger: continue

                for baseline in members:
                    if baseline == challenger: continue
                    line = QGraphicsLineItem(
                        challenger["point"]["pos"][0],
                        challenger["point"]["pos"][1],
                        baseline["point"]["pos"][0],
                        baseline["point"]["pos"][1],
                    )
                    line.setPen(pg.mkPen('grey', width=1, style=Qt.PenStyle.DashLine))
                    self.plot_widget.addItem(line)
