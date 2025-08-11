import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal

class ScatterPlotView(QWidget):
    """
    A widget for visualizing experiments as a 2D scatter plot.
    """
    experiment_selected = pyqtSignal(str)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager

        self._init_ui()
        self._populate_selectors()

        self.scatter_plot.sigClicked.connect(self._on_point_clicked)

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
        self.plot_widget = pg.PlotWidget()
        self.scatter_plot = pg.ScatterPlotItem()
        self.plot_widget.addItem(self.scatter_plot)

        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.plot_widget)

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
                return float(exp_data[key])
            # Handle nested config keys
            elif metric_key.startswith("config."):
                keys = metric_key.replace("config.", "").split('.')
                config, _ = self.manager.load_experiment_config(exp_data["name"])
                if config:
                    return float(get_nested(config, keys))
            return None
        except (ValueError, TypeError, KeyError):
            return None

    def update_plot(self):
        self.plot_widget.clear()
        self.scatter_plot = pg.ScatterPlotItem(
            hoverable=True, hoverPen=pg.mkPen("r", width=2)
        )
        self.plot_widget.addItem(self.scatter_plot)

        x_metric = self.x_axis_selector.currentText()
        y_metric = self.y_axis_selector.currentText()
        color_metric = self.color_selector.currentText()

        self.plot_widget.setLabel("bottom", x_metric)
        self.plot_widget.setLabel("left", y_metric)

        experiments = self.manager.get_experiments_data()
        points = []

        color_values = []
        points = []

        for exp in experiments:
            x = self._get_metric_value(x_metric, exp)
            y = self._get_metric_value(y_metric, exp)
            color_val = self._get_metric_value(color_metric, exp)

            if x is not None and y is not None:
                points.append({
                    "pos": (x, y),
                    "data": exp["name"],
                    "brush": pg.mkBrush("b"), # Default brush
                })
                if color_val is not None:
                    color_values.append(color_val)

        if points and color_values and len(color_values) == len(points):
            min_c = min(color_values)
            max_c = max(color_values)
            cmap = pg.ColorMap(
                pos=[min_c, max_c],
                color=[(0, 0, 255, 255), (255, 0, 0, 255)] # Blue to Red
            )
            for i, p in enumerate(points):
                p["brush"] = cmap.map(color_values[i], 'qcolor')

            # Add a color bar
            grad_legend = pg.GradientLegend((20, 150), (-10, -30))
            grad_legend.setLabels({f"{min_c:.2f}": 0, f"{max_c:.2f}": 1})
            grad_legend.setColorMap(cmap)
            self.plot_widget.addItem(grad_legend)

        self.scatter_plot.setData(points)

        # --- Draw lines for races ---
        races = {}
        for i, p in enumerate(points):
            exp_name = p["data"]
            exp_data = next((e for e in experiments if e["name"] == exp_name), None)
            if exp_data and exp_data.get("race_id") != "N/A":
                race_id = exp_data["race_id"]
                if race_id not in races:
                    races[race_id] = []
                races[race_id].append({"point": p, "data": exp_data})

        for race_id, members in races.items():
            if len(members) > 1:
                challenger = next((m for m in members if not m["data"]["is_baseline_for"]), None)
                if not challenger: continue

                for baseline in members:
                    if baseline == challenger: continue
                    line = pg.QtGui.QGraphicsLineItem(
                        challenger["point"]["pos"][0],
                        challenger["point"]["pos"][1],
                        baseline["point"]["pos"][0],
                        baseline["point"]["pos"][1],
                    )
                    line.setPen(pg.mkPen('grey', width=1, style=Qt.PenStyle.DashLine))
                    self.plot_widget.addItem(line)
