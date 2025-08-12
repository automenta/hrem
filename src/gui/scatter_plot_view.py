import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QMenu,
    QGraphicsLineItem,
    QPushButton,
)
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        menu = self.getPlotItem().vb.getMenu(event)
        menu.addSeparator()

        filter_action = QAction("Filter selected in Experiments Tab", menu)
        clear_action = QAction("Clear selection", menu)

        filter_action.setEnabled(len(self.parent_view.selected_points) > 0)

        filter_action.triggered.connect(self.parent_view._emit_filter_signal)
        clear_action.triggered.connect(self.parent_view._clear_selection)

        menu.addAction(filter_action)
        menu.addAction(clear_action)
        # The menu is executed by the default context menu handler, so we don't need to call exec.
        # However, since we are overriding the event, we need to handle it.
        # By getting the menu and adding actions, we have modified the menu that will be
        # displayed by the default event handler. We don't need to call exec_ ourselves.
        # We do, however, need to call the superclass's method to ensure the event is processed.
        # Since PlotWidget itself doesn't implement contextMenuEvent, we pass it to the PlotItem.
        self.getPlotItem().contextMenuEvent(event)

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
        self.run_pca_button.clicked.connect(self._run_pca)
        self.color_selector.currentTextChanged.connect(self.update_plot_styles)
        self.size_selector.currentTextChanged.connect(self.update_plot_styles)

    def highlight_point(self, name_to_highlight: str):
        """
        Highlights a single point on the scatter plot corresponding to the given name.
        """
        self.selected_points = []
        for point in self.all_points_data:
            if point['data'] == name_to_highlight:
                self.selected_points.append(point)
                break  # Assuming names are unique
        self._update_selection_highlighting()
        # Also, we might want to move the view to center on the point
        if self.selected_points:
            pos = self.selected_points[0]['pos']
            self.plot_widget.getViewBox().setXRange(pos[0] - 1, pos[0] + 1, padding=0.1)
            self.plot_widget.getViewBox().setYRange(pos[1] - 1, pos[1] + 1, padding=0.1)


    def clear_highlight(self):
        """
        Clears the current selection and highlighting.
        """
        self._clear_selection()

    def _on_point_clicked(self, plot, points):
        if points:
            exp_name = points[0].data()
            self.experiment_selected.emit(exp_name)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Controls ---
        controls_layout = QHBoxLayout()
        self.run_pca_button = QPushButton("Run PCA")
        self.color_selector = QComboBox()
        self.size_selector = QComboBox()

        controls_layout.addWidget(self.run_pca_button)
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Color:"))
        controls_layout.addWidget(self.color_selector)
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Size:"))
        controls_layout.addWidget(self.size_selector)


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

        # Create a temporary copy to modify, to avoid issues withsetData
        temp_points_data = [p.copy() for p in self.all_points_data]

        for p in temp_points_data:
            if p['data'] in selected_names:
                p['pen'] = pg.mkPen('w', width=2)
                p['size'] = 15 # Make highlighted points larger
            else:
                # Reset to original style from update_plot_styles
                # This part is tricky because the original style is dynamic.
                # We need to re-apply the styling logic.
                # For simplicity now, we just reset to a default.
                # A better implementation would store the original style.
                p['pen'] = None
                p['size'] = 10


        self.scatter_plot.setData(spots=temp_points_data)

    def _clear_selection(self):
        self.selected_points = []
        self._update_selection_highlighting()

    def _emit_filter_signal(self):
        names = [p['data'] for p in self.selected_points]
        if names:
            self.experiments_selected_for_filtering.emit(names)

    def _populate_selectors(self):
        metrics = self.manager.get_plottable_metrics()
        self.color_selector.addItems(["default"] + metrics)
        self.size_selector.addItems(["default"] + metrics)

    def _get_metric_value(self, metric_key, exp_data):
        """
        Retrieves a metric value from experiment data, handling nested keys.
        """
        if metric_key == "default":
            return 1.0 # Return a constant value

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

    def _run_pca(self):
        """
        Runs PCA on the available experiment data and updates the plot.
        """
        self.update_plot()

    def update_plot(self):
        """
        Runs PCA on the available experiment data and updates the plot.
        """
        experiments = self.manager.get_experiments_data()
        metrics = self.manager.get_plottable_metrics()

        # --- Prepare data for PCA ---
        pca_data = []
        valid_experiments = []
        for exp in experiments:
            feature_vector = [self._get_metric_value(m, exp) for m in metrics]
            if all(v is not None for v in feature_vector):
                pca_data.append(feature_vector)
                valid_experiments.append(exp)

        if len(pca_data) < 2:
            print("Not enough data for PCA.")
            self.all_points_data = []
            self.scatter_plot.clear()
            return

        df = pd.DataFrame(pca_data, columns=metrics)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)

        # --- Store data for plotting ---
        self.all_points_data = []
        for i, exp in enumerate(valid_experiments):
            self.all_points_data.append({
                "pos": (principal_components[i, 0], principal_components[i, 1]),
                "data": exp["name"],
                "exp_data": exp, # Store original data
            })

        self.plot_widget.setLabel("bottom", "Principal Component 1")
        self.plot_widget.setLabel("left", "Principal Component 2")
        self.update_plot_styles()


    def update_plot_styles(self):
        """
        Updates the color and size of the points on the scatter plot.
        """
        if not self.all_points_data:
            return

        color_metric = self.color_selector.currentText()
        size_metric = self.size_selector.currentText()

        color_values = [self._get_metric_value(color_metric, p["exp_data"]) for p in self.all_points_data]
        size_values = [self._get_metric_value(size_metric, p["exp_data"]) for p in self.all_points_data]

        # Filter out None values for robust min/max calculation
        valid_colors = [v for v in color_values if v is not None]
        valid_sizes = [v for v in size_values if v is not None]

        # --- Assign brushes based on color metric ---
        if valid_colors and color_metric != "default":
            min_c, max_c = min(valid_colors), max(valid_colors)
            cmap = pg.ColorMap(pos=[min_c, max_c], color=[(0, 0, 255, 255), (255, 255, 0, 255)])
            for i, p in enumerate(self.all_points_data):
                p["brush"] = cmap.map(color_values[i], 'qcolor') if color_values[i] is not None else pg.mkBrush("gray")
        else:
            for p in self.all_points_data:
                p["brush"] = pg.mkBrush("blue")

        # --- Assign sizes based on size metric ---
        if valid_sizes and size_metric != "default":
            min_s, max_s = min(valid_sizes), max(valid_sizes)
            if max_s == min_s: max_s += 1e-9 # Avoid division by zero
            for i, p in enumerate(self.all_points_data):
                if size_values[i] is not None:
                    # Normalize size between 5 and 20
                    p["size"] = 5 + 15 * ((size_values[i] - min_s) / (max_s - min_s))
                else:
                    p["size"] = 10
        else:
            for p in self.all_points_data:
                p["size"] = 10

        self._update_selection_highlighting()

        # --- Clear old legends and lines ---
        items_to_remove = [item for item in self.plot_widget.items() if isinstance(item, (QGraphicsLineItem, pg.GradientLegend))]
        for item in items_to_remove:
            self.plot_widget.removeItem(item)

        # --- Add new color bar ---
        if valid_colors and color_metric != "default":
            min_c, max_c = min(valid_colors), max(valid_colors)
            cmap = pg.ColorMap(pos=[min_c, max_c], color=[(0, 0, 255, 255), (255, 255, 0, 255)])
            grad_legend = pg.GradientLegend((20, 150), (-10, -30))
            grad_legend.setLabels({f"{min_c:.2g}": 0, f"{max_c:.2g}": 1})
            grad_legend.setColorMap(cmap)
            self.plot_widget.addItem(grad_legend)

        # --- Draw lines for races ---
        races = {}
        for i, p in enumerate(self.all_points_data):
            exp_data = p["exp_data"]
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
