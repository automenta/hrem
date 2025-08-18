import pyqtgraph as pg
from PyQt6.QtGui import QColor

class CombinedRacePlot(pg.PlotWidget):
    """
    A specialized PlotWidget for displaying multiple race participants' metrics
    on a single graph for easy comparison. Supports dual Y-axes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.legend = None
        self.participant_pens = {}
        self.plot_items_left = {}
        self.plot_items_right = {}

        self.setBackground('w')
        self.plot_item = self.getPlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.getAxis('left').setTextPen('k')
        self.plot_item.getAxis('bottom').setTextPen('k')

        # Create a new ViewBox for the right axis
        self.view_box_right = pg.ViewBox()
        self.plot_item.scene().addItem(self.view_box_right)
        self.plot_item.getAxis('right').linkToView(self.view_box_right)
        self.view_box_right.setXLink(self.plot_item)
        self.plot_item.getAxis('right').setTextPen('k') # Style the new axis

        self._setup_legend()

        # Update the bounds of the right view box whenever the main view box changes
        self.plot_item.vb.sigResized.connect(self._update_right_view_box)

    def _update_right_view_box(self):
        """Ensures the right-side viewbox is aligned with the main one."""
        self.view_box_right.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.view_box_right.linkedViewChanged(self.plot_item.vb, self.view_box_right.XAxis)

    def _setup_legend(self):
        """Initializes and styles the legend."""
        if self.legend:
            # Remove the old legend if it exists, to prevent duplicates
            self.legend.scene().removeItem(self.legend)
        self.legend = self.addLegend(labelTextColor='k', brush=QColor(255, 255, 255, 150))

    def _get_pen(self, participant_name: str, is_challenger: bool):
        """
        Gets a unique, consistent QPen for a given participant.
        A challenger is always green, baselines get assigned other colors.
        """
        if participant_name not in self.participant_pens:
            if is_challenger:
                color = pg.mkColor('#00A000') # Vibrant Green
            else:
                baseline_colors = [
                    '#1f77b4', '#ff7f0e', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                ]
                num_baselines = len([p for p in self.participant_pens if 'challenger' not in p])
                color = pg.mkColor(baseline_colors[num_baselines % len(baseline_colors)])

            self.participant_pens[participant_name] = pg.mkPen(color, width=3)
        return self.participant_pens[participant_name]

    def update_plot(self, participant_name: str, is_challenger: bool, metric_name: str, data: list, axis: str = 'left'):
        """
        Updates or adds a single line on the plot, assigning it to the correct axis.
        """
        if not data:
            return

        pen = self._get_pen(participant_name, is_challenger)
        plot_key = f"{participant_name}_{metric_name}"
        legend_name = f"{participant_name} ({metric_name})"

        plot_items_dict = self.plot_items_left if axis == 'left' else self.plot_items_right

        if plot_key in plot_items_dict:
            plot_items_dict[plot_key].setData(data)
        else:
            plot_item = pg.PlotDataItem(data, pen=pen, name=legend_name)
            if axis == 'right':
                self.view_box_right.addItem(plot_item)
            else:
                self.plot_item.addItem(plot_item)

            # Manually add to legend to ensure all items appear
            if self.legend:
                self.legend.addItem(plot_item, name=legend_name)

            plot_items_dict[plot_key] = plot_item

    def clear_plots(self):
        """
        Clears all plotted lines from both axes and resets the plot.
        """
        # Remove items from the legend and the plot
        if self.legend:
            for item in self.plot_items_left.values():
                self.legend.removeItem(item.name())
            for item in self.plot_items_right.values():
                self.legend.removeItem(item.name())

        # Remove items from their respective views
        for item in self.plot_items_left.values():
            self.plot_item.removeItem(item)
        for item in self.plot_items_right.values():
            self.view_box_right.removeItem(item)

        self.plot_items_left = {}
        self.plot_items_right = {}

        # Reset the legend completely
        self._setup_legend()

    def set_title(self, title: str):
        """Sets the plot title with appropriate styling."""
        self.setTitle(title, color='k', size='14pt')

    def set_labels(self, left_label: str = 'Metric Value', right_label: str = None, bottom_label: str = 'Epoch'):
        """Sets the axis labels with appropriate styling."""
        self.setLabel('left', left_label)
        self.setLabel('bottom', bottom_label)
        if right_label:
            self.getAxis('right').show()
            self.setLabel('right', right_label)
        else:
            self.getAxis('right').hide()
            self.view_box_right.clear() # Clear the view if not in use
