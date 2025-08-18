import pyqtgraph as pg
from PyQt6.QtGui import QColor

class CombinedRacePlot(pg.PlotWidget):
    """
    A specialized PlotWidget for displaying multiple race participants' metrics
    on a single graph for easy comparison.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.legend = None
        self.participant_pens = {}
        self.plot_items = {}

        # Use a dark theme for better aesthetics
        self.setBackground('w')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.getAxis('left').setTextPen('k')
        self.getAxis('bottom').setTextPen('k')

        self._setup_legend()

    def _setup_legend(self):
        """Initializes and styles the legend."""
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
                # Cycle through a list of nice, distinct colors for baselines
                baseline_colors = [
                    '#1f77b4',  # Muted Blue
                    '#ff7f0e',  # Safety Orange
                    '#d62728',  # Brick Red
                    '#9467bd',  # Muted Purple
                    '#8c564b',  # Chestnut Brown
                    '#e377c2',  # Raspberry Pink
                    '#7f7f7f',  # Middle Gray
                    '#bcbd22',  # Curry Yellow-Green
                    '#17becf',  # Blue-Cyan
                ]
                # Assign a color based on the number of baselines already present
                num_baselines = len([p for p in self.participant_pens if 'challenger' not in p])
                color = pg.mkColor(baseline_colors[num_baselines % len(baseline_colors)])

            self.participant_pens[participant_name] = pg.mkPen(color, width=3)

        return self.participant_pens[participant_name]

    def update_plot(self, participant_name: str, is_challenger: bool, metric_name: str, data: list):
        """
        Updates or adds a single line on the plot.

        Args:
            participant_name: The name of the experiment participant.
            is_challenger: Boolean indicating if this is the main challenger.
            metric_name: The name of the metric being plotted (e.g., 'test_loss').
            data: A list of numerical values for the y-axis.
        """
        if not data:
            return

        pen = self._get_pen(participant_name, is_challenger)
        plot_key = f"{participant_name}_{metric_name}"
        legend_name = f"{participant_name} ({metric_name})"

        if plot_key in self.plot_items:
            # Data exists, update it
            self.plot_items[plot_key].setData(data)
        else:
            # First time plotting this data, create a new plot item
            plot_item = self.plot(data, pen=pen, name=legend_name)
            self.plot_items[plot_key] = plot_item

    def clear_plots(self):
        """
        Clears all plotted lines and resets the stored plot items,
        but keeps the pen assignments for color consistency.
        """
        self.clear()
        self.plot_items = {}
        # Re-add the legend after clearing
        self._setup_legend()

    def set_title(self, title: str):
        """Sets the plot title with appropriate styling."""
        self.setTitle(title, color='k', size='14pt')

    def set_labels(self, left_label: str = 'Metric Value', bottom_label: str = 'Epoch'):
        """Sets the axis labels with appropriate styling."""
        self.setLabel('left', left_label)
        self.setLabel('bottom', bottom_label)
