from PyQt6.QtWidgets import (
    QWidget,
    QGraphicsView,
    QGraphicsScene,
    QVBoxLayout,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QGraphicsLineItem,
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QBrush, QPen, QColor


class TrajectoryView(QWidget):
    """
    A widget to display the experiment trajectory as a graph.
    """

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(self.view.RenderHint.Antialiasing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self.draw_graph()

    def draw_graph(self):
        self.scene.clear()
        graph_data = self.manager.get_experiment_graph()
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        roots = graph_data["roots"]

        if not nodes:
            return

        # This is a placeholder for a real layout algorithm
        positions = self._layout_graph(graph_data)

        # Draw edges
        for parent, child in edges:
            if parent in positions and child in positions:
                line = QGraphicsLineItem(
                    positions[parent].x(),
                    positions[parent].y(),
                    positions[child].x(),
                    positions[child].y(),
                )
                self.scene.addItem(line)

        # Draw nodes
        for name, pos in positions.items():
            rect = QGraphicsRectItem(pos.x() - 50, pos.y() - 20, 100, 40)
            rect.setBrush(QBrush(QColor("lightblue")))
            self.scene.addItem(rect)

            text = QGraphicsTextItem(name, parent=rect)
            text.setDefaultTextColor(QColor("black"))
            text.setPos(pos.x() - 50, pos.y() - 10)


    def _layout_graph(self, graph_data):
        # NOTE: This is a very basic layout algorithm.
        # A real implementation would be more sophisticated.
        from PyQt6.QtCore import QPointF

        positions = {}
        y = 0
        for root in graph_data["roots"]:
            positions[root] = QPointF(0, y)
            y += 60

        # Extremely naive layout for children
        for parent, child in graph_data["edges"]:
            if parent in positions:
                positions[child] = QPointF(positions[parent].x() + 150, positions[parent].y())

        return positions
