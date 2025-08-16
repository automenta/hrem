from PyQt6.QtWidgets import (
    QWidget,
    QGraphicsView,
    QGraphicsScene,
    QVBoxLayout,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QGraphicsLineItem,
)
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QBrush, QPen, QColor, QPainter

COLOR_NODE_DEFAULT = QColor("lightblue")
COLOR_NODE_HIGHLIGHT = QColor("yellow")


class TrajectoryView(QWidget):
    """
    A widget to display the experiment trajectory as a graph.
    """

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self.node_items = {}
        self.highlighted_node = None

        self.draw_graph()

    def highlight_node(self, name_to_highlight: str):
        """
        Highlights a specific node in the graph and de-highlights the previous one.
        """
        self.clear_highlight()

        if name_to_highlight in self.node_items:
            node_item = self.node_items[name_to_highlight]
            node_item.setBrush(QBrush(COLOR_NODE_HIGHLIGHT))
            self.highlighted_node = node_item
            self.view.ensureVisible(node_item)

    def clear_highlight(self):
        """
        Resets the currently highlighted node to its default color.
        """
        if self.highlighted_node:
            self.highlighted_node.setBrush(QBrush(COLOR_NODE_DEFAULT))
        self.highlighted_node = None

    def draw_graph(self):
        self.scene.clear()
        self.node_items.clear()
        self.highlighted_node = None

        graph_data = self.manager.get_experiment_graph()
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        if not nodes:
            return

        positions = self._layout_graph(graph_data)

        # Draw edges first (so they are in the background)
        for parent, child in edges:
            if parent in positions and child in positions:
                line = QGraphicsLineItem(
                    positions[parent].x(),
                    positions[parent].y(),
                    positions[child].x(),
                    positions[child].y(),
                )
                line.setPen(QPen(QColor("gray")))
                self.scene.addItem(line)

        # Draw nodes
        for name, pos in positions.items():
            rect = QGraphicsRectItem(pos.x() - 50, pos.y() - 20, 100, 40)
            rect.setBrush(QBrush(COLOR_NODE_DEFAULT))
            rect.setPen(QPen(QColor("black")))
            self.scene.addItem(rect)
            self.node_items[name] = rect  # Store reference to the node item

            text = QGraphicsTextItem(name, parent=rect)
            text.setDefaultTextColor(QColor("black"))
            # Center text
            text_rect = text.boundingRect()
            text.setPos(
                pos.x() - text_rect.width() / 2, pos.y() - text_rect.height() / 2
            )

    def _layout_graph(self, graph_data):
        # A simple hierarchical layout algorithm.
        positions = {}
        levels = {}

        # Find the level of each node
        def get_level(node_name):
            if node_name in levels:
                return levels[node_name]

            parents = [p for p, c in graph_data["edges"] if c == node_name]
            if not parents:
                levels[node_name] = 0
                return 0

            max_parent_level = max(get_level(p) for p in parents)
            levels[node_name] = max_parent_level + 1
            return levels[node_name]

        for node_name in graph_data["nodes"]:
            get_level(node_name)

        # Position nodes based on levels
        level_counts = {}
        max_level = 0
        for node_name, level in levels.items():
            if level not in level_counts:
                level_counts[level] = 0

            x = level * 180  # Horizontal spacing
            y = level_counts[level] * 70  # Vertical spacing
            positions[node_name] = QPointF(x, y)

            level_counts[level] += 1
            if level > max_level:
                max_level = level

        return positions
