# ui.SimulationWidget

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QMouseEvent

class SimulationWidget(QWidget):
    def __init__(self, grid_size: int =  50):
        super().__init__()

        self.grid_size = grid_size
        self.cell_size = self.width() // grid_size

        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

        self.setMinimumSize(
            grid_size * self.cell_size,
            grid_size * self.cell_size
        )

    def paintEvent(self, _):
        painter = QPainter(self)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y][x] == 1:
                    painter.setBrush(QColor(0,0,0))
                else:
                    painter.setBrush(QColor(255,255,255))

                painter.drawRect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

    def mousePressEvent(self, event: QMouseEvent):
        x = event.pos().x() // self.cell_size
        y = event.pos().y() // self.cell_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y][x] = 1 - self.grid[y][x]
            self.update()

    def update_grid(self, grid):
        self.grid = grid
        self.update()