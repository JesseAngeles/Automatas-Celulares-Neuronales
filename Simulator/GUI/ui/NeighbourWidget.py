# ui.NeighbourWidget

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QMouseEvent
from PyQt5.QtCore import pyqtSignal

class NeighbourWidget(QWidget):
    cell_toggled = pyqtSignal(int, int, int)    

    def __init__(self, grid_size: int =  7):
        super().__init__()

        self.grid_size = grid_size
        self.grid = [[0] * grid_size for _ in range(grid_size)]

        # self.setMinimumSize(
        #     grid_size * self.cell_size,
        #     grid_size * self.cell_size
        # )

    @property
    def cell_size(self):
        return min(self.width(), self.height()) // self.grid_size

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
            new_val = 1 - self.grid[y][x]
            self.grid[y][x] = new_val
            self.cell_toggled.emit(x, y, new_val)
            self.update()

    def update_grid(self, grid):
        self.grid = grid
        self.update()