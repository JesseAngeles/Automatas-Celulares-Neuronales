# ui.SimulationWidget

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QMouseEvent
from PyQt5.QtCore import pyqtSignal, Qt

class SimulationWidget(QWidget):
    cell_toggled = pyqtSignal(int, int, int)

    def __init__(self, grid_size: int = 100):
        super().__init__()

        self.grid_size = grid_size
        self.grid = [[0] * grid_size for _ in range(grid_size)]
        self.setMinimumSize(200, 200)
        self.setMinimumSize(
            grid_size * self.cell_size,
            grid_size * self.cell_size
        )

        self._active_state = 1
        self._state_colors: list[QColor] = [QColor(255, 255, 255), QColor(0, 0, 0)]
        self._last_painted: tuple[int, int] | None = None

    @property
    def cell_size(self):
        return min(self.width(), self.height()) // self.grid_size

    def paintEvent(self, _):
        painter = QPainter(self)
        n = len(self._state_colors)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                s = self.grid[y][x]
                color = self._state_colors[s] if s < n else QColor(128, 128, 128)
                painter.setBrush(color)
                painter.drawRect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

    def mousePressEvent(self, event: QMouseEvent):
        self._last_painted = None
        self._paint_at(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self._paint_at(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._last_painted = None

    def _paint_at(self, event: QMouseEvent):
        x = event.pos().x() // self.cell_size
        y = event.pos().y() // self.cell_size
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return
        if (x, y) == self._last_painted:
            return
        self._last_painted = (x, y)
        current = self.grid[y][x]
        new_val = 0 if current == self._active_state else self._active_state
        self.grid[y][x] = new_val
        self.cell_toggled.emit(x, y, new_val)
        self.update()

    def set_active_state(self, state: int):
        self._active_state = state

    def set_state_colors(self, colors: list[QColor]):
        self._state_colors = list(colors)
        self.update()

    def update_grid(self, grid):
        self.grid = grid
        self.update()
