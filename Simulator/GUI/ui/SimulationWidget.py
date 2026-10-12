# ui.SimulationWidget

import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QMouseEvent, QImage
from PyQt5.QtCore import pyqtSignal, Qt, QRectF


class SimulationWidget(QWidget):
    cell_toggled = pyqtSignal(int, int, int)

    def __init__(self, grid_size: int = 100):
        super().__init__()

        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        self._active_state = 1
        self._state_colors: list[QColor] = [QColor(255, 255, 255), QColor(0, 0, 0)]
        self._color_table: list[int] = self._build_color_table()

        self._last_painted: tuple[int, int] | None = None
        self._cell_size_cached = 0

        self.setMinimumSize(200, 200)
        self.setCursor(Qt.CrossCursor)
        self.setStyleSheet(
            "SimulationWidget { background: #ffffff; border: 1px solid #e3e3e8; border-radius: 8px; }"
        )
        self._recompute_cell_size()

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------
    @property
    def cell_size(self):
        return self._cell_size_cached

    def _recompute_cell_size(self):
        if self.grid_size <= 0:
            self._cell_size_cached = 0
            return
        self._cell_size_cached = max(1, min(self.width(), self.height()) // self.grid_size)

    def resizeEvent(self, event):
        self._recompute_cell_size()
        super().resizeEvent(event)

    def _build_color_table(self) -> list[int]:
        # Tabla de 256 entradas (requerida por Format_Indexed8)
        default = QColor(128, 128, 128).rgb()
        n = len(self._state_colors)
        table = [0] * 256
        for i in range(256):
            table[i] = self._state_colors[i].rgb() if i < n else default
        return table

    def paintEvent(self, _):
        painter = QPainter(self)
        if self._cell_size_cached <= 0:
            return

        if not self.grid.flags["C_CONTIGUOUS"]:
            self.grid = np.ascontiguousarray(self.grid)

        h, w = self.grid.shape
        bytes_per_line = self.grid.strides[0]

        img = QImage(
            self.grid.data, w, h, bytes_per_line, QImage.Format_Indexed8
        )
        img.setColorTable(self._color_table)

        dst_w = w * self._cell_size_cached
        dst_h = h * self._cell_size_cached
        painter.drawImage(QRectF(0, 0, dst_w, dst_h), img)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        self._last_painted = None
        self._paint_at(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self._paint_at(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._last_painted = None

    def _paint_at(self, event: QMouseEvent):
        cs = self._cell_size_cached
        if cs <= 0:
            return
        x = event.pos().x() // cs
        y = event.pos().y() // cs
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return
        if (x, y) == self._last_painted:
            return
        self._last_painted = (x, y)

        current = int(self.grid[y, x])
        new_val = 0 if current == self._active_state else self._active_state
        self.grid[y, x] = new_val
        self.cell_toggled.emit(int(x), int(y), int(new_val))
        self.update()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def set_active_state(self, state: int):
        self._active_state = state

    def set_state_colors(self, colors):
        self._state_colors = list(colors)
        self._color_table = self._build_color_table()
        self.update()

    def render_image(self) -> QImage:
        # Solo la matriz, con el tamaño de celda que se ve en pantalla
        if not self.grid.flags["C_CONTIGUOUS"]:
            self.grid = np.ascontiguousarray(self.grid)
        h, w = self.grid.shape
        img = QImage(self.grid.data, w, h, self.grid.strides[0], QImage.Format_Indexed8)
        img.setColorTable(self._color_table)
        cs = max(1, self._cell_size_cached)
        return img.scaled(w * cs, h * cs, Qt.IgnoreAspectRatio, Qt.FastTransformation).convertToFormat(
            QImage.Format_RGB32
        )

    def update_grid(self, grid):
        # Acepta ndarray o list-of-lists. Convierte a uint8 C-contiguo si hace falta.
        arr = np.asarray(grid)
        if arr.ndim == 1:
            arr = arr.reshape(self.grid_size, self.grid_size)
        if arr.dtype != np.uint8 or not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
        self.grid = arr
        self.update()
