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
        self._gradient_table: list[int] = self._build_gradient_table()

        # Modo continuo: grid float32; _index_grid (uint8) alimenta el QImage.
        # En modo discreto _index_grid ES grid (mismo objeto, sin copia).
        self._is_continuous = False
        self._index_grid = self.grid

        self._last_painted: tuple[int, int] | None = None
        self._draw_side = 0.0
        self._cell_size_f = 0.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        self.setMinimumSize(200, 200)
        self.setCursor(Qt.CrossCursor)
        self.setStyleSheet(
            "SimulationWidget { background: #ffffff; border: 1px solid #e3e3e8; border-radius: 8px; }"
        )
        self._recompute_geometry()

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------
    @property
    def cell_size(self):
        return self._cell_size_f

    def _recompute_geometry(self):
        # La grilla es cuadrada: se escala al lado menor del viewport
        # (sin dejar remanente de división entera) y se centra en el otro eje.
        if self.grid_size <= 0:
            self._draw_side = 0.0
            self._cell_size_f = 0.0
            self._offset_x = 0.0
            self._offset_y = 0.0
            return
        side = float(min(self.width(), self.height()))
        self._draw_side = side
        self._cell_size_f = side / self.grid_size
        self._offset_x = (self.width() - side) / 2.0
        self._offset_y = (self.height() - side) / 2.0

    def resizeEvent(self, event):
        self._recompute_geometry()
        super().resizeEvent(event)

    def _build_color_table(self) -> list[int]:
        # Tabla de 256 entradas (requerida por Format_Indexed8)
        default = QColor(128, 128, 128).rgb()
        n = len(self._state_colors)
        table = [0] * 256
        for i in range(256):
            table[i] = self._state_colors[i].rgb() if i < n else default
        return table

    def _build_gradient_table(self) -> list[int]:
        # Colormap continuo: interpolación lineal color(estado 0) → color(estado 1)
        # en 256 niveles, sobre el mismo path Format_Indexed8 del modo discreto.
        c0 = self._state_colors[0]
        c1 = self._state_colors[1] if len(self._state_colors) > 1 else QColor(0, 0, 0)
        table = [0] * 256
        for i in range(256):
            t = i / 255.0
            table[i] = QColor(
                round(c0.red()   + (c1.red()   - c0.red())   * t),
                round(c0.green() + (c1.green() - c0.green()) * t),
                round(c0.blue()  + (c1.blue()  - c0.blue())  * t),
            ).rgb()
        return table

    def paintEvent(self, _):
        painter = QPainter(self)
        if self._draw_side <= 0:
            return

        if not self._index_grid.flags["C_CONTIGUOUS"]:
            self._index_grid = np.ascontiguousarray(self._index_grid)

        h, w = self._index_grid.shape
        bytes_per_line = self._index_grid.strides[0]

        img = QImage(
            self._index_grid.data, w, h, bytes_per_line, QImage.Format_Indexed8
        )
        img.setColorTable(self._gradient_table if self._is_continuous else self._color_table)

        painter.drawImage(
            QRectF(self._offset_x, self._offset_y, self._draw_side, self._draw_side), img
        )

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
        cf = self._cell_size_f
        if cf <= 0:
            return
        # Floor division en float: los clicks en el margen (coordenada negativa
        # tras restar el offset) caen en -1, no en la celda 0.
        x = int((event.pos().x() - self._offset_x) // cf)
        y = int((event.pos().y() - self._offset_y) // cf)
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return
        if (x, y) == self._last_painted:
            return
        self._last_painted = (x, y)

        current = float(self.grid[y, x])
        new_val = 0 if current == self._active_state else self._active_state
        self.grid[y, x] = new_val
        if self._is_continuous:
            self._index_grid[y, x] = np.uint8(min(max(float(new_val), 0.0), 1.0) * 255)
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
        self._gradient_table = self._build_gradient_table()
        self.update()

    def update_grid(self, grid):
        # Acepta ndarray o list-of-lists. dtype float → render continuo (colormap);
        # dtype entero → render discreto (tabla de colores por estado).
        arr = np.asarray(grid)
        if arr.ndim == 1:
            arr = arr.reshape(self.grid_size, self.grid_size)

        if np.issubdtype(arr.dtype, np.floating):
            if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
            self.grid = arr
            self._is_continuous = True
            # Cuantización a 256 niveles una vez por frame; paintEvent no asigna
            self._index_grid = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            if arr.dtype != np.uint8 or not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr, dtype=np.uint8)
            self.grid = arr
            self._is_continuous = False
            self._index_grid = arr
        self.update()
