# ui.NeighbourWidget

import random
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QMouseEvent
from PyQt5.QtCore import Qt


def _rand_color():
    return QColor(random.randint(60, 210), random.randint(60, 210), random.randint(60, 210))


class NeighbourWidget(QWidget):

    def __init__(self, grid_size: int = 7):
        super().__init__()
        self._cell_px  = 20
        self.grid_size = grid_size
        self.center    = grid_size // 2

        self.grid = [[0] * grid_size for _ in range(grid_size)]

        self.active_state = 1
        self.center_color = QColor(200, 175, 100)

        # group_colors[0] = vacío (blanco), group_colors[1+] = grupos
        self.group_colors: list[QColor] = [QColor(255, 255, 255), QColor(50, 120, 220)]

        self._last_painted: tuple[int, int] | None = None

        self.setFixedSize(grid_size * self._cell_px, grid_size * self._cell_px)

    # ------------------------------------------------------------------
    # Grupos
    # ------------------------------------------------------------------
    def n_groups(self) -> int:
        return len(self.group_colors) - 1

    def add_group(self, color: QColor = None) -> int:
        if color is None:
            color = _rand_color()
        self.group_colors.append(color)
        return self.n_groups()

    def remove_last_group(self) -> bool:
        if self.n_groups() <= 1:
            return False
        last_state = self.n_groups()
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == last_state:
                    self.grid[row][col] = 0
        self.group_colors.pop()
        if self.active_state == last_state:
            self.active_state = self.n_groups()
        self.update()
        return True

    def set_active_state(self, state: int):
        self.active_state = state

    # ------------------------------------------------------------------
    # Datos
    # ------------------------------------------------------------------
    def get_neighbourhood(self) -> list:
        """Devuelve lista de grupos ordenados por primera aparición en el grid
        (arriba-izquierda → abajo-derecha), independiente del número de grupo."""
        groups: dict[int, list] = {}
        first_seen: dict[int, int] = {}
        pos = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                s = self.grid[row][col]
                if s > 0:
                    if s not in first_seen:
                        first_seen[s] = pos
                    groups.setdefault(s, []).append(
                        [col - self.center, row - self.center]
                    )
                pos += 1
        order = sorted(groups, key=lambda s: first_seen[s])
        return [groups[s] for s in order]

    def load_neighbourhood(self, neighbourhood: list):
        """Carga una vecindad agrupada en el grid."""
        self.reset()
        while self.n_groups() < len(neighbourhood):
            self.add_group()
        for group_idx, group in enumerate(neighbourhood):
            state = group_idx + 1
            for col_off, row_off in group:
                col = self.center + col_off
                row = self.center + row_off
                if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
                    self.grid[row][col] = state
        self.update()

    def reset(self):
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.update()

    # ------------------------------------------------------------------
    # Pintura
    # ------------------------------------------------------------------
    def paintEvent(self, _):
        painter = QPainter(self)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                s = self.grid[row][col]
                is_center = (col == self.center and row == self.center)

                if s == 0:
                    color = self.center_color if is_center else QColor(255, 255, 255)
                else:
                    color = (self.group_colors[s]
                             if s < len(self.group_colors)
                             else QColor(128, 128, 128))

                x = col * self._cell_px
                y = row * self._cell_px

                # Relleno con borde gris
                painter.setPen(QPen(Qt.gray, 0.5))
                painter.setBrush(QBrush(color))
                painter.drawRect(x, y, self._cell_px, self._cell_px)

                # Borde negro extra para la celda central
                if is_center:
                    painter.setPen(QPen(Qt.black, 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(x + 1, y + 1, self._cell_px - 2, self._cell_px - 2)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent):
        self._last_painted = None
        self._paint_at(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._last_painted = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self._paint_at(event)

    def _paint_at(self, event: QMouseEvent):
        col = event.pos().x() // self._cell_px
        row = event.pos().y() // self._cell_px
        if not (0 <= col < self.grid_size and 0 <= row < self.grid_size):
            return
        if (col, row) == self._last_painted:
            return
        self._last_painted = (col, row)
        current = self.grid[row][col]
        self.grid[row][col] = 0 if current == self.active_state else self.active_state
        self.update()
