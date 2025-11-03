from PyQt5.QtWidgets import QGraphicsRectItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QPen, QColor

class CellItem(QGraphicsRectItem):
    def __init__(self, x, y, size, states = [0,1], colors = [QColor(255,255,255), QColor(0,0,0)]):
        super().__init__(0, 0, size, size)
        self.row = x
        self.col = y
        self.states = states
        self.colors = colors
        self.state = 0
        self.num_states = len(states)

        self.setBrush(QBrush(self.colors[self.state]))
        self.setPen(QPen(Qt.gray, 0.5))
        self.setPos(y * size, x * size)

    def add_state(self, color):
        self.states.append(self.num_states)
        self.colors.append(color)
        self.num_states += 1

    def toggle_state(self):
        """Cambia el estado y el color de la celda"""
        self.state = (self.state + 1) % self.num_states
        color = self.colors[self.state]
        self.setBrush(QBrush(color))

    def set_state(self, state):
        """Cambia el estado al dado"""
        if state < self.num_states:
            self.state = state
            self.setBrush(QBrush(self.colors[self.state]))

    def change_color(self, color, state):
        self.colors[state] = color
        if self.state == state:
            self.setBrush(QBrush(color))