from PyQt5.QtWidgets import (
    QHBoxLayout, QFrame, QWidget, QVBoxLayout,
    QPushButton, QGraphicsView, QMessageBox, QColorDialog
)
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QRectF, pyqtSignal

from gui.Scene import Scene
from gui.CellItem import CellItem
from model.CAConfig import CAConfig

from utils.colors import random_colors


class NeighbourWindow(QWidget):
    saved = pyqtSignal(object)

    def __init__(self, parent=None, config=CAConfig()):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Neighbour configuration")
        self.setGeometry(0, 0, 500, 500)
        
        self.last_button_clicked = None
        self.neighbourhoods_size = len(config.neighbourhood)

        # Generar colores (el 0 siempre es blanco)
        self.colors = [QColor(255, 255, 255)] + random_colors(self.neighbourhoods_size)
        print(len(self.colors))
        self.center_color = QColor(200, 175, 100)

        left_panel = QVBoxLayout()
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        ################################################
        #   FRAME PARA LA GRILLA
        ################################################
        ca_frame = QFrame()
        ca_frame.setFrameShape(QFrame.Box)
        ca_frame.setStyleSheet("background-color: #999;")
        ca_layout = QVBoxLayout(ca_frame)
        ca_layout.setContentsMargins(0, 0, 0, 0)

        ################################################
        #   ESCENA Y CELDAS
        ################################################
        self.scene = Scene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, False)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        ca_layout.addWidget(self.view)

        self.center = int(config.grid_neighbour_size / 2)
        self.cells = []

        for i in range(config.grid_neighbour_size):
            row = []
            for j in range(config.grid_neighbour_size):
                if i == self.center and j == self.center:
                    colors = [self.center_color] + self.colors[1::]
                else:
                    colors = list(self.colors)
                cell = CellItem(x=i, y=j, size=config.cell_neighbour_size, states=list(range(0,self.neighbourhoods_size+1)), colors=colors)
                self.scene.addItem(cell)
                row.append(cell)
            self.cells.append(row)

        scene_width = config.grid_neighbour_size * config.cell_neighbour_size
        scene_height = config.grid_neighbour_size * config.cell_neighbour_size
        self.scene.setSceneRect(QRectF(0, 0, scene_width, scene_height))
        self.view.setFixedSize(scene_width + 2, scene_height + 2)

        # Cargar vecindades existentes
        for i, neighbourhood in enumerate(config.neighbourhood, start=1):
            for n in neighbourhood:
                self.cells[self.center + n[0]][self.center + n[1]].set_state(i)

        ################################################
        #   BOTONES DE CONTROL
        ################################################
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_button_clicked)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_button_clicked)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_button_clicked)

        controls_layout.addWidget(add_button)
        controls_layout.addWidget(save_button)
        controls_layout.addWidget(cancel_button)

        ################################################
        #   DESCRIPCIÓN / BOTONES DE COLOR
        ################################################
        right_panel = QVBoxLayout()
        description_frame = QFrame()
        description_frame.setFrameShape(QFrame.Box)
        self.description_layout = QVBoxLayout(description_frame)

        self.neighbour_button = []

        # Crear botón para el estado 0
        for i, color in enumerate(self.colors):
            button = QPushButton(f"Set {i}")
            button.setStyleSheet(f"background-color: {color.name()}")
            button.clicked.connect(self.color_button_clicked)
            self.neighbour_button.append(button)
            self.description_layout.addWidget(button)

        ################################################
        #   ENSAMBLAR TODO
        ################################################
        main_layout.addLayout(left_panel, stretch=3)
        left_panel.addWidget(ca_frame)
        left_panel.addWidget(controls_frame)
        main_layout.addLayout(right_panel, stretch=3)
        right_panel.addWidget(description_frame)

    ################################################
    #   ACCIONES DE BOTONES
    ################################################
    def add_button_clicked(self, s):
        print("add", s)
        color = random_colors(1)[0]

        for row in self.cells:
            for cell in row:
                cell.add_state(color)

        self.neighbourhoods_size += 1
        self.colors.append(color)

        new_button = QPushButton(f"Set {self.neighbourhoods_size}")
        new_button.clicked.connect(self.color_button_clicked)
        new_button.setStyleSheet(f"background-color: {color.name()}")
        self.neighbour_button.append(new_button)
        self.description_layout.addWidget(new_button)

    def save_button_clicked(self, s):
        grouped_cells = {}
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                if cell.state != 0:  # ignorar estado 0
                    rel_x = i - self.center
                    rel_y = j - self.center
                    grouped_cells.setdefault(cell.state, []).append([rel_x, rel_y])

        if not grouped_cells:
            QMessageBox.warning(self, "Warning", "neighbourhood needs at least 1 element")
            return

        self.saved.emit([grouped_cells[state] for state in sorted(grouped_cells.keys())])

        if self.parent:
            self.parent.show()
        self.close()

    def cancel_button_clicked(self, s):
        if self.parent:
            self.parent.show()
        self.close()

    def color_button_clicked(self):
        button = self.sender()
        index = self.neighbour_button.index(button)
        if self.last_button_clicked == index:
            color = QColorDialog.getColor()
            if color.isValid():
                button.setStyleSheet(f"background-color: {color.name()};")
                self.colors[index] = color

                for row in self.cells:
                    for cell in row:
                        cell.change_color(color, index)
                if index == 0:
                    self.cells[self.center][self.center].change_color(self.center_color,0)
        
        self.last_button_clicked = index
        self.scene.state = index