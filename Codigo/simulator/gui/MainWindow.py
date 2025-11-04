from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QPushButton,
    QMainWindow,
    QStatusBar,
    QWidget,
    QVBoxLayout,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QDialog,
    QColorDialog,
    QGraphicsView
)
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPainter, QColor

from gui.NeighbourWindow import NeighbourWindow
from gui.Dialog import Dialog
from gui.CellItem import CellItem
from gui.Scene import Scene

from model.CAConfig import CAConfig
from model.CASimulator import CASimulator

from utils.colors import random_colors

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Cellular Automaton simulator")
        self.setGeometry(200,0,1280,1000)

        ################################################
        #   VARIABLES
        ################################################

        # Clases
        self.config = CAConfig(num_states=2)
        self.ca_simulator = CASimulator(self.config)
        
        self.config.evolution = [self.ca_simulator.get_grid()]
        self.steps = 0

        # Window
        self.last_button_clicked = None
        self.max_states = 21
        self.colors = [QColor(255,255,255)] + random_colors(self.config.num_states - 1)
        
        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.evolve_space)
        self.speed = 125 
        self.timer.setInterval(self.speed)

        ################################################
        #   PANELES
        ################################################

        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Panel izquierdo
        left_panel = QVBoxLayout()

        ca_frame = QFrame()
        ca_frame.setFrameShape(QFrame.Box)
        ca_frame.setStyleSheet("background-color: #999;")
        ca_frame.setMinimumHeight(850)
        ca_frame.setMinimumWidth(850)
        
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.Box)
        controls_layout = QHBoxLayout(controls_frame)

        # Panel derecho
        right_panel = QVBoxLayout()

        config_frame = QFrame()
        config_frame.setFrameShape(QFrame.Box)
        config_layout = QVBoxLayout(config_frame)

        states_frame = QFrame()
        states_frame.setFrameShape(QFrame.Box)
        self.states_layout = QVBoxLayout(states_frame)

        ################################################
        #   INPUTS Y LABELS
        ################################################

        self.state_input = QLineEdit()
        self.state_input.setPlaceholderText("Number of states: ")
        self.state_input.setText(str(self.config.num_states))
        
        self.rule_input = QLineEdit()
        self.rule_input.setPlaceholderText("Rule number: ")
        self.rule_input.setText(str(self.config.rule))

        ################################################
        #   CA
        ################################################

        layout = QVBoxLayout(ca_frame)
        layout.setContentsMargins(0,0,0,0)

        self.scene = Scene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, False)        
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(self.view)

        self.cells = []
        for i in range(self.config.grid_size):
            row = []
            for j in range(self.config.grid_size):
                cell = CellItem(x = i, y = j, size = self.config.cell_size, states=list(range(0, self.config.num_states)), colors= self.colors)
                self.scene.addItem(cell)
                row.append(cell)
            self.cells.append(row)

        scene_width = self.config.grid_size * self.config.cell_size
        scene_height = self.config.grid_size * self.config.cell_size
        
        self.scene.setSceneRect(QRectF(0, 0, scene_width, scene_height))
        self.view.setFixedSize(scene_width + 2, scene_height + 2)

        ################################################
        #   BOTONES
        ################################################

        # controls
        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_button_clicked)
        
        step_button = QPushButton("Step")
        step_button.clicked.connect(self.step_button_clicked)
        
        back_step_button = QPushButton("Back Step")
        back_step_button.clicked.connect(self.back_step_button_clicked)
        
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_button_clicked)

        speed_up_button = QPushButton(">")
        speed_up_button.clicked.connect(self.speed_up_button_clicked)

        slow_down_button = QPushButton("<")
        slow_down_button.clicked.connect(self.slow_down_button_clicked)

        # config
        neighbour_button = QPushButton("Neighbour")
        neighbour_button.clicked.connect(self.neighbour_button_clicked)

        frontier_button = QPushButton("Frontier")
        frontier_button.clicked.connect(self.frontier_button_clicked)

        states_button = QPushButton("States")
        states_button.clicked.connect(self.states_button_clicked)

        self.state_buttons = []
        for i, color in enumerate(self.colors):
            button = QPushButton(f"State {i}")
            button.setStyleSheet(f"background-color: {color.name()}")
            button.clicked.connect(self.color_button_clicked)
            self.state_buttons.append(button)


        rule_button = QPushButton("Rule")
        rule_button.clicked.connect(self.rule_button_clicked)

        ################################################
        #   JUNTAR ELEMENTOS
        ################################################
        self.setCentralWidget(central_widget)
        main_layout.addLayout(left_panel, stretch=3)

        # CA
        left_panel.addWidget(ca_frame, stretch=3)

        # CONTROLS
        left_panel.addWidget(controls_frame, stretch=3)
        for btn in [slow_down_button, speed_up_button, back_step_button, start_button, step_button, reset_button]:
            controls_layout.addWidget(btn)

        # CONFIG
        main_layout.addLayout(right_panel, stretch=3)
        right_panel.addWidget(config_frame, stretch=3)
        
        config_layout.addWidget(neighbour_button)
        config_layout.addWidget(frontier_button)
        config_layout.addWidget(self.state_input)
        config_layout.addWidget(states_button)
        config_layout.addWidget(states_frame)
        for btn in self.state_buttons:
            self.states_layout.addWidget(btn)
        config_layout.addWidget(self.rule_input)
        config_layout.addWidget(rule_button)

        ################################################
        #   ACCIONES
        ################################################

        # Import button
        import_button = QAction("Import", self)
        import_button.setStatusTip("Import a CA evolution")
        import_button.triggered.connect(self.import_button_clicked)

        # Export button
        export_button = QAction("Export", self)
        export_button.setStatusTip("Export the CA evolution")
        export_button.triggered.connect(self.export_button_clicked)

        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction(import_button)
        file_menu.addAction(export_button)

    # Button events

    def start_button_clicked(self, s):
        grid_states = [[cell.state for cell in row] for row in self.cells]
        self.ca_simulator.set_grid(grid=grid_states)

        if self.is_running:
            self.timer.stop()
        else:
            self.config.evolution = [grid_states]
            self.timer.start()

        self.is_running = 1 - self.is_running

    def step_button_clicked(self, s):
        grid_states = [[cell.state for cell in row] for row in self.cells]
        self.ca_simulator.set_grid(grid=grid_states)
        self.evolve_space()
    
    def back_step_button_clicked(self, s):
        if len(self.config.evolution) > 1:
            self.config.evolution.pop()
            grid_states = self.config.evolution[-1]
            self.ca_simulator.set_grid(grid_states)
            for i, row in enumerate(self.ca_simulator.get_grid()):
                for j, state in enumerate(row):
                    self.cells[i][j].set_state(state)
        
    def reset_button_clicked(self, s):
        # Detener la animacion
        self.is_running = False
        self.timer.stop()

        if self.ca_simulator.get_grid() == self.config.evolution[-1]:
            self.ca_simulator.set_grid([[0 for _ in range(self.config.grid_size)] for _ in range(self.config.grid_size)])
        else:
            self.ca_simulator.set_grid(self.config.evolution[0])
            self.config.evolution = [self.config.evolution[0]]

        for i, row in enumerate(self.ca_simulator.get_grid()):
            for j, state in enumerate(row):
                self.cells[i][j].set_state(state)

    def speed_up_button_clicked(self, s):
        self.speed = int(self.speed / 1.2)
        self.timer.setInterval(self.speed)

    def slow_down_button_clicked(self, s):
        self.speed = int(self.speed * 1.2)
        self.timer.setInterval(self.speed)

    def neighbour_button_clicked(self, s):
        self.neighbour_window = NeighbourWindow(parent=self, config=self.config)
        self.neighbour_window.saved.connect(self.handle_neighbour_data)  # ← conectar señal
        self.hide()
        self.neighbour_window.show()

    def frontier_button_clicked(self, s):
        print("Frontier", s)

    def states_button_clicked(self, s):
        try:
            new_total_states = int(self.state_input.text())
        except ValueError:
            return

        if new_total_states > self.max_states:
            new_total_states = self.max_states
            self.state_input.setText(str(new_total_states))

        if new_total_states < self.config.num_states:
            self.state_dialog = Dialog("The states are fewer, want to change?", parent=self)
            result = self.state_dialog.exec_()
            if result != QDialog.Accepted:
                self.state_input.setText(str(self.config.num_states))
                return
            else:
                self.colors = self.colors[:new_total_states]
        else:
            self.colors.extend(random_colors(new_total_states - len(self.colors)))

        self.config.num_states = min(new_total_states, self.max_states)
        self.state_input.setText(str(self.config.num_states))
        self.ca_simulator.set_states(self.config.num_states)


        # Eliminar los botones antiguos
        for btn in self.state_buttons:
            self.states_layout.removeWidget(btn)
            btn.setParent(None)

        # Crear los nuevos botones
        self.state_buttons = []
        for i, color in enumerate(self.colors):
            new_btn = QPushButton(f"State {i}")
            new_btn.setStyleSheet(f"background-color: {color.name()}")
            new_btn.clicked.connect(self.color_button_clicked)
            self.states_layout.addWidget(new_btn)
            self.state_buttons.append(new_btn)

        # Añadir color a todas las celulas
        for row in self.cells:
            for cell in row:
                cell.reset_states(self.colors)

    def rule_button_clicked(self, s):
        try:
            new_rule = int(self.rule_input.text())
        except ValueError:
            return

        self.config.rule = new_rule
        self.ca_simulator.set_rule(self.config.rule)

    def color_button_clicked(self, s):
        button = self.sender()
        index = self.state_buttons.index(button)
        # Se presiono el mismo boton dos veces
        if self.last_button_clicked == index:
            color = QColorDialog.getColor()
            if color.isValid():
                button.setStyleSheet(f"background-color: {color.name()};")

            for row in self.cells:
                for cell in row:
                    cell.change_color(color, index)
        
        self.last_button_clicked = index
        self.scene.state = index

    def import_button_clicked(self, s):
        print("import", s)

    def export_button_clicked(self, s):
        print("export", s)

    # Signal events
    def handle_neighbour_data(self, active_cells):
        self.config.neighbourhood = active_cells
        self.ca_simulator.set_neighbourhood(self.config.neighbourhood)

    # fucntions
    def evolve_space(self):
        self.ca_simulator.step()
        self.config.evolution.append(self.ca_simulator.get_grid())
        for i, row in enumerate(self.ca_simulator.get_grid()):
            for j, state in enumerate(row):
                self.cells[i][j].set_state(state)

app = QApplication([])
window = MainWindow()
window.show()
app.exec()