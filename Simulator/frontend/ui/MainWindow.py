# ui.MainWindow

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout

from ui.ControlPane import ControlPane
from ui.SimulationWidget import SimulationWidget
from ui.ConfigPane import ConfigPane

from controllers.SimulationController import SimulationController
from controllers.SimulationConfig import SimulationConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cellular Automata Simulator")
        self.setFixedSize(QSize(1200, 900))

        # create config
        self.config = SimulationConfig()

        # create widgets
        self.control_pane = ControlPane()
        self.control_pane.setFixedWidth(200)
        
        self.simulation = SimulationWidget()

        self.config_pane = ConfigPane()
        self.config_pane.setFixedWidth(200)

        # Definir conexiones
        self.control_pane.play_clicked.connect(self.handle_play)
        self.control_pane.reset_clicked.connect(self.handle_reset)

        # define controller        
        self.controller = SimulationController(self.simulation, self.config)
        
        # connect UI
        self.control_pane.connect_controller(self.controller)
        self.config_pane.connect_config(self.config)
    
        # Definir el layout    
        layout = QHBoxLayout()

        layout.addWidget(self.control_pane)
        layout.addWidget(self.simulation)
        layout.addWidget(self.config_pane)

        # definir container
        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def handle_play(self):
        running = self.controller.is_running
        self.control_pane.update_play_button(running)
    
    def handle_reset(self):
        empty_grid = [[0] * self.simulation.grid_size for _ in range(self.simulation.grid_size)]
        self.simulation.update_grid(empty_grid)
        self.handle_play()