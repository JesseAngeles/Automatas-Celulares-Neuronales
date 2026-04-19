# ui.ControlPane

from PyQt5.QtWidgets import QWidget, QPushButton, QSlider, QVBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal, Qt

class ControlPane(QWidget):
    # signal
    play_clicked = pyqtSignal()
    step_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    speed_moved = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        # Define speed slider
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(750)
        
        # Define labels
        self.simulation_label = QLabel("Simulation")
        self.speed_label = QLabel("Speed")
        self.grid_label = QLabel("Grid")
        self.record_label = QLabel("Record")
        self.file_label = QLabel("Files")

        # Define buttons
        self.step_button = QPushButton("step")
        self.play_button = QPushButton("play")
        self.backstep_button = QPushButton("backstep")
        self.erase_button = QPushButton("Erase")
        self.random_button = QPushButton("Random")
        self.record_botton = QPushButton("Record N steps")
        self.import_button = QPushButton("Import evolution")
        self.export_button = QPushButton("Export evolution") 

        # Definir el layout
        layout = QVBoxLayout()

        # Añadir elementos
        layout.addWidget(self.simulation_label)
        layout.addWidget(self.step_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.backstep_button)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_slider)
        
        layout.addWidget(self.grid_label)
        layout.addWidget(self.erase_button)
        layout.addWidget(self.random_button)

        layout.addWidget(self.record_label)
        layout.addWidget(self.record_botton)
        
        layout.addWidget(self.file_label)
        layout.addWidget(self.import_button)
        layout.addWidget(self.export_button)

        # Definir layout
        self.setLayout(layout)

        # Connect the signals
        self.play_button.clicked.connect(self.play_clicked.emit)
        self.step_button.clicked.connect(self.step_clicked.emit)
        self.erase_button.clicked.connect(self.reset_clicked.emit)
        self.speed_slider.valueChanged.connect(self.speed_moved.emit)

    def update_play_button(self, is_running: bool):
        self.play_button.setText("play" if is_running else "pause")

    def connect_controller(self, controller):
        self.step_button.clicked.connect(controller.on_step_button)
        self.play_button.clicked.connect(controller.on_play_button)
        self.backstep_button.clicked.connect(controller.on_backstep_button)
        self.erase_button.clicked.connect(controller.on_erase_button)
        self.random_button.clicked.connect(controller.on_random_button)