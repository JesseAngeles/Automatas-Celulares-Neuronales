# ui.ControlPane

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

class ControlPane(QWidget):
    # signal
    play_clicked = pyqtSignal();
    step_clicked = pyqtSignal();
    reset_clicked = pyqtSignal();

    def __init__(self):
        super().__init__()

        # Define buttons
        self.increase_speed = QPushButton("more speed")
        self.decrease_speed = QPushButton("less speed")
        self.step_button = QPushButton("step")
        self.play_button = QPushButton("play")
        self.backstep_button = QPushButton("backstep")
        self.erase_button = QPushButton("Erase")

        # Definir el layout
        layout = QVBoxLayout()

        # Añadir elementos
        layout.addWidget(self.increase_speed)
        layout.addWidget(self.decrease_speed)
        layout.addWidget(self.step_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.backstep_button)
        layout.addWidget(self.erase_button)

        # Definir layout
        self.setLayout(layout)

        # Connect the signals
        self.play_button.clicked.connect(self.play_clicked.emit)
        self.step_button.clicked.connect(self.step_clicked.emit)
        self.erase_button.clicked.connect(self.reset_clicked.emit)

    def update_play_button(self, is_running: bool):
        self.play_button.setText("play" if is_running else "pause")

    def connect_controller(self, controller):
        self.increase_speed.clicked.connect(controller.on_increase_speed_button)
        self.decrease_speed.clicked.connect(controller.on_decrease_speed_button)
        self.step_button.clicked.connect(controller.on_step_button)
        self.play_button.clicked.connect(controller.on_play_button)
        self.backstep_button.clicked.connect(controller.on_backstep_button)
        self.erase_button.clicked.connect(controller.on_erase_button)