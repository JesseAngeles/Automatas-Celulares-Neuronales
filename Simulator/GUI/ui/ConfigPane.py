# ui.ConfigPane

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout

class ConfigPane(QWidget):
    def __init__(self):
        super().__init__()

        # define buttons
        self.neighbour_button = QPushButton("Neighbour")
        self.frontier_button = QPushButton("Frontier")
        self.states_button = QPushButton("States")
        self.rule_button = QPushButton("Rule")

        # define layout
        layout = QVBoxLayout()

        # add elements
        layout.addWidget(self.neighbour_button)
        layout.addWidget(self.frontier_button)
        layout.addWidget(self.states_button)
        layout.addWidget(self.rule_button)
    
        # define layout
        self.setLayout(layout)

    def connect_config(self, config):
        self.neighbour_button.clicked.connect(config.on_neighbour_button)
        self.frontier_button.clicked.connect(config.on_frontier_button)
        self.states_button.clicked.connect(config.on_states_button)
        self.rule_button.clicked.connect(config.on_rule_button)
