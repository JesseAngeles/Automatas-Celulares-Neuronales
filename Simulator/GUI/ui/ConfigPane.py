# ui.ConfigPane

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox
from ui.NeighbourWidget import NeighbourWidget

class ConfigPane(QWidget):
    def __init__(self):
        super().__init__()

        self.neighbout_widget = NeighbourWidget()

        # define labels
        self.neighbour_label = QLabel("Neighbourhood")
        self.states_label = QLabel("States")
        self.frontier_label = QLabel("Frontier")
        self.rule_label = QLabel("Rule")

        # define buttons
        self.neighbour_button = QPushButton("Neighbour")
        self.frontier_button = QPushButton("Frontier")
        self.states_button = QPushButton("States")
        self.rule_button = QPushButton("Rule")

        # define other inputs
        self.states_spin = QSpinBox()
        self.states_spin.setRange(0, 10)
        self.states_spin.setValue(2)

        self.rule_spin = QSpinBox()
        self.rule_spin.setRange(0, 10000)
        self.rule_spin.setValue(210)

        # define layout
        layout = QVBoxLayout()
        states_layout = QHBoxLayout()
        rule_layout = QHBoxLayout()

        # add elements
        layout.addWidget(self.neighbour_label)
        layout.addWidget(self.neighbout_widget)
        layout.addWidget(self.neighbour_button)

        layout.addWidget(self.states_label)
        states_layout.addWidget(self.states_spin)
        states_layout.addWidget(self.states_button)
        layout.addLayout(states_layout)

        layout.addWidget(self.frontier_label)
        layout.addWidget(self.frontier_button)

        layout.addWidget(self.rule_label)
        rule_layout.addWidget(self.rule_spin)
        rule_layout.addWidget(self.rule_button)
        layout.addLayout(rule_layout)

        # define layout
        self.setLayout(layout)

    def connect_config(self, config):
        self.neighbour_button.clicked.connect(config.on_neighbour_button)
        self.frontier_button.clicked.connect(config.on_frontier_button)
        self.states_button.clicked.connect(config.on_states_button)
        self.rule_button.clicked.connect(config.on_rule_button)
