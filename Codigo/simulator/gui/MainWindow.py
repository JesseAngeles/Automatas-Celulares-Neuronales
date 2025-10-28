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
    QColorDialog
)

from gui.NeighbourWindow import NeighbourWindow
from gui.Dialog import Dialog
from model.CAConfig import CAConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Cellular Automaton simulator")
        self.setGeometry(200,0,1280,1080)

        ################################################
        #   VARIABLES
        ################################################

        self.config = CAConfig(num_states=2)

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
        ca_frame.setMinimumHeight(875)
        ca_frame.setMinimumWidth(875)
        
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
        self.state_input.setText("2")
        
        self.rule_input = QLineEdit()
        self.rule_input.setPlaceholderText("Rule number: ")
        self.rule_input.setText("1")

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
        for i in range(self.config.num_states):
            self.state_buttons.append(QPushButton(f"State {i}"))
            self.state_buttons[i].clicked.connect(self.color_button_clicked)
        
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


    def start_button_clicked(self, s):
        print("start", s)

    def step_button_clicked(self, s):
        print("step", s)

    def back_step_button_clicked(self, s):
        print(" back step", s)

    def reset_button_clicked(self, s):
        print("reset", s)

    def speed_up_button_clicked(self, s):
        print("speed up", s)

    def slow_down_button_clicked(self, s):
        print("slow down", s)

    def neighbour_button_clicked(self, s):
        print("Neighbour", s)
        self.neighbour_window = NeighbourWindow(parent=self)
        self.hide()
        self.neighbour_window.show()

    def frontier_button_clicked(self, s):
        print("Frontier", s)

    def states_button_clicked(self, s):
        print("State", s)
        try:
            new_total_states = int(self.state_input.text())
        except ValueError:
            print("Ingrese un número válido")
            return

        if new_total_states >= self.config.num_states:
            self.config.num_states = new_total_states
        else:
            self.state_dialog = Dialog("The states are fewer, want to change?", parent=self)
            result = self.state_dialog.exec_()
            if result == QDialog.Accepted:
                self.config.num_states = new_total_states
            else:
                print("Canceled")
                return

        # --- Actualizar botones ---
        # Primero eliminamos los botones antiguos del layout
        for btn in self.state_buttons:
            self.states_layout.removeWidget(btn)
            btn.setParent(None)  # elimina referencia visual

        # Limpiar la lista de botones
        self.state_buttons = []

        # Crear los nuevos botones según el nuevo número de estados
        for i in range(self.config.num_states):
            new_btn = QPushButton(f"State {i}")
            new_btn.clicked.connect(self.color_button_clicked)
            self.states_layout.addWidget(new_btn)
            self.state_buttons.append(new_btn)

    def rule_button_clicked(self, s):
        print("Rule", s)

    def color_button_clicked(self, s):
        print("color", s)
        button = self.sender()

        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()};")

    def import_button_clicked(self, s):
        print("import", s)

    def export_button_clicked(self, s):
        print("export", s)


app = QApplication([])
window = MainWindow()
window.show()
app.exec()