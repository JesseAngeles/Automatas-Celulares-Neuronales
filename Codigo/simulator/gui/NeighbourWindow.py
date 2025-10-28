from PyQt5.QtWidgets import QHBoxLayout, QFrame, QWidget, QVBoxLayout, QPushButton

class NeighbourWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Neighbour configuration")
        self.setGeometry(0,0,500,500)

        layout =QVBoxLayout()
        self.setLayout(layout)

        ################################################
        #   FRAMES
        ################################################

        # ca
        ca_frame = QFrame()
        ca_frame.setFrameShape(QFrame.Box)
        ca_frame.setStyleSheet("background-color: #999;")
        ca_frame.setMinimumHeight(500)
        ca_frame.setMinimumWidth(600)

        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.Box)
        controls_layout = QHBoxLayout(controls_frame)

        ################################################
        #   BUTTONS
        ################################################

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_button_clicked)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_button_clicked)

        ################################################
        #   JUNTAR ELEMENTOS
        ################################################
        layout.addWidget(ca_frame)
        layout.addWidget(controls_frame)

        controls_layout.addWidget(save_button)
        controls_layout.addWidget(cancel_button)

    def save_button_clicked(self, s):
        print("save", s)
        if self.parent:
            self.parent.show()
        self.close()

    def cancel_button_clicked(self, s):
        print("cancel", s)
        if self.parent:
            self.parent.show()
        self.close()