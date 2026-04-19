from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

# Ventana emergente
class Dialog(QDialog):
    def __init__(self, message=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("State change")
        self.setGeometry(300, 300, 300, 150)

        layout = QVBoxLayout()

        # Mensaje
        layout.addWidget(QLabel(message))

        # Botones
        button_layout = QHBoxLayout()
        accept_button = QPushButton("Accept")
        reject_button = QPushButton("Cancel")
        button_layout.addWidget(accept_button)
        button_layout.addWidget(reject_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Conectar botones
        accept_button.clicked.connect(self.accept) 
        reject_button.clicked.connect(self.reject) 
