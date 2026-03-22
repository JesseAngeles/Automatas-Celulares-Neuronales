# main.py

import sys
from PyQt5.QtWidgets import QApplication

from ui.MainWindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(100)
    window.show()

    app.exec()
