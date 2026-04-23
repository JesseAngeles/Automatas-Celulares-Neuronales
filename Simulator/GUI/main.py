# main.py

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5.QtWidgets import QApplication

from ui.MainWindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(100)
    window.show()

    app.exec()
