# main.py

import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5.QtWidgets import QApplication

from ui.MainWindow import MainWindow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100, help="Tamaño inicial de la ventana")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(args.size)
    window.showMaximized()

    app.exec()