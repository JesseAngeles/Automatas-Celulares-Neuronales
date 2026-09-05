# main.py

import argparse
import ctypes
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Precarga libcudart del paquete pip nvidia-cuda-runtime para que la extensión CA lo encuentre
try:
    import nvidia.cu13
    for _dir in nvidia.cu13.__path__:
        _cudart = os.path.join(_dir, "lib", "libcudart.so.13")
        if os.path.exists(_cudart):
            ctypes.CDLL(_cudart, mode=ctypes.RTLD_GLOBAL)
            break
except (ImportError, OSError):
    pass  # se asume que libcudart está disponible en el sistema

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