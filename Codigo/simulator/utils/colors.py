import random
from PyQt5.QtGui import QColor

def random_colors(number):
    return [
        QColor(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        for _ in range(number)
    ]