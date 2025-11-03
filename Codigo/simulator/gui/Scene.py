from PyQt5.QtWidgets import QGraphicsScene

class Scene(QGraphicsScene):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.mouse_down = False
        self.last_item = None
        self.state = -1

    def mousePressEvent(self, event):
        self.mouse_down = True
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if hasattr(item, "toggle_state"):
            if self.state >=0:
                item.set_state(self.state)
            else:
                item.toggle_state()
            self.last_item = item

    def mouseMoveEvent(self, event):
        if self.mouse_down:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if item is not None and item is not self.last_item and hasattr(item, "toggle_state"):
                if self.state >=0:
                    item.set_state(self.state)
                else:
                    item.toggle_state()
                self.last_item = item

    def mouseReleaseEvent(self, event):
        self.mouse_down = False      
        self.last_item = None  