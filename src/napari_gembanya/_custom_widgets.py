from qtpy.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

class CollapsibleBox(QWidget):
    def __init__(self,title="Collapsible Box", parent=None):
        super().__init__(parent)

        self.toggle_title = QLabel(title)
        self.toggle_title.setStyleSheet("QLabel { padding: 5px; }")
        self.toggle_title.setIndent(10)

        self.content_widget = QWidget()
        self.content_widget.hide()

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toggle_title)
        self.layout.addWidget(self.content_widget)

        self.setLayout(self.layout)
        self.toggle_title.mousePressEvent = self.toggle_content

    def toggle_content(self, event):
        self.content_widget.setVisible(not self.content_widget.isVisible())

