from gui.gui import *
import sys

app = QApplication(sys.argv)
window = GUI()
window.setWindowTitle('Homework 2: Segmentation Result')
window.show()
sys.exit(app.exec())
