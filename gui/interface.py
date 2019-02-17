from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget, QFileDialog)


class DiagnosisInterface(QDialog):
    def __init__(self, parent=None):
        super(DiagnosisInterface, self).__init__(parent)

        self.original_palette = QApplication.palette()

        folder = str(QFileDialog.getExistingDirectory())


if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    gallery = DiagnosisInterface()
    gallery.show()
    sys.exit(app.exec_())

