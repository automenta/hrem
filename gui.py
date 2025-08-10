import sys
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainGUI


def main():
    """
    The main entry point for the GUI application.
    """
    app = QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
