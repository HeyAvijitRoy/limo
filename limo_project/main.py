import sys
import os

# Add parent directory of limo_project to sys.path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from limo_project.ui.main_window import MainWindow, get_icon_path

APP_USER_MODEL_ID = "com.avijitroy.limo"


def set_windows_app_user_model_id():
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def main():
    # Set the working directory to the project directory to ensure files are placed correctly
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    set_windows_app_user_model_id()

    app = QApplication(sys.argv)
    app.setApplicationName("Local Intelligent Media Organizer")
    app.setDesktopFileName(APP_USER_MODEL_ID)
    app.setWindowIcon(QIcon(get_icon_path()))
    
    # Establish SQLite database path in project folder
    db_path = os.path.join(project_dir, "face_index.db")
    
    window = MainWindow(db_path=db_path)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
