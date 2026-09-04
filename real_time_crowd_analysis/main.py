import sys
import os
import traceback

# Add the root directory to sys.path to allow imports from real_time_crowd_analysis
# when running from inside the real_time_crowd_analysis directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir.endswith('real_time_crowd_analysis'):
    # We are inside the real_time_crowd_analysis directory, add parent to path
    root_dir = os.path.dirname(script_dir)
else:
    # We are in the root directory
    root_dir = script_dir

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def main():
    print("Initializing application...")
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except ImportError as e:
        print(f"Failed to import PyQt6: {e}")
        print("Please install PyQt6 using: pip install PyQt6")
        traceback.print_exc()
        sys.exit(1)

    print("Loading UI...")
    try:
        from real_time_crowd_analysis.ui.splash_screen import show_splash_screen
        from real_time_crowd_analysis.ui.theme import get_stylesheet
    except ImportError as e:
        print(f"Failed to import UI modules: {e}")
        traceback.print_exc()
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet())

    splash = show_splash_screen()

    # Try to load the main dashboard
    try:
        print("Loading dashboard...")
        from real_time_crowd_analysis.ui.dashboard import DashboardWindow
        window = DashboardWindow()
    except Exception as e:
        print(f"Failed to load dashboard: {e}")
        traceback.print_exc()
        print("Creating fallback minimal UI...")
        try:
            # Fallback minimal window
            from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
            window = QMainWindow()
            window.setWindowTitle("Real-Time Crowd Analysis - Fallback Mode")
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            label = QLabel("Dashboard failed to load. Running in fallback mode.\n"
                           "Check the console for errors.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            window.setCentralWidget(central_widget)
            window.resize(800, 600)
        except Exception as fallback_err:
            print(f"CRITICAL: Fallback UI failed: {fallback_err}")
            traceback.print_exc()
            sys.exit(1)

    # Close splash screen and show main window
    splash.close_splash()
    window.show()
    print("Window shown")
    print("Launching dashboard...")

    sys.exit(app.exec())

if __name__ == "__main__":
    main()