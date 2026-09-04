"""
Theme Module for Real-Time Crowd Analysis and Threat Detection
"""

from real_time_crowd_analysis.utils.config import config
from real_time_crowd_analysis.utils.logger import setup_logger

logger = setup_logger("theme")

class ThemeManager:
    """Manages UI themes and styling"""
    
    def __init__(self):
        self.current_theme = config.UI_THEME
        self.colors = self._get_theme_colors()
    
    def _get_theme_colors(self) -> dict:
        """Get color scheme for current theme"""
        if self.current_theme == "dark":
            return {
                'background': '#0a0a0a',
                'surface': '#1a1a1a',
                'primary': config.PRIMARY_COLOR,  # Blue
                'secondary': '#0066CC',
                'accent': '#00FFFF',
                'warning': config.WARNING_COLOR,  # Red
                'success': config.SUCCESS_COLOR,  # Green
                'text_primary': '#FFFFFF',
                'text_secondary': '#B0B0B0',
                'border': '#333333',
                'hover': '#2a2a2a'
            }
        else:  # light theme
            return {
                'background': '#FFFFFF',
                'surface': '#F5F5F5',
                'primary': config.PRIMARY_COLOR,  # Blue
                'secondary': '#0066CC',
                'accent': '#00FFFF',
                'warning': config.WARNING_COLOR,  # Red
                'success': config.SUCCESS_COLOR,  # Green
                'text_primary': '#000000',
                'text_secondary': '#666666',
                'border': '#CCCCCC',
                'hover': '#E0E0E0'
            }
    
    def get_color(self, color_name: str) -> str:
        """Get a color value by name"""
        return self.colors.get(color_name, '#000000')
    
    def set_theme(self, theme_name: str):
        """Set the current theme"""
        self.current_theme = theme_name
        self.colors = self._get_theme_colors()
        logger.info(f"Theme changed to: {theme_name}")
    
    def get_stylesheet(self) -> str:
        """Get Qt stylesheet for the current theme"""
        if self.current_theme == "dark":
            return f"""
            QMainWindow {{
                background-color: {self.colors['background']}; /* Dark black background */
                color: {self.colors['text_primary']}; /* Light grey text */
            }}
            QWidget:enabled {{ /* Apply to enabled widgets to allow transparent backgrounds for some elements */
                background-color: {self.colors['background']};
                color: {self.colors['text_primary']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.colors['border']};
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {self.colors['primary']};
            }}
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['secondary']};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['text_secondary']};
                color: {self.colors['border']};
            }}
            QLabel {{
                color: {self.colors['text_primary']};
            }}
            QLineEdit, QComboBox, QSpinBox {{
                background-color: {self.colors['surface']};
                color: {self.colors['text_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                padding: 5px;
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
                border: 2px solid {self.colors['primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid {self.colors['border']};
                background-color: {self.colors['background']};
            }}
            QTabBar::tab {{
                background-color: {self.colors['surface']};
                color: {self.colors['text_primary']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.colors['primary']};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {self.colors['hover']};
            }}
            QProgressBar {{
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                text-align: center;
                background-color: {self.colors['surface']};
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['primary']};
                border-radius: 2px;
            }}
            QTableWidget {{
                background-color: {self.colors['surface']};
                alternate-background-color: {self.colors['background']};
                color: {self.colors['text_primary']};
                gridline-color: {self.colors['border']};
            }}
            QHeaderView::section {{
                background-color: {self.colors['primary']};
                color: white;
                padding: 5px;
                border: 1px solid {self.colors['border']};
            }}
            """
        else:
            # Light theme stylesheet
            return f"""
            QMainWindow {{
                background-color: {self.colors['background']};
                color: {self.colors['text_primary']};
            }}
            QWidget {{
                background-color: {self.colors['background']};
                color: {self.colors['text_primary']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.colors['border']};
                border-radius: 5px;
                margin-top: 1ex; /* Space for title */
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                    color: {self.colors['primary']};
                }}
                QPushButton {{
                    background-color: {self.colors['primary']};
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {self.colors['hover']};
                }}
                QPushButton:pressed {{
                    background-color: {self.colors['secondary']};
                }}
                QLabel {{
                    color: {self.colors['text_primary']};
                }}
                QLineEdit, QComboBox, QSpinBox {{
                    background-color: {self.colors['surface']};
                    color: {self.colors['text_primary']};
                    border: 1px solid {self.colors['border']};
                    border-radius: 3px;
                    padding: 5px;
                }}
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
                    border: 2px solid {self.colors['primary']};
                }}
                QTabWidget::pane {{
                    border: 1px solid {self.colors['border']};
                    background-color: {self.colors['background']};
                }}
                QTabBar::tab {{
                    background-color: {self.colors['surface']};
                    color: {self.colors['text_primary']};
                    padding: 8px 16px;
                    margin-right: 2px;
                }}
                QTabBar::tab:selected {{
                    background-color: {self.colors['primary']};
                    color: white;
                }}
                QProgressBar::chunk {{
                    background-color: {self.colors['primary']};
                }}
                """

# Global theme instance
theme_manager = ThemeManager()

# Convenience functions
def get_theme_color(color_name: str) -> str:
    """Get a color from the current theme"""
    return theme_manager.get_color(color_name)

def get_stylesheet() -> str:
    """Get the current theme stylesheet"""
    return theme_manager.get_stylesheet()

def set_theme(theme_name: str):
    """Set the current theme"""
    theme_manager.set_theme(theme_name)