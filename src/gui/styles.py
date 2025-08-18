# src/gui/styles.py

# A professional and modern color palette
PALETTE = {
    "dark_primary": "#1a202c",    # Very dark (mostly black)
    "primary": "#2d3748",         # Dark
    "secondary": "#4a5568",       # Medium-dark
    "light_gray": "#a0aec0",      # Light Gray (for borders, secondary text)
    "white": "#ffffff",
    "accent_blue": "#3182ce",     # A strong blue for highlights
    "accent_green": "#38a169",    # A strong green for success
    "accent_red": "#e53e3e",      # A strong red for errors/danger
    "accent_yellow": "#dd6b20",   # A strong orange/yellow for warnings
    "challenger_color": "#2f855a", # A rich green for the challenger
    "baseline_color": "#2c5282",  # A deep blue for baselines
}

def get_stylesheet():
    """
    Returns the global stylesheet for the application.
    """
    return f"""
        QMainWindow, QDialog {{
            background-color: {PALETTE['primary']};
            color: {PALETTE['white']};
        }}
        QWidget {{
            background-color: {PALETTE['primary']};
            color: {PALETTE['white']};
            font-family: "Segoe UI", "Arial", sans-serif;
        }}
        QGroupBox {{
            background-color: {PALETTE['secondary']};
            border: 1px solid {PALETTE['light_gray']};
            border-radius: 5px;
            margin-top: 1ex; /* leave space at the top for the title */
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            left: 10px;
        }}
        QLabel {{
            color: {PALETTE['white']};
            font-size: 10pt;
        }}
        QLabel[cssClass="title"] {{
            font-size: 16pt;
            font-weight: bold;
            color: {PALETTE['white']};
        }}
        QPushButton {{
            background-color: {PALETTE['accent_blue']};
            color: {PALETTE['white']};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 10pt;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: #2b6cb0; /* A slightly darker blue */
        }}
        QPushButton:pressed {{
            background-color: #2c5282; /* An even darker blue */
        }}
        QPushButton:disabled {{
            background-color: {PALETTE['secondary']};
            color: {PALETTE['light_gray']};
        }}
        QLineEdit, QTextEdit, QComboBox {{
            background-color: {PALETTE['dark_primary']};
            color: {PALETTE['white']};
            border: 1px solid {PALETTE['light_gray']};
            border-radius: 4px;
            padding: 5px;
        }}
        QTableWidget {{
            background-color: {PALETTE['dark_primary']};
            color: {PALETTE['white']};
            gridline-color: {PALETTE['secondary']};
            border: 1px solid {PALETTE['light_gray']};
        }}
        QHeaderView::section {{
            background-color: {PALETTE['secondary']};
            color: {PALETTE['white']};
            padding: 4px;
            border: 1px solid {PALETTE['light_gray']};
            font-weight: bold;
        }}
        QTableWidget::item {{
            padding: 5px;
        }}
        QTableWidget::item:selected {{
            background-color: {PALETTE['accent_blue']};
            color: {PALETTE['white']};
        }}
        QSplitter::handle {{
            background-color: {PALETTE['secondary']};
            border: 1px solid {PALETTE['light_gray']};
        }}
        QSplitter::handle:horizontal {{
            width: 1px;
        }}
        QSplitter::handle:vertical {{
            height: 1px;
        }}
        QMessageBox {{
            background-color: {PALETTE['secondary']};
        }}
    """
