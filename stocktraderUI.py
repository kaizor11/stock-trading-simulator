import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, 
                             QPushButton, QVBoxLayout,QLineEdit,
                             QWidget, QFileDialog,QMessageBox, 
                             QTableWidget, QTableWidgetItem, QLabel)

import sys

#import trading rules here
from mainwave import MainWaveDetector

class StockApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.df = None
        #UI instances
        self.setWindowTitle("ui TEST with size of 800 400") #title
        self.resize(800,400)          #size

        # this is the central part of the window 
        self.central_widget = QWidget ()           
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)


        self.label = QLabel("lable test")
        self.layout.addWidget(self.label)

        self.btn_load = QPushButton("main wave test")
        self.btn_load.clicked.connect(self.call_main_wave) 
        self.layout.addWidget(self.btn_load)

        self.btn_load_file = QPushButton("Load CSV")
        self.layout.addWidget(self.btn_load_file)
        self.btn_load_file.clicked.connect(self.load_csv)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)
        self.input_stock = QLineEdit()
        self.input_stock.setPlaceholderText("Enter stock code")
        self.layout.addWidget(self.input_stock)
        stock_code = self.input_stock.text()
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.label.setText(f"Loaded: {file_path}")
        self.show_df_in_table(self.df)

    def call_main_wave(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load a CSV first.")
            return

        try:
            detector = MainWaveDetector(self.df)
            result = detector.detect_main_wave(strict=True)
        except Exception as e:
            QMessageBox.critical(self, "MainWave Error", str(e))
            return

        
        print("called mainwave.py")
        self.label.setText("MainWave done. ")

    def show_df_in_table(self, df: pd.DataFrame):
        
        max_rows = min(len(df), 5000)
        max_cols = min(len(df.columns), 50)
        view = df.iloc[:max_rows, :max_cols]

        self.table.clear()
        self.table.setRowCount(view.shape[0])
        self.table.setColumnCount(view.shape[1])
        self.table.setHorizontalHeaderLabels([str(c) for c in view.columns])

        for r in range(view.shape[0]):
            for c in range(view.shape[1]):
                item = QTableWidgetItem(str(view.iat[r, c]))
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = StockApp()
    window.show()
    sys.exit(app.exec())