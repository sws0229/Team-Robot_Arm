import sys
import serial
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QLabel)
from PySide6.QtCore import Qt

class ArmHangEosaKeyboardController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("키보드 제어")
        self.resize(500, 350)

        
        self.port = 'COM3'  
        self.baudrate = 9600
        self.serial_conn = None
        
    
        self.current_q1 = 90
        self.current_q2 = 90
        self.current_q3 = 90
        self.current_gripper = 90

        # UI 및 통신 시작
        self.init_ui()
        self.connect_arduino()

    def connect_arduino(self):
        """아두이노와 시리얼 통신을 연결합니다."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.update_status(f" 아두이노 연결 ({self.port})", "green")
        except Exception:
            self.update_status(f" 연결 실패 ({self.port}).", "red")

    def update_status(self, text, color):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 16px;")

    def init_ui(self):
        """키보드 입력을 위한 전용 UI 구성"""
        self.setFocusPolicy(Qt.StrongFocus) 
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)

        # 상단 타이틀
        header = QLabel("로봇팔 키보드 제어")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #333; margin-top: 10px;")
        main_layout.addWidget(header)

        # 상태창
        self.status_lbl = QLabel("연결 시도")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_lbl)

        # 실시간 각도 표시창
        self.angle_lbl = QLabel(f"Q1: {self.current_q1}° | Q2: {self.current_q2}° | Q3: {self.current_q3}° | EE: {self.current_gripper}°")
        self.angle_lbl.setAlignment(Qt.AlignCenter)
        self.angle_lbl.setStyleSheet("font-size: 18px; background-color: #2c3e50; color: white; padding: 15px; border-radius: 10px;")
        main_layout.addWidget(self.angle_lbl)

        # 하단 조작 가이드
        guide = QLabel(
            "   [ 키보드 조작 가이드 ]\n"
            "   - Q / A : q1 좌우 회전\n"
            "   - W / S : q2 오르내리기\n"
            "   - 방향키 위/아래 : q3 굽히기\n"
            "   - 방향키 좌/우 : 집게 열기/닫기"
        )
        guide.setStyleSheet("background-color: #ecf0f1; padding: 15px; border-radius: 5px; color: #2c3e50; line-height: 150%;")
        main_layout.addWidget(guide)

        self.log_lbl = QLabel("시스템 대기")
        self.log_lbl.setAlignment(Qt.AlignCenter)
        self.log_lbl.setStyleSheet("color: #888;")
        main_layout.addWidget(self.log_lbl)

    def keyPressEvent(self, event):
        """키보드 버튼 조작 (Direct Joint Control)"""
        step = 5 
        key = event.key()

        # 각도 계산 루직
        if key == Qt.Key_Q: self.current_q1 = min(180, self.current_q1 + step)
        elif key == Qt.Key_A: self.current_q1 = max(0, self.current_q1 - step)
        elif key == Qt.Key_W: self.current_q2 = min(180, self.current_q2 + step)
        elif key == Qt.Key_S: self.current_q2 = max(0, self.current_q2 - step)
        elif key == Qt.Key_Up: self.current_q3 = min(180, self.current_q3 + step)
        elif key == Qt.Key_Down: self.current_q3 = max(0, self.current_q3 - step)
        elif key == Qt.Key_Left: self.current_gripper = min(180, self.current_gripper + step)
        elif key == Qt.Key_Right: self.current_gripper = max(0, self.current_gripper - step)
        else: return

        # UI 업데이트
        self.angle_lbl.setText(f"Q1: {self.current_q1}° | Q2: {self.current_q2}° | Q3: {self.current_q3}° | EE: {self.current_gripper}°")
        
        # 데이터 전송
        self.send_to_arduino(self.current_gripper, self.current_q1, self.current_q2, self.current_q3)

    def send_to_arduino(self, g, q1, q2, q3):
        """아두이노 패킷 전송"""
        cmd = f"<MOVE,{g},{q1},{q2},{q3},500,500,500,500>\n"
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(cmd.encode())
            self.log_lbl.setText(f"데이터 전송 중: {cmd.strip()}")
        else:
            self.log_lbl.setText(f"시뮬레이션 모드: {cmd.strip()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ArmHangEosaKeyboardController()
    ex.show()
    sys.exit(app.exec())