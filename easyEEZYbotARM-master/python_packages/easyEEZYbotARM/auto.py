import cv2
import numpy as np
import math
import time
import serial 

from db_manager import DBManager

# ==========================================
# 1. 아두이노 시리얼 통신 설정
# ==========================================
try:
    ser = serial.Serial('COM3', 9600, timeout=1) 
    time.sleep(2) 
except Exception as e:
    print(f"Serial Connect Error: {e}")
    ser = None

# ==========================================
# 2. 3x3 그리드 룩업 테이블
# ==========================================
lookup_table = {
    (0, 0): (0, 45, 120, 90), (0, 1): (0, 90, 120, 90), (0, 2): (0, 135, 120, 90),
    (1, 0): (0, 45, 90, 110), (1, 1): (0, 90, 90, 110), (1, 2): (0, 135, 90, 110),
    (2, 0): (0, 45, 60, 130), (2, 1): (0, 90, 60, 130), (2, 2): (0, 135, 60, 130)
}

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    if ser:
        command = f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},{move_time},{move_time},{move_time},{move_time}>"
        ser.write(command.encode())
        print(f"Sent: {command}")

cap = cv2.VideoCapture(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
grid_w = frame_width // 3
grid_h = frame_height // 3

db = DBManager(r'C:\Team-Robot_Arm\easyEEZYbotARM-master\python_packages\robot_arm.db')
last_log_time = 0
cooldown_seconds = 3.0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.line(frame, (grid_w, 0), (grid_w, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (grid_w*2, 0), (grid_w*2, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h), (frame_width, grid_h), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h*2), (frame_width, grid_h*2), (255, 255, 0), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        if 1000 < area < 40000 and 0.5 < aspect_ratio < 1.5:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perfect_circle_area = math.pi * (radius ** 2)
            if perfect_circle_area == 0: continue
            
            circle_ratio = area / perfect_circle_area
            current_time = time.time()
            
            # --- 상태 판별 로직 시작 ---
            is_defective = False
            defect_msg = ""
            
            if circle_ratio > 0.95:
                # 모양은 정상이어도 표면이 찌그러졌는지 확인
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.6), 255, -1)
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))
                
                if internal_edge_pixels > 15:
                    is_defective = True
                    defect_msg = "Top Dent" # 표면 흠집 불량
            else:
                is_defective = True
                defect_msg = f"Shape ({circle_ratio:.2f})" # 외곽 찌그러짐 불량
            # --- 상태 판별 로직 끝 ---


            # --- 실행 로직 ---
            row = min(int(cy) // grid_h, 2)
            col = min(int(cx) // grid_w, 2)

            if is_defective:
                # 🔴 불량품(Damaged)일 때 작동: 화면 표시, DB 기록, 로봇팔 이동
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(frame, f"Damaged: {defect_msg}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Damaged", "B")
                    
                    # 로봇팔 이동 명령 전송 (불량품 픽업)
                    target_angles = lookup_table.get((row, col))
                    if target_angles:
                        ee, j1, j2, j3 = target_angles
                        send_robot_command("M", ee, j1, j2, j3)
                        
                    last_log_time = current_time
                    
            else:
                # 🟢 양품(Normal)일 때 작동: 화면 표시, DB 기록 (로봇팔은 안 움직임)
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(frame, "Normal Pass", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Normal", "A")
                    last_log_time = current_time

    cv2.imshow("Smart Ball Check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
db.close()
if ser:
    ser.close()