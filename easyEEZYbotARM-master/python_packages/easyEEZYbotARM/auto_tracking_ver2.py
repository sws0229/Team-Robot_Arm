import cv2
import numpy as np
import math
import time
import serial
import sys
import os

# ==========================================
# 0. 경로 설정 및 DB 임포트 
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, '..'))
sys.path.append(parent_dir)
from db_manager import DBManager

db_path = os.path.join(parent_dir, 'robot_arm.db')
db = DBManager(db_path)

# ==========================================q
# 1. 아두이노 시리얼 통신 설정
# ==========================================
try:
    ser = serial.Serial('COM3', 9600, timeout=1) 
    time.sleep(2) 
    print("아두이노 통신 연결 성공!")
except Exception as e:
    print(f"아두이노 연결 실패 (시뮬레이션 모드): {e}")
    ser = None

# ==========================================
# ==========================================
GRIPPER_OPEN = 90    
GRIPPER_CLOSE = 150  
HOVER_J2 = 120       
HOVER_J3 = 90        
DISCARD_J1 = 180     

lookup_table = {
    (0, 0): (0, 45, 120, 90), (0, 1): (0, 90, 120, 90), (0, 2): (0, 135, 120, 90),
    (1, 0): (0, 45, 90, 110), (1, 1): (0, 90, 90, 110), (1, 2): (0, 135, 90, 110),
    (2, 0): (0, 45, 60, 130), (2, 1): (0, 90, 200, 40), (2, 2): (0, 135, 60, 130)
}

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    if ser:
        command = f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},{move_time},{move_time},{move_time},{move_time}>"
        ser.write(command.encode())
        print(f"명령 전송됨: {command}")

# ==========================================
# 3. 자동 수거(Pick and Discard) 시퀀스 
# ==========================================
def pick_and_discard(row, col):
    target_angles = lookup_table.get((row, col))
    if not target_angles: return
    
    _, j1, j2_down, j3_down = target_angles 
    print(f"[{row},{col}] 구역 불량품 수거 시작!")
    
    send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    
    send_robot_command("M", GRIPPER_OPEN, j1, j2_down, j3_down, 1000)
    time.sleep(1.2)
    
    send_robot_command("M", GRIPPER_CLOSE, j1, j2_down, j3_down, 500)
    time.sleep(0.8)
    
    send_robot_command("M", GRIPPER_CLOSE, j1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    
    send_robot_command("M", GRIPPER_CLOSE, DISCARD_J1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    
    send_robot_command("M", GRIPPER_OPEN, DISCARD_J1, HOVER_J2, HOVER_J3, 500)
    time.sleep(0.8)
    
    print("불량품 폐기 완료! 대기 상태로 복귀.")

# ==========================================
# 4. 메인 비전 시스템 (민혁님 세팅: 3중 필터 + 흑백 공 인식)
# ==========================================
cap = cv2.VideoCapture(1) # 카메라 인덱스 0 유지
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
grid_w = frame_width // 3
grid_h = frame_height // 3

last_log_time = 0
cooldown_seconds = 8.0 # 수거 동작 시간(약 6~7초)을 고려한 넉넉한 쿨타임

print("스마트 품질 검사 시스템 가동을 시작합니다...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.line(frame, (grid_w, 0), (grid_w, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (grid_w*2, 0), (grid_w*2, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h), (frame_width, grid_h), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h*2), (frame_width, grid_h*2), (255, 255, 0), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0) 
    
    

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
        
        # 1단계 철벽: 탁구공 크기 및 비율 (범위 조정됨)
        if 2000 < area < 15000 and 0.5 < aspect_ratio < 1.5:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perfect_circle_area = math.pi * (radius ** 2)
            if perfect_circle_area == 0: continue
            
            circle_ratio = area / perfect_circle_area
            
            # 2단계 철벽: 너무 찌그러진 잡동사니 무시
            if circle_ratio < 0.60: continue 
                
            current_time = time.time()
            is_defective = False
            defect_msg = ""
            
            # 3단계: 하이브리드 상태 판별 (원형도 0.91 기준)
            if circle_ratio > 0.93:
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.65), 255, -1)
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))
                
                # 표면 흠집 검출 (빛 반사 완화를 위해 15px로 세팅)
                if internal_edge_pixels > 15:
                    is_defective = True
                    defect_msg = f"Dent ({internal_edge_pixels}px)" 
            else:
                is_defective = True
                defect_msg = f"Shape ({circle_ratio:.2f})" 

            # 그리드 위치 계산
            row = min(int(cy) // grid_h, 2)
            col = min(int(cx) // grid_w, 2)

            if is_defective:
                # 🔴 불량품 처리 (수거 시퀀스 실행)
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(frame, f"Damaged: {defect_msg}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Damaged", "B")
                    pick_and_discard(row, col) 
                    last_log_time = time.time() # 쿨타임 초기화
                    
            else:
                # 🟢 양품 처리 (수거 안 함)
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(frame, "Normal Pass", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Normal", "A")
                    last_log_time = current_time

    cv2.imshow("Smart Quality Control System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
db.close()
if ser: ser.close()
print("시스템이 정상적으로 종료되었습니다.")