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

# ==========================================
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
# 2. 로봇팔 제어 파라미터 및 룩업 테이블
# ==========================================
GRIPPER_OPEN = 90
GRIPPER_CLOSE = 150
HOVER_J2 = 120
HOVER_J3 = 90

# 분류 위치 설정 (J1 베이스 회전 각도)
DISCARD_RIGHT_J1 = 180
NORMAL_LEFT_J1 = 0

lookup_table = {
    (0, 0): (160, 30, 200, 70), (0, 1): (160, 90, 200, 60), (0, 2): (160, 110, 200, 70),
    (1, 0): (160, 30, 180, 50), (1, 1): (160, 90, 180, 40), (1, 2): (160, 110, 180, 50),
    (2, 0): (160, 30, 160, 30), (2, 1): (160, 90, 160, 20), (2, 2): (160, 110, 160, 30)
}

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    if ser:
        command = f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},{move_time},{move_time},{move_time},{move_time}>"
        ser.write(command.encode())
        print(f"명령 전송됨: {command}")

# ==========================================
# 3. 양방향 분류 (Pick and Place) 자동화 시퀀스
# ==========================================
def pick_and_place(row, col, is_defective):
    target_angles = lookup_table.get((row, col))
    if not target_angles: return

    _, j1, j2_down, j3_down = target_angles

    if is_defective:
        drop_j1 = DISCARD_RIGHT_J1
        print(f"[{row},{col}] 불량품 수거 시작 -> 오른쪽({drop_j1}도)으로 폐기!")
    else:
        drop_j1 = NORMAL_LEFT_J1
        print(f"[{row},{col}] 정상품 수거 시작 -> 왼쪽({drop_j1}도)으로 이동!")

    send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    send_robot_command("M", GRIPPER_OPEN, j1, j2_down, j3_down, 1000)
    time.sleep(1.2)
    send_robot_command("M", GRIPPER_CLOSE, j1, j2_down, j3_down, 500)
    time.sleep(0.8)
    send_robot_command("M", GRIPPER_CLOSE, j1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    send_robot_command("M", GRIPPER_CLOSE, drop_j1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    send_robot_command("M", GRIPPER_OPEN, drop_j1, HOVER_J2, HOVER_J3, 500)
    time.sleep(0.8)

    print("분류 작업 완료! 대기 상태로 복귀.")

# ==========================================
# 4. 메인 비전 시스템 
# ==========================================
cap = cv2.VideoCapture(1) 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
grid_w = frame_width // 3
grid_h = frame_height // 3

print("스마트 품질 검사 시스템 가동을 시작합니다...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 카메라 상하좌우 반전
    frame = cv2.flip(frame, -1)

    cv2.line(frame, (grid_w, 0), (grid_w, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (grid_w*2, 0), (grid_w*2, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h), (frame_width, grid_h), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h*2), (frame_width, grid_h*2), (255, 255, 0), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 대비 극대화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)

    # 2. 외곽선 형태 파악용 (강한 블러 + 원본 gray 사용)
    blurred_heavy = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh = cv2.adaptiveThreshold(blurred_heavy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
    
    # 3. 노이즈 제거 (형태 깔끔하게)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    #  4. 내부 흠집 파악용 
    blurred_light = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    
    # Canny 예민도를 한계까지 낮춤 
    edges = cv2.Canny(blurred_light, 40, 100) 

    
    cv2.imshow("X-Ray (Edges)", edges)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        if 2000 < area < 15000 and 0.5 < aspect_ratio < 1.5:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perfect_circle_area = math.pi * (radius ** 2)
            if perfect_circle_area == 0: continue

            circle_ratio = area / perfect_circle_area
            if circle_ratio < 0.60: continue

            is_defective = False
            defect_msg = ""

            if circle_ratio > 0.95:
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.65), 255, -1)
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))

                
                print(f" 스캔 중... 원형도: {circle_ratio:.2f} / 표면 흠집: {internal_edge_pixels}px")
                
                if internal_edge_pixels > 50:
                    is_defective = True
                    defect_msg = f"Dent ({internal_edge_pixels}px)"
            else:
                is_defective = True
                defect_msg = f"Shape ({circle_ratio:.2f})"

            row = min(int(cy) // grid_h, 2)
            col = min(int(cx) // grid_w, 2)

            if is_defective:
                #  불량품 인식
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(frame, f"Damaged: {defect_msg}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Smart Quality Control System", frame)
                cv2.waitKey(1)

                print(" 불량 탁구공 발견 1초 뒤 수거를 시작합니다...")
                time.sleep(1.0)

                db.insert_log("Damaged", "B")
                pick_and_place(row, col, is_defective=True)

                break
            else:
                #  정상품 인식
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(frame, "Normal Pass", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Smart Quality Control System", frame)
                cv2.waitKey(1)

                print(" 정상 탁구공 확인 1초 뒤 이동을 시작합니다...")
                time.sleep(1.0)

                db.insert_log("Normal", "A")
                pick_and_place(row, col, is_defective=False)

                break

    cv2.imshow("Smart Quality Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
db.close()
if ser: ser.close()
print("시스템이 정상적으로 종료되었습니다.")