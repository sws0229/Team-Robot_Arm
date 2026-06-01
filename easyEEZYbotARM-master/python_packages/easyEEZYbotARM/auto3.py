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
    # 아두이노가 연결된 포트 번호 ('COM3') 확인 및 변경 필요
    ser = serial.Serial('COM3', 9600, timeout=1) 
    time.sleep(2) # 시리얼 연결 안정화를 위한 대기 시간
except Exception as e:
    print(f"Serial Connect Error: {e}")
    ser = None

# ==========================================
# 2. 로봇팔 제어 파라미터 및 룩업 테이블 (4축 EEZYbotARM MK2 기준)
# ==========================================
GRIPPER_OPEN = 90    # 탁구공을 놓기 위해 집게를 여는 각도
GRIPPER_CLOSE = 150  # 탁구공을 꽉 잡기 위해 집게를 닫는 각도
HOVER_J2 = 120       # 공 위로 이동하거나 버리러 갈 때 어깨(J2)를 띄워두는 각도 (높이)
HOVER_J3 = 90        # 공 위로 이동하거나 버리러 갈 때 팔꿈치(J3)를 띄워두는 각도 (높이)

DISCARD_J1 = 180         # 불량품을 버릴 폐기함의 위치 (베이스 J1 회전 각도 - 예: 우측)
NORMAL_DISCARD_J1 = 0    # 정상품을 버릴 분류함의 위치 (베이스 J1 회전 각도 - 예: 좌측)

# 화면을 3x3으로 나눈 9개 구역의 좌표(row, col)와 모터 각도 매핑
# 형태: (임시_ee, j1_회전각도, j2_하강각도, j3_하강각도)
lookup_table = {
    (0, 0): (160,70,200,70), (0, 1): (160,90,200,70), (0, 2): (160,110,200,70),
    (1, 0): (160,70,180,50), (1, 1): (160,90,180,50), (1, 2): (160,110,180,50),
    (2, 0): (160,70,160,30), (2, 1): (160,90,160,30), (2, 2): (160,110,160,30)
}

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    """아두이노로 모터 각도와 이동 시간(ms)을 포함한 프로토콜 문자열 전송"""
    if ser:
        command = f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},{move_time},{move_time},{move_time},{move_time}>"
        ser.write(command.encode())
        print(f"Sent: {command}")

# ==========================================
# 3. 탁구공 분류 (Pick and Place) 자동화 시퀀스
# ==========================================
def pick_and_discard(row, col, target_discard_angle):
    """지정된 그리드 좌표에서 탁구공을 집어 지정된 각도의 목적지에 버리는 연속 동작"""
    target_angles = lookup_table.get((row, col))
    if not target_angles: return
    
    _, j1, j2_down, j3_down = target_angles 
    
    print(f"[{row},{col}] 수거 시퀀스 시작 -> 목표 각도: {target_discard_angle}")
    
    # [1단계] 목표 상공으로 이동: 집게 열고, Z축은 띄운 상태로(HOVER) 이동
    send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    
    # [2단계] 탁구공 위치로 하강: 테이블에 저장된 Z축 각도(down)로 내려감
    send_robot_command("M", GRIPPER_OPEN, j1, j2_down, j3_down, 1000)
    time.sleep(1.2)
    
    # [3단계] 집게 닫기: 공을 잡음
    send_robot_command("M", GRIPPER_CLOSE, j1, j2_down, j3_down, 500)
    time.sleep(0.8)
    
    # [4단계] 다시 상공으로 들어올리기: 바닥에 긁히지 않게 Z축 상승
    send_robot_command("M", GRIPPER_CLOSE, j1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    
    # [5단계] 지정된 목적지 장소로 회전: 전달받은 target_discard_angle 각도로 회전 이동
    send_robot_command("M", GRIPPER_CLOSE, target_discard_angle, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    
    # [6단계] 집게 열기: 공을 떨어뜨림
    send_robot_command("M", GRIPPER_OPEN, target_discard_angle, HOVER_J2, HOVER_J3, 500)
    time.sleep(0.8)
    
    print("수거 완료 및 대기 상태 복귀")

# ==========================================
# 4. 메인 비전 시스템 (웹캠 및 영상 처리)
# ==========================================
# [설치 주의] 3D 물리 공간과 2D 화면 좌표를 일치시키기 위해 
# 카메라는 반드시 탁구공 작업 공간의 수직 위(Top-down)에 설치해야 합니다.
cap = cv2.VideoCapture(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
grid_w = frame_width // 3
grid_h = frame_height // 3

# 로그 저장을 위한 DB 관리자 객체 생성
db = DBManager(r'C:\project\Team-Robot_Arm\easyEEZYbotARM-master\python_packages\robot_arm.db')

last_log_time = 0
cooldown_seconds = 8.0 

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, -1)
    
    cv2.line(frame, (grid_w, 0), (grid_w, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (grid_w*2, 0), (grid_w*2, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h), (frame_width, grid_h), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h*2), (frame_width, grid_h*2), (255, 255, 0), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 4)
    
    # [수정됨] 조명 및 그림자로 인한 흠집 오인식 방지를 위해 Canny 임계값 상향 조정 (30,100 -> 50,150)
    edges = cv2.Canny(blurred, 50, 150) 
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        if 3000 < area < 40000 and 0.85 < aspect_ratio < 1.15:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perfect_circle_area = math.pi * (radius ** 2)
            if perfect_circle_area == 0: continue
            
            circle_ratio = area / perfect_circle_area
            current_time = time.time()
            
            is_defective = False
            defect_msg = ""
            
            if circle_ratio > 0.95:
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.6), 255, -1)
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))
                
                if internal_edge_pixels > 15:
                    is_defective = True
                    defect_msg = "Top Dent" 
            else:
                is_defective = True
                defect_msg = f"Shape ({circle_ratio:.2f})" 

            row = min(int(cy) // grid_h, 2)
            col = min(int(cx) // grid_w, 2)

            if is_defective:
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3) 
                cv2.putText(frame, f"Damaged: {defect_msg}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Damaged", "B")
                    pick_and_discard(row, col, DISCARD_J1)
                    
                    # [수정됨] 로봇팔 동작 중(sleep) 카메라 버퍼에 쌓인 과거 프레임 강제 비우기 (화면 밀림 현상 방지)
                    for _ in range(10): 
                        cap.read()
                        
                    last_log_time = time.time()
                    
            else:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(frame, "Normal Pass", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Normal", "A")
                    pick_and_discard(row, col, NORMAL_DISCARD_J1)
                    
                    # [수정됨] 로봇팔 동작 중(sleep) 카메라 버퍼에 쌓인 과거 프레임 강제 비우기 (화면 밀림 현상 방지)
                    for _ in range(10): 
                        cap.read()
                        
                    last_log_time = time.time()

    cv2.imshow("Smart Ball Check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# ==========================================
# 5. 자원 해제 및 프로그램 종료
# ==========================================
cap.release()
cv2.destroyAllWindows()
db.close()
if ser:
    ser.close()