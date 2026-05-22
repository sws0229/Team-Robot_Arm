import cv2
import numpy as np
import math
import sys
import os
import time
from serial_communication import arduinoController

# python_packages 폴더의 db_manager 모듈 임포트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python_packages')))
from db_manager import DBManager

# --- 1. 환경 설정 ---
PORT = "COM3"
BAUD_RATE = 9600

DEADZONE_X = 25  
DEADZONE_Y = 25
STEP_X = 6.0     
STEP_Y = 5.0  
cooldown = 0.2   

# --- 2. 초기화 ---
controller = arduinoController(port=PORT)
db = DBManager('../robot_arm.db')

current_q1, current_q2, current_q3 = 90.0, 90.0, 90.0  

try:
    controller.openSerialPort(baudRate=BAUD_RATE)
    print("아두이노 연결 성공! 명암(이진화) 추적 모드 가동!")
except Exception as e:
    print(f"아두이노 연결 실패 - 시뮬레이션 모드: {e}")

cap = cv2.VideoCapture(1) # 카메라 번호 주의 (1번)
IMG_CENTER_X, IMG_CENTER_Y = 320, 240
last_action_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- 3. 영상 분석ㄱ ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # 그림자 방어용 적응형 이진화
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 101, 2)
    
    # 표면 굴곡(찌그러짐) 감지용 에지 추출
    edges = cv2.Canny(blurred, 30, 100)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 크기 기준
        if 500 < area < 40000:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius = int(cx), int(cy), int(radius)

            perfect_circle_area = math.pi * (radius ** 2)
            if perfect_circle_area == 0: continue
            circle_ratio = area / perfect_circle_area
            
            # 원형도 기준 
            if circle_ratio > 0.80:
                
                # 원형도 및 내부 찌그러짐 검사
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (cx, cy), int(radius * 0.6), 255, -1)
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))
                
                is_damaged = internal_edge_pixels > 15
                status = "Damaged" if is_damaged else "Normal"
                color = (0, 0, 255) if is_damaged else (0, 255, 0)
                
                # --- 4. 상대 제어 로봇 무빙  ---
                error_x = cx - IMG_CENTER_X
                error_y = cy - IMG_CENTER_Y
                
                current_time = time.time()
                if current_time - last_action_time > cooldown:
                    
                    if error_x > DEADZONE_X: current_q1 -= STEP_X
                    elif error_x < -DEADZONE_X: current_q1 += STEP_X
                        
                    if error_y > DEADZONE_Y: current_q2 += STEP_Y
                    elif error_y < -DEADZONE_Y: current_q2 -= STEP_Y
                    
                    current_q1 = max(0, min(180, current_q1))
                    current_q2 = max(0, min(180, current_q2))
                    
                    try:
                        msg = controller.composeMessage(current_q1, current_q2, current_q3, servoAngle_EE=90, instruction="MOVE")
                        controller.sendToArduino(msg)
                        
                        action_type = "B" if is_damaged else "A"
                        db.insert_log(status, action_type)
                        last_action_time = current_time
                    except Exception as e:
                        pass 
                
                # 시각화: 테두리 그리기
                cv2.drawContours(frame, [cnt], -1, color, 2)
                cv2.putText(frame, f"{status}", (cx-50, cy-radius-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                break # 공 하나 찾으면 집중

    
    cv2.imshow("Smart Sorting System (MinHyeok)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
db.close()