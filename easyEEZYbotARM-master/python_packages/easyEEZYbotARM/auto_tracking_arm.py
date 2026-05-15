import cv2
import numpy as np
import time
import math # math 모듈 임포트 추가 (원넓이 계산용)
from kinematic_model import EEZYbotARM_Mk2
from serial_communication import arduinoController
from db_manager import DBManager # DB 관리 파일 임포트

# --- 1. 환경 설정 ---
PORT = "COM3"
BAUD_RATE = 9600
PIXEL_TO_MM = 0.4       
Z_FLOOR = 40            
CAM_X_OFFSET = 180      

# --- 2. 초기화 ---
arm = EEZYbotARM_Mk2(0, 90, -90) #
controller = arduinoController(port=PORT) #
db = DBManager('robot_arm.db') # DB 객체 생성

try:
    controller.openSerialPort(baudRate=BAUD_RATE)
except:
    print("아두이노 연결 실패 - 시뮬레이션 모드")

cap = cv2.VideoCapture(1)
IMG_CENTER_X, IMG_CENTER_Y = 320, 240
last_action_time = 0
cooldown = 2.0 # 동일 공 중복 기록 방지 (초)

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- 3. 영상 분석 (찌그러짐 감지 로직 포함) ---
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
        
        if 500 < area < 40000:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius = int(cx), int(cy), int(radius)

            perfect_circle_area = math.pi * (radius ** 2)
            if perfect_circle_area == 0: continue
            circle_ratio = area / perfect_circle_area
            
            
            if circle_ratio > 0.80:
                
                # 원형도 및 내부 찌그러짐 검사
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (cx, cy), int(radius * 0.6), 255, -1)
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))
                
                # 상태 판별
                is_damaged = internal_edge_pixels > 15
                status = "Damaged" if is_damaged else "Normal"
                color = (0, 0, 255) if is_damaged else (0, 255, 0)
                
                # --- 4. 좌표 변환 및 이동 ---
                target_x = CAM_X_OFFSET + (IMG_CENTER_Y - cy) * PIXEL_TO_MM
                target_y = (cx - IMG_CENTER_X) * PIXEL_TO_MM

                current_time = time.time()
                if current_time - last_action_time > cooldown:
                    try:
                        # 1) 로봇 이동 명령
                        q1, q2, q3 = arm.inverseKinematics(target_x, target_y, Z_FLOOR)
                        s1, s2, s3 = arm.map_kinematicsToServoAngles(q1=q1, q2=q2, q3=q3)
                        msg = controller.composeMessage(s1, s2, s3, servoAngle_EE=90)
                        controller.sendToArduino(msg)
                        
                        # 2) DB에 결과 저장
                        action_type = "B" if is_damaged else "A"
                        db.insert_log(status, action_type)
                        
                        last_action_time = current_time
                        print(f"[{status}] 로그 저장 및 로봇 이동: X={int(target_x)}, Y={int(target_y)}")
                    except:
                        pass

                # 시각화 데이터 출력
                cv2.drawContours(frame, [cnt], -1, color, 2)
                cv2.putText(frame, f"{status} (Edges:{internal_edge_pixels})", (cx-50, cy-radius-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Smart Sorting System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
db.close()