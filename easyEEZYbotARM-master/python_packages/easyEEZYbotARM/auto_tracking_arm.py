import cv2
import numpy as np
import math
import time

# 기존 클래스 임포트
from kinematic_model import EEZYbotARM_Mk2
from serial_communication import arduinoController

# --- 1. 환경 설정 ---
PORT = "COM3"          # 아두이노 연결 포트 (상황에 맞게 수정) 
BAUD_RATE = 9600        # 시리얼 통신 속도 [cite: 2, 3]
PIXEL_TO_MM = 0.5       # 1픽셀당 실제 거리(mm) 변환 비율 (실측 후 수정 필요)
Z_TARGET = 50           # 탁구공이 놓인 바닥의 z 좌표 (mm) 
CAMERA_OFFSET_X = 150   # 로봇 베이스에서 카메라까지의 x축 거리 (mm)

# --- 2. 객체 초기화 ---
# 로봇 모델 (Mk2) 초기화 
arm = EEZYbotARM_Mk2(initial_q1=0, initial_q2=90, initial_q3=-90) 

# 아두이노 컨트롤러 초기화 및 연결 
controller = arduinoController(port=PORT)
try:
    controller.openSerialPort(baudRate=BAUD_RATE)
    print("아두이노 연결 성공!")
except Exception as e:
    print(f"연결 실패: {e}. 시뮬레이션 모드로 실행합니다.")

# 카메라 캡처 시작 
cap = cv2.VideoCapture(0)
image_center_x = 320  # 640x480 해상도 기준 중앙값

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- 3. 영상 처리 (pingpong.py 로직 활용)  ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 101, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1000 < area < 40000:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # 원형도 검사 
            if 0.5 < aspect_ratio < 1.5:
                # --- 4. 좌표 변환 (Mapping) ---
                # 화면 좌표(pixel)를 로봇 좌표(mm)로 변환
                # 로봇의 x축은 앞뒤, y축은 좌우임 
                target_y_mm = (cx - image_center_x) * PIXEL_TO_MM
                target_x_mm = CAMERA_OFFSET_X + (cy * PIXEL_TO_MM) # 카메라 각도에 따라 조정 필요
                
                # --- 5. 운동학 및 제어 명령 ---
                try:
                    # 1단계: 역기능학으로 관절 각도 계산 
                    q1, q2, q3 = arm.inverseKinematics(target_x_mm, target_y_mm, Z_TARGET)
                    
                    # 2단계: 물리적 서보 각도로 매핑 
                    s1, s2, s3 = arm.map_kinematicsToServoAngles(q1=q1, q2=q2, q3=q3)
                    
                    # 3단계: 아두이노 패킷 생성 및 전송 
                    # <MOVE, gripper, q1, q2, q3, t_ee, t1, t2, t3> [cite: 7]
                    msg = controller.composeMessage(s1, s2, s3, servoAngle_EE=90, instruction="MOVE")
                    controller.sendToArduino(msg)
                    
                    # 시각화
                    cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                    cv2.putText(frame, f"Move to: {int(target_x_mm)}, {int(target_y_mm)}", 
                                (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                except Exception as e:
                    # 범위를 벗어나는 등의 오류 시 처리 
                    cv2.putText(frame, "Out of Range", (int(x), int(y)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Auto Tracking System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
if hasattr(controller, 'serialPort'):
    controller.closeSerialPort() [cite: 2]