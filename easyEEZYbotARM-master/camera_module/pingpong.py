import cv2
import numpy as np
import math
import sys
import os
import time

# python_packages 폴더의 db_manager 모듈 임포트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python_packages')))
from db_manager import DBManager

cap = cv2.VideoCapture(1) # 애니캠

# DB 객체 생성 (데이터베이스 파일은 최상위 폴더에 생성됨)
db = DBManager('../robot_arm.db')

last_log_time = 0
cooldown_seconds = 3.0 # 동일 객체 인식 후 3초간 DB 저장 무시

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    
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
            
            # DB 저장 로직 (쿨다운 적용)
            current_time = time.time()
            
            if circle_ratio > 0.95:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(frame, f"Normal ({circle_ratio:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Normal", "A")
                    last_log_time = current_time
            else:
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(frame, f"Damaged ({circle_ratio:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Damaged", "B")
                    last_log_time = current_time

    cv2.imshow("Smart Ball Check", frame)
    cv2.imshow("Computer Vision", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
db.close() # 프로그램 종료 시 DB 연결 해제