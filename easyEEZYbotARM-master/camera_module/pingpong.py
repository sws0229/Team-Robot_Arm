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
    
    # 1. [업데이트] 그림자 방어를 위한 적응형 이진화
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        101, 2
    )
    
    # 2. [추가] 표면 굴곡(찌러짐 선) 감지용 Canny 에지
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
            
            # [1단계 필터] 외곽선이 완벽한 원형에 가까운가?
            if circle_ratio > 0.95:
                
                # [2단계 필터] 공 안쪽 윗면 찌러짐 추가 검사
                cx, cy, radius = int(cx), int(cy), int(radius)
                
                # 외곽선 노이즈 제외를 위해 반지름을 60%로 줄인 내부 마스크 생성
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (cx, cy), int(radius * 0.6), 255, -1)
                
                # 내부 마스크 영역 안의 에지 픽셀 개수 계산
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))
                
                # 내부 에지가 기준치(15)보다 많으면 윗면 찌그러짐(불량) 처리
                if internal_edge_pixels > 15:
                    cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                    cv2.putText(frame, "Damaged (Top Dent)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # DB 저장 (불량 로깅)
                    if current_time - last_log_time > cooldown_seconds:
                        db.insert_log("Damaged", "B")
                        last_log_time = current_time
                        
                else:
                    # 외곽선도 깨끗하고 안쪽 표면도 매끄러운 진짜 정상 공
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                    cv2.putText(frame, f"Normal ({circle_ratio:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # DB 저장 (정상 로깅)
                    if current_time - last_log_time > cooldown_seconds:
                        db.insert_log("Normal", "A")
                        last_log_time = current_time
            else:
                # 외곽 테두리 자체가 타원형이거나 일그러진 불량 공
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(frame, f"Damaged ({circle_ratio:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # DB 저장 (불량 로깅)
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Damaged", "B")
                    last_log_time = current_time

    cv2.imshow("Smart Ball Check", frame)
    cv2.imshow("Computer Vision", thresh)
    cv2.imshow("Surface Edges", edges) # 표면 에지 디버깅용 창 추가

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
db.close() # 프로그램 종료 시 DB 연결 해제