import cv2
import numpy as np
import math

cap = cv2.VideoCapture(1) # 애니캠

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
        
        # 1. 바닥 노이즈 완벽 차단! (너무 크거나, 너무 길쭉한 건 무시)
        aspect_ratio = float(w) / h
        if 1000 < area < 40000 and 0.5 < aspect_ratio < 1.5:
            
            # 2. 새로운 필살기: 물체를 감싸는 '가상의 완벽한 원' 찾기
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perfect_circle_area = math.pi * (radius ** 2)
            
            if perfect_circle_area == 0: continue
            
            # 3. 점수 계산: (실제 탁구공 면적 / 가상의 완벽한 원 면적)
            circle_ratio = area / perfect_circle_area
            
            # 4. 판별 (보통 정상 공은 0.85 이상, 눌린 공은 0.80 이하로 뚝 떨어집니다)
            if circle_ratio > 0.95:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(frame, f"Normal ({circle_ratio:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(frame, f"Damaged ({circle_ratio:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Smart Ball Check", frame)
    cv2.imshow("Computer Vision", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()