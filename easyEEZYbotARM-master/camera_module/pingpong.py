import cv2
import math

# 웹캠 켜기 (0번 또는 1번 카메라)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # [조건 1] 크기가 1,000 ~ 50,000 사이인 것만 취급 (너무 크거나 작은 것 무시)
        if 1000 < area < 50000: 
            
            # 물체를 감싸는 네모 박스 구하기
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # [조건 2] 가로세로 비율이 0.7 ~ 1.3 사이 (정사각형/원형에 가까운지)
            if 0.7 < aspect_ratio < 1.3:
                
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = (4 * math.pi * area) / (perimeter ** 2)
                
                # [조건 3] 찐 탁구공 판별!
                if circularity > 0.85:
                    # 완전 둥글면 정상
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                    cv2.putText(frame, "Normal", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                elif 0.6 < circularity <= 0.85:
                    # 살짝 덜 둥글면 찌그러진 공 (원형도 0.6 이상 0.85 이하)
                    cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
                    cv2.putText(frame, "Damaged", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                else:
                    # 원형도가 0.6 이하로 너무 낮으면 탁구공이 아님 (무시)
                    pass 

    cv2.imshow("Smart Ball Check", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()