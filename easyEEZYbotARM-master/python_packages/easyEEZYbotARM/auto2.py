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
    # [수정 포인트] 아두이노가 연결된 포트 번호 ('COM3') 확인 및 변경 필요
    ser = serial.Serial('COM3', 9600, timeout=1) 
    time.sleep(2) # 시리얼 연결 안정화를 위한 대기 시간
except Exception as e:
    print(f"Serial Connect Error: {e}")
    ser = None

# ==========================================
# 2. 로봇팔 제어 파라미터 및 룩업 테이블
# ==========================================
# [수정 포인트] 나중에 실제 조립 후 아래 상수들의 각도를 변경해 위치를 맞춥니다.
GRIPPER_OPEN = 90    # 탁구공을 놓기 위해 집게를 여는 각도
GRIPPER_CLOSE = 150   # 탁구공을 꽉 잡기 위해 집게를 닫는 각도
HOVER_J2 = 120       # 공 위로 이동하거나 버리러 갈 때 어깨(J2)를 띄워두는 각도 (높이)
HOVER_J3 = 90        # 공 위로 이동하거나 버리러 갈 때 팔꿈치(J3)를 띄워두는 각도 (높이)
DISCARD_J1 = 180     # 불량품을 버릴 폐기함의 위치 (베이스 J1의 좌우 회전 각도)

# [수정 포인트] 화면을 3x3으로 나눈 9개 구역의 좌표(row, col)와 모터 각도 매핑
# 형태: (임시_ee, j1_회전각도, j2_하강각도, j3_하강각도)
# 나중에 공을 집을 때의 Z축 높이를 조절하려면 3번째(j2)와 4번째(j3) 값을 수정하세요.
lookup_table = {
    (0, 0): (0, 45, 120, 90), (0, 1): (0, 90, 120, 90), (0, 2): (0, 135, 120, 90),
    (1, 0): (0, 45, 90, 110), (1, 1): (0, 90, 90, 110), (1, 2): (0, 135, 90, 110),
    (2, 0): (0, 45, 60, 130), (2, 1): (0, 90, 60, 130), (2, 2): (0, 135, 60, 130)
}

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    """아두이노로 모터 각도와 이동 시간(ms)을 포함한 프로토콜 문자열 전송"""
    if ser:
        command = f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},{move_time},{move_time},{move_time},{move_time}>"
        ser.write(command.encode())
        print(f"Sent: {command}")

# ==========================================
# 3. 불량품 수거 (Pick and Discard) 자동화 시퀀스
# ==========================================
def pick_and_discard(row, col):
    """지정된 그리드 좌표에서 탁구공을 집어 폐기함에 버리는 연속 동작"""
    target_angles = lookup_table.get((row, col))
    if not target_angles: return
    
    # 룩업 테이블에서 해당 구역의 베이스 회전(j1), 높이 하강 각도(j2_down, j3_down) 추출
    _, j1, j2_down, j3_down = target_angles 
    
    print(f"[{row},{col}] 불량품 수거 시퀀스 시작")
    
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
    
    # [5단계] 폐기 장소로 회전: DISCARD_J1 각도로 회전 이동
    send_robot_command("M", GRIPPER_CLOSE, DISCARD_J1, HOVER_J2, HOVER_J3, 1000)
    time.sleep(1.2)
    
    # [6단계] 집게 열기: 공을 떨어뜨림
    send_robot_command("M", GRIPPER_OPEN, DISCARD_J1, HOVER_J2, HOVER_J3, 500)
    time.sleep(0.8)
    
    print("수거 완료 및 대기 상태 복귀")

# ==========================================
# 4. 메인 비전 시스템 (웹캠 및 영상 처리)
# ==========================================
# 웹캠 켜기 (0은 기본 내장 캠, 1은 외장 USB 캠일 확률이 높음)
cap = cv2.VideoCapture(1)

# 카메라 해상도 가져와서 화면을 3등분(가로, 세로) 하는 픽셀 길이 계산
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
grid_w = frame_width // 3
grid_h = frame_height // 3

# 로그 저장을 위한 DB 관리자 객체 생성
db = DBManager(r'C:\project\Team-Robot_Arm\easyEEZYbotARM-master\python_packages\robot_arm.db')

last_log_time = 0
cooldown_seconds = 8.0 # 수거 동작(약 6~7초) 중 중복 인식을 막기 위한 쿨타임

while True:
    ret, frame = cap.read()
    if not ret: break # 프레임을 못 가져오면 종료
    
    # 화면에 3x3 그리드 구분선 그리기 (시각화 목적)
    cv2.line(frame, (grid_w, 0), (grid_w, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (grid_w*2, 0), (grid_w*2, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h), (frame_width, grid_h), (255, 255, 0), 1)
    cv2.line(frame, (0, grid_h*2), (frame_width, grid_h*2), (255, 255, 0), 1)

    # 영상 전처리 (흑백 전환 -> 블러 처리 -> 임계값 -> 엣지 검출)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0) # 배경 패턴 뭉개기
    # [수정 포인트] 조명에 따라 101, 4 등의 파라미터 조절 필요 가능성 있음
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 4)
    edges = cv2.Canny(blurred, 30, 100) # 표면 흠집 검출을 위한 윤곽선 추출
    
    # 노이즈 제거 (모폴로지 연산)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 덩어리(윤곽선) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h # 가로 세로 비율
        
        # 1차 필터링: 탁구공 크기(3000~40000)이고 비율이 1:1에 가까운지 확인
        if 3000 < area < 40000 and 0.85 < aspect_ratio < 1.15:
            # 해당 객체를 감싸는 최소한의 완벽한 원을 그림
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perfect_circle_area = math.pi * (radius ** 2)
            if perfect_circle_area == 0: continue
            
            # 찌그러짐 판단: 실제 객체 넓이 / 완벽한 원의 넓이
            circle_ratio = area / perfect_circle_area
            current_time = time.time()
            
            # --- 불량 상태 판별 로직 ---
            is_defective = False
            defect_msg = ""
            
            # 1. 외곽은 동그란 모양인지 확인 (비율이 95% 이상이면 정상 형태)
            if circle_ratio > 0.95:
                # 2. 형태가 정상이면, 내부에 흠집이나 찌그러짐(Top Dent)이 있는지 확인
                inner_mask = np.zeros_like(gray)
                # 객체 내부의 60% 영역만 마스킹 처리
                cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.6), 255, -1)
                # Canny 엣지 검출 결과와 내부 마스크가 겹치는 픽셀 수 계산
                internal_edge_pixels = np.sum((edges == 255) & (inner_mask == 255))
                
                # 내부 윤곽선 픽셀이 15개를 넘어가면 표면 불량으로 판정
                if internal_edge_pixels > 15:
                    is_defective = True
                    defect_msg = "Top Dent" # 표면 흠집
            else:
                # 95% 이하이면 형태 자체가 찌그러진 불량으로 판정
                is_defective = True
                defect_msg = f"Shape ({circle_ratio:.2f})" # 외곽 찌그러짐

            # --- 로봇팔 제어 및 DB 기록 실행 로직 ---
            # 공의 중심점(cx, cy)을 기준으로 3x3 그리드 중 어디(row, col)에 있는지 계산
            row = min(int(cy) // grid_h, 2)
            col = min(int(cx) // grid_w, 2)

            if is_defective:
                # 🔴 불량품(Damaged) 처리
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3) # 빨간색 테두리
                cv2.putText(frame, f"Damaged: {defect_msg}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 쿨타임이 지났다면(명령이 중복으로 들어가는 것 방지)
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Damaged", "B") # DB에 불량 로그 저장
                    pick_and_discard(row, col)    # 불량품 수거 시퀀스 실행
                    last_log_time = time.time()   # 수거 완료 후 타이머 리셋
                    
            else:
                # 🟢 양품(Normal) 처리
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3) # 초록색 테두리
                cv2.putText(frame, "Normal Pass", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 쿨타임이 지났다면 DB에 양품 로그만 저장하고 로봇팔은 움직이지 않음
                if current_time - last_log_time > cooldown_seconds:
                    db.insert_log("Normal", "A")
                    last_log_time = current_time

    # 영상 출력
    cv2.imshow("Smart Ball Check", frame)
    # 'q' 키를 누르면 루프 탈출 및 프로그램 종료
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# ==========================================
# 5. 자원 해제 및 프로그램 종료
# ==========================================
cap.release()
cv2.destroyAllWindows()
db.close()
if ser:
    ser.close()