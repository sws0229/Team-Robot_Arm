import cv2
import numpy as np
import math
import time
import serial
import sys
import os
from datetime import datetime

# ==========================================
# 0. 경로 설정 및 DB 임포트
# ==========================================
base_dir   = os.path.dirname(os.path.abspath(__file__))
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
GRIPPER_OPEN  = 90
GRIPPER_CLOSE = 150
HOVER_J2      = 120
HOVER_J3      = 90

DISCARD_RIGHT_J1 = 105  # 오른쪽 폐기함 (45+60)
NORMAL_LEFT_J1   = 0    # 왼쪽 정상함 (45-45, 최소값 0)

# 탁구공을 내려놓을 때 사용할 고정 J2, J3 각도
DROP_J2 = 180
DROP_J3 = 50

lookup_table = {
    (0, 0): (160,  20, 190, 60), (0, 1): (160,  50, 180, 50), (0, 2): (160,  70, 180, 55),
    (1, 0): (160,  20, 170, 40), (1, 1): (160,  45, 170, 40), (1, 2): (160,  90, 165, 40),
    (2, 0): (160,  10, 160, 20), (2, 1): (160,  45, 155, 20), (2, 2): (160,  80, 160, 20),
}

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    if ser:
        command = (f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},"
                   f"{move_time},{move_time},{move_time},{move_time}>")
        ser.write(command.encode())
        print(f"명령 전송됨: {command}")

# ==========================================
# 3. [핵심 수정] sleep 대신 keep_alive_sleep 사용
# ==========================================
def keep_alive_sleep(seconds, cap=None):
    """
    time.sleep 대신 이걸 씁니다.
    매 30ms마다 waitKey를 호출해 OpenCV 창이 응답없음이 되지 않도록 유지.
    cap을 넘기면 카메라 프레임도 계속 갱신해서 화면도 살아있게 유지.
    """
    end_time = time.time() + seconds
    while time.time() < end_time:
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                # ✅ [수정] 상하좌우 동시 반전 (마주보는 셋업)
                frame = cv2.flip(frame, -1)
                cv2.putText(frame, "Robot Moving...",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 165, 255), 3)
                cv2.imshow("Smart Quality Control System", frame)
        cv2.waitKey(30)  # 30ms마다 OS 이벤트 처리 → 창 응답 유지


# ==========================================
# 4. 양방향 분류 (Pick and Place) 자동화 시퀀스
# ==========================================
def pick_and_place(row, col, is_defective, cap=None):
    """
    분류 시퀀스 실행 후 결과 dict 반환.
    DB 의 update_action_result(**result) 에 그대로 전달 가능한 키를 담는다.
    """
    started = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = {
        'target_j1': None,
        'command_text': None,
        'action_started': started,
        'action_finished': None,
        'simulation': 1 if ser is None else 0,
        'success': 0,
        'error_msg': None,
    }

    target_angles = lookup_table.get((row, col))
    if not target_angles:
        result['error_msg'] = f'invalid grid ({row},{col})'
        result['action_finished'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return result

    _, j1, j2_down, j3_down = target_angles
    drop_j1 = DISCARD_RIGHT_J1 if is_defective else NORMAL_LEFT_J1
    result['target_j1'] = drop_j1
    direction = f"오른쪽({drop_j1}도) 폐기" if is_defective else f"왼쪽({drop_j1}도) 이동"
    print(f"[{row},{col}] {'불량품' if is_defective else '정상품'} 수거 시작 -> {direction}!")

    try:
        send_robot_command("M", GRIPPER_OPEN,  j1,      HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.2, cap)
        send_robot_command("M", GRIPPER_OPEN,  j1,      j2_down,  j3_down,  1000)
        keep_alive_sleep(1.2, cap)
        send_robot_command("M", GRIPPER_CLOSE, j1,      j2_down,  j3_down,   500)
        keep_alive_sleep(0.8, cap)
        send_robot_command("M", GRIPPER_CLOSE, j1,      HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.2, cap)
        send_robot_command("M", GRIPPER_CLOSE, drop_j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.2, cap)

        # 정상품/불량품 모두 내려놓을 때 고정된 DROP_J2, DROP_J3 각도 사용
        send_robot_command("M", GRIPPER_CLOSE, drop_j1, DROP_J2,  DROP_J3,  1000)
        keep_alive_sleep(1.2, cap)
        send_robot_command("M", GRIPPER_OPEN,  drop_j1, DROP_J2,  DROP_J3,   500)
        keep_alive_sleep(0.8, cap)

        # 내려놓고 다시 호버 높이로 복귀
        send_robot_command("M", GRIPPER_OPEN,  drop_j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.0, cap)

        # 시퀀스 마지막 명령 문자열 (대표값으로 저장)
        result['command_text'] = (f"<M,{GRIPPER_OPEN},{drop_j1},{HOVER_J2},"
                                  f"{HOVER_J3},1000,1000,1000,1000>")
        result['success'] = 1
        print("분류 작업 완료! 대기 상태로 복귀.")
    except Exception as e:
        result['success'] = 0
        result['error_msg'] = str(e)
        print(f"[오류] 동작 실패: {e}")

    result['action_finished'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return result

# ==========================================
# 5. 튜닝 파라미터
# ==========================================
BRIGHT_THRESHOLD = 200
MIN_BALL_AREA    = 1500
MAX_BALL_AREA    = 30000
CIRCLE_RATIO_MIN = 0.65
DENT_THRESHOLD   = 200
COOLDOWN_SEC     = 8.0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ==========================================
# 6. 헬퍼 함수
# ==========================================
def find_bright_ball_candidates(gray):
    _, bright_mask = cv2.threshold(gray, BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel_open  = np.ones((7,  7),  np.uint8)
    kernel_close = np.ones((11, 11), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN,  kernel_open)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow("Bright Mask", bright_mask)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def is_valid_ball(cnt):
    area = cv2.contourArea(cnt)
    if not (MIN_BALL_AREA < area < MAX_BALL_AREA):
        return False, None
    x, y, w, h = cv2.boundingRect(cnt)
    if h == 0:
        return False, None
    if not (0.6 < float(w) / h < 1.4):
        return False, None
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    if radius == 0:
        return False, None
    circle_ratio = area / (math.pi * radius ** 2)
    if circle_ratio < CIRCLE_RATIO_MIN:
        return False, None
    return True, (cx, cy, radius, circle_ratio)


def check_defect(edges, cx, cy, radius):
    inner_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.65), 255, -1)
    return int(np.sum((edges == 255) & (inner_mask == 255)))


# ==========================================
# 7. 메인 비전 시스템
# ==========================================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 종료합니다.")
    db.close()
    if ser: ser.close()
    sys.exit(1)

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
grid_w = frame_width  // 3
grid_h = frame_height // 3

fail_count       = 0
FAIL_LIMIT       = 30
last_action_time = 0.0

print("스마트 품질 검사 시스템 가동 (ver4 fix3)")
print(f"  밝기 임계값: {BRIGHT_THRESHOLD}  |  흠집 기준: {DENT_THRESHOLD}px  |  쿨다운: {COOLDOWN_SEC}s")

while True:
    ret, frame = cap.read()

    if not ret:
        fail_count += 1
        print(f"[경고] 프레임 읽기 실패 ({fail_count}/{FAIL_LIMIT})")
        if fail_count >= FAIL_LIMIT:
            print("카메라 신호 없음. 종료합니다.")
            break
        cv2.waitKey(100)
        continue
    fail_count = 0

    # ✅ [수정] 상하좌우 동시 반전 (마주보는 셋업)
    frame = cv2.flip(frame, -1)

    for xp in [grid_w, grid_w * 2]:
        cv2.line(frame, (xp, 0), (xp, frame_height), (255, 255, 0), 1)
    for yp in [grid_h, grid_h * 2]:
        cv2.line(frame, (0, yp), (frame_width, yp), (255, 255, 0), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_enhanced = clahe.apply(gray)
    blurred_edge  = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    edges         = cv2.Canny(blurred_edge, 40, 100)
    cv2.imshow("X-Ray (Edges)", edges)

    # 쿨다운 중이면 화면만 갱신하고 검출 스킵
    now = time.time()
    if now - last_action_time < COOLDOWN_SEC:
        remaining = COOLDOWN_SEC - (now - last_action_time)
        cv2.putText(frame, f"Cooldown {remaining:.1f}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.imshow("Smart Quality Control System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    contours = find_bright_ball_candidates(gray)

    for cnt in contours:
        valid, info = is_valid_ball(cnt)
        if not valid:
            continue

        cx, cy, radius, circle_ratio = info
        edge_px = check_defect(edges, cx, cy, radius)

        print(f" 스캔 중... 원형도: {circle_ratio:.2f} | 반지름: {int(radius)}px | 흠집: {edge_px}px")

        defect_reason = None  # 'Shape' / 'Dent' / None
        if circle_ratio < 0.90:
            is_defective = True
            defect_reason = 'Shape'
            defect_msg   = f"Shape ({circle_ratio:.2f})"
        elif edge_px > DENT_THRESHOLD:
            is_defective = True
            defect_reason = 'Dent'
            defect_msg   = f"Dent ({edge_px}px)"
        else:
            is_defective = False
            defect_msg   = ""

        row = min(int(cy) // grid_h, 2)
        col = min(int(cx) // grid_w, 2)
        x   = int(cx) - int(radius)
        y   = int(cy) - int(radius)

        color = (0, 0, 255) if is_defective else (0, 255, 0)
        label = f"Damaged: {defect_msg}" if is_defective else "Normal Pass"

        cv2.circle(frame, (int(cx), int(cy)), int(radius), color, 3)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Smart Quality Control System", frame)
        cv2.waitKey(1)

        wait_msg = (" 불량 탁구공 발견 1초 뒤 수거를 시작합니다..."
                    if is_defective else " 정상 탁구공 확인 1초 뒤 이동을 시작합니다...")
        print(wait_msg)
        keep_alive_sleep(1.0, cap)

        # 1) 검출 정보 먼저 INSERT, log id 받기
        log_id = db.insert_detection(
            status        = "Damaged" if is_defective else "Normal",
            action        = "B" if is_defective else "A",
            defect_reason = defect_reason,
            circle_ratio  = round(float(circle_ratio), 4),
            radius_px     = int(radius),
            edge_px       = int(edge_px),
            grid_row      = int(row),
            grid_col      = int(col),
            pixel_cx      = int(cx),
            pixel_cy      = int(cy),
        )

        # 2) 분류 시퀀스 실행 → 결과 dict 반환
        action_result = pick_and_place(row, col, is_defective=is_defective, cap=cap)

        # 3) 결과를 같은 row 에 UPDATE
        db.update_action_result(log_id, **action_result)

        last_action_time = time.time()
        break

    cv2.imshow("Smart Quality Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 8. 종료 처리
# ==========================================
cap.release()
cv2.destroyAllWindows()
db.close()
if ser:
    ser.close()
print("시스템이 정상적으로 종료되었습니다.")