# ============================================================
# 스마트 품질 검사 시스템 (auto.py) - 기말 발표용 조명 방어 버전
# 카메라로 탁구공을 감지 → 불량 판별 → 로봇팔로 자동 분류
# [수정] 어두운 환경 대응: HSV V값 완화 + 캐니 폴백 + circle_ratio 완화
# ============================================================

import cv2          
import numpy as np  
import math         
import time         
import serial       
import sys          
import os           
import json         
from datetime import datetime  

# ==========================================
# 0. 경로 설정 및 DB 임포트
# ==========================================

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, '..'))
sys.path.append(parent_dir)

from db_manager import DBManager

db_path = os.path.join(parent_dir, 'robot_arm.db')
db = DBManager(db_path)

# ==========================================
# 1. 아두이노 시리얼 통신 설정
# ==========================================

try:
    ser = serial.Serial('COM3', 9600, timeout=1, write_timeout=2)
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

HOME_J1 = 90   
HOME_J2 = 120  
HOME_J3 = 90   

HOVER_J2 = 120  
HOVER_J3 = 90   

DISCARD_RIGHT_J1 = 180  
NORMAL_LEFT_J1   = 0    

DROP_J2 = 180
DROP_J3 = 90

_lookup_json_path = os.path.join(base_dir, 'lookup_table.json')
try:
    with open(_lookup_json_path, 'r', encoding='utf-8') as _f:
        _raw = json.load(_f)
    lookup_table = {
        tuple(int(x) for x in k.split(',')): tuple(v)
        for k, v in _raw['grid'].items()
    }
    print(f"[룩업 테이블] {_lookup_json_path} 로드 완료 ({len(lookup_table)}칸)")
except Exception as _e:
    print(f"[오류/경고] lookup_table.json 파싱 실패 또는 없음. 기본값 사용: {_e}")
    lookup_table = {
        (0, 0): (160,  60, 190, 60), (0, 1): (160,  95, 200, 65), (0, 2): (160, 130, 200, 65),
        (1, 0): (160,  60, 180, 50), (1, 1): (160,  95, 180, 50), (1, 2): (160, 130, 185, 50),
        (2, 0): (160,  60, 165, 30), (2, 1): (160,  95, 165, 30), (2, 2): (160, 130, 170, 30),
    }

# ==========================================
# 3. 종료 플래그
# ==========================================
should_quit = False

# ==========================================
# 로봇팔 명령 전송 함수
# ==========================================
def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    if ser:  
        command = (f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},"
                   f"{move_time},{move_time},{move_time},{move_time}>")
        ser.write(command.encode())  
        print(f"명령 전송됨: {command}")

# ==========================================
# 3-1. 홈 포지션 초기화 함수
# ==========================================
def reset_to_home():
    if ser is None:
        print("[홈 초기화] 시뮬레이션 모드 — 명령 전송 생략")
        return

    print("[홈 초기화] 로봇팔을 홈 포지션으로 이동 중...")
    try:
        send_robot_command("M", GRIPPER_OPEN, HOME_J1, HOME_J2, HOME_J3, move_time=2000)
        time.sleep(2.5)  
        print("[홈 초기화] 완료 — 검사 시작.")
    except Exception as e:
        print(f"[홈 초기화 실패] 수동으로 자세를 확인하세요: {e}")

# ==========================================
# 4. keep_alive_sleep
# ==========================================
def keep_alive_sleep(seconds, cap=None):
    global should_quit  
    end_time = time.time() + seconds  

    while time.time() < end_time:
        if cap is not None:
            ret, frame = cap.read()  
            if ret:
                frame = cv2.flip(frame, -1)  
                cv2.putText(frame, "Robot Moving...",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 165, 255), 3)
                cv2.imshow("Smart Quality Control System", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            should_quit = True  
            break

# ==========================================
# 5. 양방향 분류 (Pick and Place) 자동화 시퀀스
# ==========================================
def pick_and_place(row, col, is_defective, cap=None):
    started = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = {
        'target_j1'      : None,       
        'command_text'   : None,       
        'action_started' : started,    
        'action_finished': None,       
        'simulation'     : 1 if ser is None else 0,  
        'success'        : 0,          
        'error_msg'      : None,       
    }

    target_angles = lookup_table.get((row, col))
    if not target_angles:
        result['error_msg']       = f'invalid grid ({row},{col})'
        result['action_finished'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return result

    _, j1, j2_down, j3_down = target_angles
    drop_j1 = DISCARD_RIGHT_J1 if is_defective else NORMAL_LEFT_J1
    result['target_j1'] = drop_j1

    direction = f"오른쪽({drop_j1}도) 폐기" if is_defective else f"왼쪽({drop_j1}도) 이동"
    print(f"[{row},{col}] {'불량품' if is_defective else '정상품'} 수거 시작 -> {direction}!")

    try:
        send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.5, cap)  

        send_robot_command("M", GRIPPER_OPEN, j1, j2_down, j3_down, 1000)
        keep_alive_sleep(1.5, cap)  

        send_robot_command("M", GRIPPER_CLOSE, j1, j2_down, j3_down, 500)
        keep_alive_sleep(1.0, cap)  

        send_robot_command("M", GRIPPER_CLOSE, j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.5, cap)  

        send_robot_command("M", GRIPPER_CLOSE, drop_j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.5, cap)  

        send_robot_command("M", GRIPPER_CLOSE, drop_j1, DROP_J2, DROP_J3, 1000)
        keep_alive_sleep(1.5, cap)  

        send_robot_command("M", GRIPPER_OPEN, drop_j1, DROP_J2, DROP_J3, 500)
        keep_alive_sleep(1.0, cap)  

        send_robot_command("M", GRIPPER_OPEN, drop_j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.5, cap)  

        result['command_text'] = (f"<M,{GRIPPER_OPEN},{drop_j1},{HOVER_J2},"
                                  f"{HOVER_J3},1000,1000,1000,1000>")
        result['success'] = 1  
        print("분류 작업 완료! 대기 상태로 복귀.")

    except Exception as e:
        result['success']   = 0
        result['error_msg'] = str(e)
        print(f"[오류] 동작 실패: {e}")

        print("[복구] 안전 복귀 명령 전송 시도...")
        try:
            send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
            keep_alive_sleep(1.5, cap)  
        except Exception as recover_err:
            print(f"[복구 실패] 수동 개입이 필요합니다: {recover_err}")

    result['action_finished'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return result

# ==========================================
# 6. 튜닝 파라미터 (비전 검사 기준값)
# ==========================================

# [수정] 어두운 환경 대응: V 하한 50→30, S 하한 70→40으로 완화
LOWER_ORANGE = np.array([0,  40, 30])   # H, S, V 하한선
UPPER_ORANGE = np.array([25, 255, 255]) # H, S, V 상한선

MIN_BALL_AREA    = 1500   
MAX_BALL_AREA    = 30000  

# [수정] 어두울 때 마스크가 덜 깔끔해져 circle_ratio가 낮게 나오므로 0.50→0.40으로 완화
CIRCLE_RATIO_MIN = 0.40

ELLIPSE_RATIO_THRESHOLD  = 0.94  
BRIGHTNESS_STD_THRESHOLD = 60.0  
DENT_THRESHOLD = 50  
COOLDOWN_SEC = 7.0    

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ==========================================
# 7. 헬퍼 함수 (비전 검사)
# ==========================================

def find_bright_ball_candidates(frame):
    """
    1차: HSV 색상 기반 탐지 (주황색 마스크)
    2차: HSV 실패 시 캐니 엣지 기반 폴백 (어두운 환경 대응)
    """
    # --- 1차: HSV 색상 마스크 ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    kernel_open  = np.ones((5,  5),  np.uint8)
    kernel_close = np.ones((11, 11), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN,  kernel_open)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel_close)

    cv2.imshow("Orange Ball Mask (HSV)", orange_mask)

    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 유효 면적 윤곽선이 하나라도 있으면 HSV 결과 반환
    valid_hsv = [c for c in contours if MIN_BALL_AREA < cv2.contourArea(c) < MAX_BALL_AREA]
    if valid_hsv:
        return valid_hsv

    # --- 2차 폴백: 어두울 때 캐니 엣지 기반 원 탐지 ---
    print("[폴백] HSV 마스크 실패 → 캐니 엣지 기반 탐지로 전환")
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edge_mask = cv2.Canny(blurred, 40, 100)

    # 엣지를 팽창·채워서 윤곽 영역 생성
    dilated = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=2)
    filled  = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    fallback_contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return fallback_contours

def is_valid_ball(cnt):
    area = cv2.contourArea(cnt)
    if not (MIN_BALL_AREA < area < MAX_BALL_AREA):
        return False, None

    x, y, w, h = cv2.boundingRect(cnt)
    if h == 0:  
        return False, None
    
    if not (0.4 < float(w) / h < 1.6):
        return False, None

    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    if radius == 0:  
        return False, None

    circle_ratio = area / (math.pi * radius ** 2)
    if circle_ratio < CIRCLE_RATIO_MIN:
        return False, None  

    return True, (cx, cy, radius, circle_ratio)

def check_shape_defect_ellipse(cnt):
    if len(cnt) < 5:
        return False, 1.0  

    try:
        ellipse = cv2.fitEllipse(cnt)          
        _, (major, minor), _ = ellipse         

        if major == 0:
            return False, 1.0  

        if major < minor:
            major, minor = minor, major

        ratio        = minor / major
        is_defective = ratio < ELLIPSE_RATIO_THRESHOLD  
        return is_defective, round(ratio, 4)
    except cv2.error:
        return False, 1.0

def check_brightness_defect(gray_enhanced, cx, cy, radius):
    mask = np.zeros(gray_enhanced.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius * 0.8), 255, -1)  
    roi_pixels = gray_enhanced[mask == 255]
    if roi_pixels.size == 0:
        return False, 0.0  

    std_dev      = float(np.std(roi_pixels))
    is_defective = std_dev > BRIGHTNESS_STD_THRESHOLD  
    return is_defective, round(std_dev, 2)

def check_defect_edge(edges, cx, cy, radius):
    inner_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.65), 255, -1)
    return int(np.sum((edges == 255) & (inner_mask == 255)))

def judge_defect(cnt, gray_enhanced, edges, cx, cy, radius, circle_ratio):
    shape_bad,  ellipse_ratio = check_shape_defect_ellipse(cnt)           
    edge_px                   = check_defect_edge(edges, cx, cy, radius)  
    edge_bad                  = edge_px > DENT_THRESHOLD                  

    # [수정] 조명 반사로 인한 오탐을 막기 위해 밝기 편차 판정은 강제 비활성화
    bright_bad, std_dev       = False, 0.0 

    print(f"  [검사] ellipse={ellipse_ratio:.3f}(기준<{ELLIPSE_RATIO_THRESHOLD}) | "
          f"std=비활성 | "
          f"edge={edge_px}px(기준>{DENT_THRESHOLD}) | "
          f"circle={circle_ratio:.2f}")

    debug_info = {
        'ellipse_ratio': ellipse_ratio,  
        'std_dev'      : std_dev,        
        'edge_px'      : edge_px,        
        'circle_ratio' : circle_ratio,   
    }

    if shape_bad:
        return True, 'Shape', debug_info

    if edge_bad:
        return True, 'Dent', debug_info

    if bright_bad: # 무조건 False로 통과됨
        return True, 'Dent', debug_info

    return False, None, debug_info

# ==========================================
# 8. 메인 비전 시스템
# ==========================================

cap = cv2.VideoCapture(1)

# [추가] 다이소 카메라 자동 노출(Auto Exposure) 끄기
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
cap.set(cv2.CAP_PROP_EXPOSURE, -5)        

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 종료합니다.")
    db.close()
    if ser:
        ser.close()
    sys.exit(1)

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

reset_to_home()

grid_w = max(frame_width  // 3, 1)  
grid_h = max(frame_height // 3, 1)  

fail_count       = 0    
FAIL_LIMIT       = 30   
last_action_time = 0.0  

print("스마트 품질 검사 시스템 가동 (ver_Final - 조명 최적화)")
print(f"  타원 비율 기준: < {ELLIPSE_RATIO_THRESHOLD}")
print(f"  밝기 std 기준: 비활성화됨 (오탐지 방지)")
print(f"  엣지 기준(보조): > {DENT_THRESHOLD}px  |  쿨다운: {COOLDOWN_SEC}s")
print(f"  HSV 하한: {LOWER_ORANGE}  |  circle_ratio 최소: {CIRCLE_RATIO_MIN}")

try:
    while True:  
        if should_quit:
            break

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

        frame = cv2.flip(frame, -1)

        for xp in [grid_w, grid_w * 2]:  
            cv2.line(frame, (xp, 0), (xp, frame_height), (255, 255, 0), 1)
        for yp in [grid_h, grid_h * 2]:  
            cv2.line(frame, (0, yp), (frame_width, yp), (255, 255, 0), 1)

        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        gray_enhanced = clahe.apply(gray)                        
        blurred_edge  = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)  
        edges         = cv2.Canny(blurred_edge, 40, 100)         
        cv2.imshow("X-Ray (Edges)", edges)  

        now = time.time()
        if now - last_action_time < COOLDOWN_SEC:
            remaining = COOLDOWN_SEC - (now - last_action_time)  
            cv2.putText(frame, f"Cooldown {remaining:.1f}s",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.imshow("Smart Quality Control System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  

        # [수정] 원본 프레임을 넘겨주어 HSV → 캐니 폴백 순으로 탐지
        contours = find_bright_ball_candidates(frame)

        ball_queue = []  

        for cnt in contours:
            valid, info = is_valid_ball(cnt)
            if not valid:
                continue

            cx, cy, radius, circle_ratio = info

            is_defective, defect_reason, dbg = judge_defect(
                cnt, gray_enhanced, edges, cx, cy, radius, circle_ratio
            )

            row = min(int(cy) // grid_h, 2)
            col = min(int(cx) // grid_w, 2)

            ball_queue.append((cx, cy, radius, circle_ratio,
                               is_defective, defect_reason, dbg, row, col))

        ball_queue.sort(key=lambda b: b[2], reverse=True)

        for ball in ball_queue:
            cx, cy, radius, circle_ratio, is_defective, defect_reason, dbg, row, col = ball

            if is_defective:
                defect_msg = (
                    f"Shape (ellipse={dbg['ellipse_ratio']:.2f})"
                    if defect_reason == 'Shape'
                    else f"Dent (edge={dbg['edge_px']}px)"
                )
            else:
                defect_msg = ""

            x = int(cx) - int(radius)
            y = int(cy) - int(radius)
            color = (0, 0, 255) if is_defective else (0, 255, 0)
            label = f"Damaged: {defect_msg}" if is_defective else "Normal Pass"

            cv2.circle(frame, (int(cx), int(cy)), int(radius), color, 3)

            label_y = max(y - 10, 15)
            cv2.putText(frame, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            debug_text = (f"ellipse={dbg['ellipse_ratio']:.2f} "
                          f"std={dbg['std_dev']:.0f} "
                          f"edge={dbg['edge_px']}")
            debug_y = min(y + int(radius) * 2 + 20, frame_height - 10)
            cv2.putText(frame, debug_text, (x, debug_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if ball_queue:
            cv2.putText(frame, f"Balls: {len(ball_queue)}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Smart Quality Control System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        for ball in ball_queue:
            if should_quit:
                break

            cx, cy, radius, circle_ratio, is_defective, defect_reason, dbg, row, col = ball

            wait_msg = (" 불량 탁구공 발견 1초 뒤 수거를 시작합니다..."
                        if is_defective else " 정상 탁구공 확인 1초 뒤 이동을 시작합니다...")
            print(wait_msg)
            keep_alive_sleep(1.0, cap)

            log_id = None
            try:
                log_id = db.insert_detection(
                    status        = "Damaged" if is_defective else "Normal",
                    action        = "B"       if is_defective else "A",
                    defect_reason = defect_reason,
                    circle_ratio  = round(float(circle_ratio), 4),
                    radius_px     = int(radius),
                    edge_px       = int(dbg['edge_px']),
                    grid_row      = int(row),
                    grid_col      = int(col),
                    pixel_cx      = int(cx),
                    pixel_cy      = int(cy),
                )
            except Exception as db_err:
                print(f"[DB 오류] insert_detection 실패: {db_err}")

            action_result = pick_and_place(row, col, is_defective=is_defective, cap=cap)

            last_action_time = time.time()

            if log_id is not None:
                try:
                    db.update_action_result(log_id, **action_result)
                except Exception as db_err:
                    print(f"[DB 오류] update_action_result 실패: {db_err}")

            if should_quit:
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ==========================================
# 9. 종료 처리
# ==========================================
finally:
    cap.release()
    cv2.destroyAllWindows()

    try:
        db.close()
    except Exception:
        pass

    if ser:
        try:
            ser.close()
        except Exception:
            pass

    print("시스템이 정상적으로 종료되었습니다.")