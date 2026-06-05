# ============================================================
# 스마트 품질 검사 시스템 (auto.py)
# 카메라로 탁구공을 감지 → 불량 판별 → 로봇팔로 자동 분류
# ============================================================

import cv2          # 영상 처리 라이브러리 (카메라 입력, 이미지 분석, 화면 출력)
import numpy as np  # 수치 배열 연산 (마스크 생성, 픽셀 통계 계산 등)
import math         # 수학 함수 (원형도 계산에 pi 사용)
import time         # 시간 관련 함수 (쿨다운, sleep 대기)
import serial       # 아두이노와 USB 시리얼 통신
import sys          # 시스템 종료(sys.exit) 및 경로 조작
import os           # 파일/디렉토리 경로 처리
from datetime import datetime  # 작업 시작/종료 시각 기록용


# ==========================================
# 0. 경로 설정 및 DB 임포트
# ==========================================

# 현재 파일(auto.py)이 위치한 디렉토리 절대 경로
base_dir = os.path.dirname(os.path.abspath(__file__))

# 부모 디렉토리(한 단계 위) 경로 → db_manager.py가 여기에 있음
parent_dir = os.path.abspath(os.path.join(base_dir, '..'))

# 부모 디렉토리를 모듈 검색 경로에 추가해야 import 가능
sys.path.append(parent_dir)

# 부모 디렉토리의 db_manager 모듈에서 DBManager 클래스 불러오기
from db_manager import DBManager

# DB 파일 경로 지정 및 DBManager 객체 생성 (연결 시작)
db_path = os.path.join(parent_dir, 'robot_arm.db')
db = DBManager(db_path)


# ==========================================
# 1. 아두이노 시리얼 통신 설정
# ==========================================

try:
    # COM3 포트, 9600bps 보드레이트로 아두이노와 시리얼 연결
    # timeout=1: 응답 없을 때 1초 후 포기
    ser = serial.Serial('COM3', 9600, timeout=1)

    # 아두이노가 초기화되는 시간(약 2초) 동안 대기
    time.sleep(2)
    print("아두이노 통신 연결 성공!")

except Exception as e:
    # 아두이노가 연결되지 않아도 프로그램은 계속 실행 (시뮬레이션 모드)
    print(f"아두이노 연결 실패 (시뮬레이션 모드): {e}")
    ser = None  # ser가 None이면 이후 send_robot_command에서 명령 전송을 건너뜀


# ==========================================
# 2. 로봇팔 제어 파라미터 및 룩업 테이블
# ==========================================

# 그리퍼(집게) 서보 각도 상수
GRIPPER_OPEN  = 90   # 그리퍼 열린 상태 (공 안 잡음)
GRIPPER_CLOSE = 150  # 그리퍼 닫힌 상태 (공 잡음)

# 호버 높이: 공을 집거나 놓은 후 이동할 때의 J2, J3 안전 높이
HOVER_J2 = 120  # 팔꿈치 관절(J2) 들어올린 각도
HOVER_J3 = 90   # 손목 관절(J3) 들어올린 각도

# 공을 내려놓을 목적지의 J1(베이스 회전) 각도
DISCARD_RIGHT_J1 = 105  # 불량품 → 오른쪽 폐기함 방향
NORMAL_LEFT_J1   = 0    # 정상품 → 왼쪽 정상함 방향

# 공을 실제로 내려놓을 때의 J2, J3 각도 (바닥에 가까운 낮은 자세)
DROP_J2 = 180
DROP_J3 = 50

# 룩업 테이블: 카메라 화면을 3×3 그리드로 나눴을 때
# 각 칸 (row, col) → (그리퍼 각도, J1, J2_down, J3_down) 미리 측정한 관절 각도
# J2_down, J3_down은 해당 칸의 공에 닿기 위해 실제로 내려가는 각도
lookup_table = {
    (0, 0): (160,  20, 190, 60), (0, 1): (160,  50, 180, 50), (0, 2): (160,  70, 180, 55),
    (1, 0): (160,  20, 170, 40), (1, 1): (160,  45, 170, 40), (1, 2): (160,  90, 165, 40),
    (2, 0): (160,  10, 160, 20), (2, 1): (160,  45, 155, 20), (2, 2): (160,  80, 160, 20),
}


# ==========================================
# 3. 종료 플래그
# ==========================================

# 로봇 동작 중(keep_alive_sleep 안)에서 q키가 눌렸을 때
# 메인 루프를 안전하게 종료하기 위한 전역 플래그
should_quit = False


# ==========================================
# 로봇팔 명령 전송 함수
# ==========================================

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    """
    아두이노에 명령 문자열을 전송합니다.

    매개변수:
        instruction : 명령 유형 (예: "M" = Move)
        ee          : 엔드이펙터(그리퍼) 각도
        j1 ~ j3     : 각 관절 서보 각도
        move_time   : 각 관절이 목표 각도까지 이동하는 시간(ms), 기본 1000ms

    시리얼 오류 발생 시 예외를 그대로 위로 올려서
    pick_and_place의 except 블록이 처리하게 합니다.
    """
    if ser:  # 아두이노가 연결된 경우에만 전송 (None이면 시뮬레이션 모드)
        # 아두이노가 파싱하는 형식: <명령,그리퍼,J1,J2,J3,시간,시간,시간,시간>
        command = (f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},"
                   f"{move_time},{move_time},{move_time},{move_time}>")
        ser.write(command.encode())  # 문자열을 바이트로 인코딩해 전송; 실패 시 예외 발생
        print(f"명령 전송됨: {command}")


# ==========================================
# 4. keep_alive_sleep — OpenCV 창 응답 유지 + q키 감지
# ==========================================

def keep_alive_sleep(seconds, cap=None):
    """
    일반 time.sleep() 대신 사용하는 대기 함수.

    OpenCV는 waitKey()를 주기적으로 호출하지 않으면 창이 '응답 없음'이 됩니다.
    이 함수는 지정한 시간(seconds) 동안 30ms마다 waitKey를 호출해 창을 살려둡니다.

    매개변수:
        seconds : 대기할 총 시간(초)
        cap     : 카메라 객체. 전달하면 대기 중에도 카메라 프레임을 계속 갱신해
                  로봇이 움직이는 동안 실시간 영상을 보여줄 수 있음
    """
    global should_quit  # 전역 종료 플래그를 이 함수 안에서도 수정 가능하게 선언

    end_time = time.time() + seconds  # 대기 종료 시각 계산

    while time.time() < end_time:
        if cap is not None:
            ret, frame = cap.read()  # 카메라에서 새 프레임 읽기
            if ret:
                frame = cv2.flip(frame, -1)  # 카메라 방향 보정 (상하좌우 반전)

                # 로봇 동작 중임을 화면에 표시
                cv2.putText(frame, "Robot Moving...",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 165, 255), 3)
                cv2.imshow("Smart Quality Control System", frame)

        # 30ms마다 키 입력 확인 (OpenCV 창 유지에도 필수)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            should_quit = True  # q키 입력 → 메인 루프 종료 신호 설정
            break


# ==========================================
# 5. 양방향 분류 (Pick and Place) 자동화 시퀀스
# ==========================================

def pick_and_place(row, col, is_defective, cap=None):
    """
    공의 그리드 위치(row, col)와 불량 여부(is_defective)를 받아
    로봇팔로 공을 집어서 정상함 또는 폐기함에 내려놓는 전체 동작을 수행합니다.

    동작 순서:
      1) 픽업: 공 위로 이동 → 내려가기 → 그리퍼 닫기 → 들어올리기
      2) 이동: 목적지(정상/폐기함) 방향으로 회전 → 내려놓기 → 그리퍼 열기
      3) 홈 복귀: 호버 높이로 올라와 대기

    오류 발생 시 그리퍼를 열고 안전 높이로 복귀 시도 (충돌 방지).

    반환값: DB의 update_action_result()에 바로 전달 가능한 결과 dict
    """

    # 작업 시작 시각 기록
    started = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 결과 딕셔너리 초기화 (DB 저장용)
    result = {
        'target_j1'      : None,       # 목적지 J1 각도 (나중에 채워짐)
        'command_text'   : None,       # 마지막으로 전송된 명령 문자열
        'action_started' : started,    # 작업 시작 시각
        'action_finished': None,       # 작업 종료 시각 (나중에 채워짐)
        'simulation'     : 1 if ser is None else 0,  # 시뮬레이션 모드 여부
        'success'        : 0,          # 성공 여부 (기본값: 실패)
        'error_msg'      : None,       # 오류 발생 시 메시지
    }

    # 룩업 테이블에서 해당 그리드 칸의 관절 각도 조회
    target_angles = lookup_table.get((row, col))
    if not target_angles:
        # 테이블에 없는 유효하지 않은 그리드 위치인 경우 즉시 반환
        result['error_msg']       = f'invalid grid ({row},{col})'
        result['action_finished'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return result

    # 룩업 테이블에서 각 관절 각도 분리
    # (첫 번째 값은 사용 안 함, _로 무시)
    _, j1, j2_down, j3_down = target_angles

    # 불량품이면 오른쪽 폐기함(105도), 정상품이면 왼쪽 정상함(0도)
    drop_j1 = DISCARD_RIGHT_J1 if is_defective else NORMAL_LEFT_J1
    result['target_j1'] = drop_j1

    # 로그용 방향 설명 문자열
    direction = f"오른쪽({drop_j1}도) 폐기" if is_defective else f"왼쪽({drop_j1}도) 이동"
    print(f"[{row},{col}] {'불량품' if is_defective else '정상품'} 수거 시작 -> {direction}!")

    try:
        # ── 픽업 시퀀스 ──────────────────────────────────────────

        # 1단계: 그리퍼 열고, 공 위 호버 높이로 이동
        send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.2, cap)  # 관절 이동 완료까지 대기

        # 2단계: 그리퍼 열린 채로 공에 닿는 높이까지 내려가기
        send_robot_command("M", GRIPPER_OPEN, j1, j2_down, j3_down, 1000)
        keep_alive_sleep(1.2, cap)

        # 3단계: 그리퍼 닫아 공 잡기 (짧은 이동 시간 500ms)
        send_robot_command("M", GRIPPER_CLOSE, j1, j2_down, j3_down, 500)
        keep_alive_sleep(0.8, cap)  # 그리퍼가 확실히 닫힐 때까지 짧게 대기

        # 4단계: 공을 잡은 채로 호버 높이로 들어올리기
        send_robot_command("M", GRIPPER_CLOSE, j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.2, cap)

        # ── 이동 및 배치 시퀀스 ──────────────────────────────────

        # 5단계: 목적지 방향(drop_j1)으로 베이스 회전, 호버 높이 유지
        send_robot_command("M", GRIPPER_CLOSE, drop_j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.2, cap)

        # 6단계: 목적지 바구니 위에서 DROP 높이(낮은 자세)로 내려가기
        send_robot_command("M", GRIPPER_CLOSE, drop_j1, DROP_J2, DROP_J3, 1000)
        keep_alive_sleep(1.2, cap)

        # 7단계: 그리퍼 열어 공 내려놓기
        send_robot_command("M", GRIPPER_OPEN, drop_j1, DROP_J2, DROP_J3, 500)
        keep_alive_sleep(0.8, cap)

        # ── 홈 복귀 ──────────────────────────────────────────────

        # 8단계: 그리퍼 열린 채로 호버 높이로 복귀 (다음 동작 대기 자세)
        send_robot_command("M", GRIPPER_OPEN, drop_j1, HOVER_J2, HOVER_J3, 1000)
        keep_alive_sleep(1.0, cap)

        # DB 저장용 마지막 명령 문자열 기록
        result['command_text'] = (f"<M,{GRIPPER_OPEN},{drop_j1},{HOVER_J2},"
                                  f"{HOVER_J3},1000,1000,1000,1000>")
        result['success'] = 1  # 모든 단계 성공
        print("분류 작업 완료! 대기 상태로 복귀.")

    except Exception as e:
        # 시리얼 통신 오류 등 예외 발생 시
        result['success']   = 0
        result['error_msg'] = str(e)
        print(f"[오류] 동작 실패: {e}")

        # ── 오류 발생 시 안전 복귀 시도 ───────────────────
        # 그리퍼를 열어 공을 내려놓고, 호버 높이로 올려 충돌을 방지
        print("[복구] 안전 복귀 명령 전송 시도...")
        try:
            send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
            keep_alive_sleep(1.2, cap)
        except Exception as recover_err:
            # 복구 명령마저 실패하면 사람이 직접 개입해야 함
            print(f"[복구 실패] 수동 개입이 필요합니다: {recover_err}")

    # 작업 종료 시각 기록 후 결과 반환
    result['action_finished'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return result


# ==========================================
# 6. 튜닝 파라미터 (비전 검사 기준값)
# ==========================================

BRIGHT_THRESHOLD = 200    # 이 값보다 밝은 픽셀만 탁구공 후보로 간주 (0~255)
MIN_BALL_AREA    = 1500   # 공으로 인식할 최소 픽셀 면적 (너무 작은 잡음 제거)
MAX_BALL_AREA    = 30000  # 공으로 인식할 최대 픽셀 면적 (너무 큰 물체 제거)

# 원형도(circle_ratio) = 실제 윤곽 면적 / 최소 외접원 면적
# 1.0이면 완전한 원, 낮을수록 찌그러진 형태
# 0.65 → 0.50으로 낮춘 이유: 심하게 찌그러진 불량 공이 이 단계에서 탈락하면 미탐지됨
CIRCLE_RATIO_MIN = 0.50

# ── 불량 판정 임계값 ──────────────────────────────────────────
# 아래 값들은 실제 탁구공 데이터를 촬영해 디버그 출력을 보면서 조정해야 함

ELLIPSE_RATIO_THRESHOLD  = 0.82  # 타원 단축/장축 비율; 이 값 미만이면 찌그러짐으로 판정
                                  # 1.0 = 완전한 원, 낮을수록 납작하게 찌그러진 공

BRIGHTNESS_STD_THRESHOLD = 60.0  # 공 내부 밝기 표준편차; 높으면 불규칙한 반사 → 불량
                                  # 현재는 비활성화 상태. 조명 환경 실측 후 활성화 권장

DENT_THRESHOLD = 200  # 공 내부 Canny 엣지 픽셀 수; 많을수록 표면 흠집 가능성 높음 (보조 판단)

COOLDOWN_SEC = 8.0    # 한 번 공을 처리한 후 다음 감지까지 기다리는 시간(초)
                      # 로봇 동작 중 중복 트리거를 방지

# CLAHE(대비 제한 적응형 히스토그램 평활화) 객체 생성
# clipLimit=2.0: 대비 증폭 상한선 (너무 높으면 노이즈 증폭됨)
# tileGridSize=(8,8): 이미지를 8×8 타일로 나눠 각각 평활화
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ==========================================
# 7. 헬퍼 함수 (비전 검사)
# ==========================================

def find_bright_ball_candidates(gray_enhanced):
    """
    CLAHE로 대비가 강화된 흑백 이미지에서 탁구공 후보 윤곽선을 찾습니다.

    처리 순서:
      1) 이진화: BRIGHT_THRESHOLD 이상 밝은 영역만 흰색으로
      2) 모폴로지 OPEN: 작은 잡음(흰색 점) 제거
      3) 모폴로지 CLOSE: 공 내부의 작은 구멍(어두운 점) 메우기
      4) 윤곽선 검출 후 반환

    매개변수:
        gray_enhanced : CLAHE 적용된 흑백 이미지

    반환값:
        contours : 검출된 윤곽선 목록
    """
    # 이진화: BRIGHT_THRESHOLD 이상이면 흰색(255), 미만이면 검정(0)
    _, bright_mask = cv2.threshold(gray_enhanced, BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 모폴로지 커널 정의
    kernel_open  = np.ones((7,  7),  np.uint8)  # OPEN용 작은 커널 (잡음 제거)
    kernel_close = np.ones((11, 11), np.uint8)  # CLOSE용 큰 커널 (구멍 메우기)

    # OPEN 연산: 작은 흰색 잡음 점 제거
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN,  kernel_open)

    # CLOSE 연산: 공 내부 어두운 점(구멍) 메우기 → 윤곽선이 더 매끄럽게
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_close)

    # 디버그용: 이진화 + 모폴로지 처리된 마스크 창에 표시
    cv2.imshow("Bright Mask", bright_mask)

    # 외곽 윤곽선만 검출 (RETR_EXTERNAL), 꼭짓점 압축 저장 (CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def is_valid_ball(cnt):
    """
    윤곽선 하나를 받아 탁구공으로 볼 수 있는지 1차 필터링합니다.

    검사 항목:
      1) 면적: MIN_BALL_AREA ~ MAX_BALL_AREA 범위 내인지
      2) 종횡비: 바운딩박스 너비/높이 비율이 0.4~1.6 범위 내인지
                (찌그러진 공도 통과시키기 위해 넓게 설정)
      3) 원형도(circle_ratio): 실제 면적 / 최소 외접원 면적 ≥ CIRCLE_RATIO_MIN

    매개변수:
        cnt : 검사할 윤곽선

    반환값:
        (True,  (cx, cy, radius, circle_ratio)) : 탁구공 후보로 통과
        (False, None)                            : 탁구공 아님
    """
    # 1차: 면적 검사 — 너무 작거나 큰 윤곽선 제거
    area = cv2.contourArea(cnt)
    if not (MIN_BALL_AREA < area < MAX_BALL_AREA):
        return False, None

    # 2차: 종횡비 검사 — 바운딩박스 기준 가로/세로 비율
    x, y, w, h = cv2.boundingRect(cnt)
    if h == 0:  # ZeroDivisionError 방지
        return False, None
    # 0.4~1.6: 심하게 찌그러진 공(납작한 타원)도 통과할 수 있도록 완화
    if not (0.4 < float(w) / h < 1.6):
        return False, None

    # 3차: 원형도 검사 — 최소 외접원 대비 실제 면적 비율
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    if radius == 0:  # ZeroDivisionError 방지
        return False, None

    # circle_ratio: 1.0에 가까울수록 완전한 원
    circle_ratio = area / (math.pi * radius ** 2)
    if circle_ratio < CIRCLE_RATIO_MIN:
        return False, None  # 너무 불규칙한 형태 → 탁구공 아님

    return True, (cx, cy, radius, circle_ratio)


def check_shape_defect_ellipse(cnt):
    """
    윤곽선에 타원을 피팅하여 단축/장축 비율로 찌그러짐(형상 불량)을 판별합니다.

    ratio = 단축 / 장축
      - 1.0에 가까울수록 정상적인 원형
      - ELLIPSE_RATIO_THRESHOLD(0.82) 미만이면 찌그러진 불량으로 판정

    매개변수:
        cnt : 윤곽선

    반환값:
        (is_defective: bool, ratio: float)
    """
    # fitEllipse는 점이 5개 이상이어야 계산 가능
    if len(cnt) < 5:
        return False, 1.0  # 점 부족 → 판정 불가, 정상으로 처리

    try:
        ellipse = cv2.fitEllipse(cnt)          # 타원 피팅
        _, (major, minor), _ = ellipse         # 장축(major), 단축(minor) 추출

        if major == 0:
            return False, 1.0  # ZeroDivisionError 방지

        # major >= minor 보장 (fitEllipse 결과에서 순서가 바뀔 수 있음)
        if major < minor:
            major, minor = minor, major

        # 단축/장축 비율 계산 (1.0에 가까울수록 정상 원)
        ratio        = minor / major
        is_defective = ratio < ELLIPSE_RATIO_THRESHOLD  # 0.82 미만이면 불량
        return is_defective, round(ratio, 4)

    except cv2.error:
        # fitEllipse 계산 실패 시 정상으로 처리
        return False, 1.0


def check_brightness_defect(gray_enhanced, cx, cy, radius):
    """
    공 내부 영역의 밝기 표준편차를 계산하여 표면 불량을 판별합니다.

    원리:
      - 정상 공: 표면이 매끄러워 하이라이트(밝은 점)가 1곳 → 낮은 표준편차
      - 불량 공: 찌그러지거나 흠집이 있어 불규칙한 반사 → 높은 표준편차

    검사 범위: 공 중심에서 반지름의 80% 이내 원형 영역 (가장자리 잡음 제외)

    매개변수:
        gray_enhanced : CLAHE 적용 흑백 이미지
        cx, cy        : 공 중심 좌표
        radius        : 공 반지름 (픽셀)

    반환값:
        (is_defective: bool, std_dev: float)

    ※ 현재 비활성화 상태. 조명 환경 실측 후 judge_defect에서 주석 해제 필요.
    """
    # 공 내부(반지름 80%)에만 적용할 원형 마스크 생성
    mask = np.zeros(gray_enhanced.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius * 0.8), 255, -1)  # -1: 내부 채우기

    # 마스크가 흰색(255)인 픽셀들의 밝기값만 추출
    roi_pixels = gray_enhanced[mask == 255]
    if roi_pixels.size == 0:
        return False, 0.0  # 픽셀이 없으면 판정 불가

    # 밝기 표준편차 계산
    std_dev      = float(np.std(roi_pixels))
    is_defective = std_dev > BRIGHTNESS_STD_THRESHOLD  # 기준 초과 시 불량
    return is_defective, round(std_dev, 2)


def check_defect_edge(edges, cx, cy, radius):
    """
    Canny 엣지 이미지에서 공 내부의 엣지 픽셀 수를 세어 흠집/크랙을 보조 판별합니다.

    원리: 정상 공의 내부는 매끄러워 엣지가 거의 없고,
          흠집·크랙이 있으면 내부에 엣지가 많이 생김.

    검사 범위: 공 중심에서 반지름의 65% 이내 (가장자리 윤곽선 제외, 내부만 보기 위해)

    매개변수:
        edges  : Canny 엣지 이미지 (흰색=엣지, 검정=배경)
        cx, cy : 공 중심 좌표
        radius : 공 반지름 (픽셀)

    반환값:
        int — 내부 엣지 픽셀 수 (DENT_THRESHOLD=200 초과 시 불량으로 판정)
    """
    # 내부 영역(반지름 65%)에만 적용할 원형 마스크 생성
    inner_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(inner_mask, (int(cx), int(cy)), int(radius * 0.65), 255, -1)

    # 엣지 이미지(흰색=255)와 마스크 내부(255)가 겹치는 픽셀 수 반환
    return int(np.sum((edges == 255) & (inner_mask == 255)))


def judge_defect(cnt, gray_enhanced, edges, cx, cy, radius, circle_ratio):
    """
    세 가지 검사 결과를 종합하여 최종 불량 여부와 원인을 반환합니다.

    판정 우선순위:
      1) fitEllipse 비율  → 형상 변형(찌그러짐) 주요 지표
      2) Canny 엣지 수    → 표면 흠집 보조 판단
      3) 내부 밝기 std    → 환경 실측 후 활성화 권장 (현재 비활성)

    매개변수:
        cnt           : 탁구공 윤곽선
        gray_enhanced : CLAHE 적용 흑백 이미지
        edges         : Canny 엣지 이미지
        cx, cy        : 공 중심 좌표
        radius        : 공 반지름
        circle_ratio  : 원형도 (is_valid_ball에서 계산된 값)

    반환값:
        (is_defective: bool, defect_reason: str|None, debug_info: dict)
        - defect_reason: 'Shape'(찌그러짐) / 'Dent'(흠집) / None(정상)
        - debug_info   : 임계값 튜닝을 위한 수치 데이터
    """
    # 각 검사 함수 호출
    shape_bad,  ellipse_ratio = check_shape_defect_ellipse(cnt)           # 형상 불량 여부
    bright_bad, std_dev       = check_brightness_defect(gray_enhanced, cx, cy, radius)  # 밝기 불량 여부
    edge_px                   = check_defect_edge(edges, cx, cy, radius)  # 엣지 픽셀 수
    edge_bad                  = edge_px > DENT_THRESHOLD                  # 엣지 기준 초과 여부

    # 디버그 출력: 콘솔에 각 지표 수치를 출력해 임계값 조정에 활용
    print(f"  [검사] ellipse={ellipse_ratio:.3f}(기준<{ELLIPSE_RATIO_THRESHOLD}) | "
          f"std={std_dev:.1f}(기준>{BRIGHTNESS_STD_THRESHOLD}) | "
          f"edge={edge_px}px(기준>{DENT_THRESHOLD}) | "
          f"circle={circle_ratio:.2f}")

    # DB 저장 및 화면 출력에 사용할 디버그 정보 묶음
    debug_info = {
        'ellipse_ratio': ellipse_ratio,  # 타원 비율 (낮을수록 찌그러짐)
        'std_dev'      : std_dev,        # 내부 밝기 표준편차
        'edge_px'      : edge_px,        # 내부 엣지 픽셀 수
        'circle_ratio' : circle_ratio,   # 원형도
    }

    # 우선순위 1: 형상 불량 (타원 비율 기준 초과)
    if shape_bad:
        return True, 'Shape', debug_info

    # 우선순위 2: 엣지 기반 흠집 불량
    if edge_bad:
        return True, 'Dent', debug_info

    # 우선순위 3: 밝기 std 판정 (조명 환경 실측 후 아래 주석 해제)
    # if bright_bad:
    #     return True, 'Dent', debug_info

    # 모든 검사 통과 → 정상품
    return False, None, debug_info


# ==========================================
# 8. 메인 비전 시스템
# ==========================================

# 카메라 장치 열기 (인덱스 1 = 두 번째 카메라; USB 외장 카메라)
cap = cv2.VideoCapture(1)

# 카메라 열기 실패 시 자원 정리 후 종료
if not cap.isOpened():
    print("카메라를 열 수 없습니다. 종료합니다.")
    db.close()
    if ser:
        ser.close()
    sys.exit(1)

# 카메라 해상도 읽기
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3×3 그리드 한 칸의 픽셀 크기 계산
# max(..., 1): 해상도가 0으로 반환되는 오류 상황에서 ZeroDivisionError 방지
grid_w = max(frame_width  // 3, 1)  # 한 칸의 가로 픽셀 수
grid_h = max(frame_height // 3, 1)  # 한 칸의 세로 픽셀 수

fail_count       = 0    # 연속 프레임 읽기 실패 횟수 카운터
FAIL_LIMIT       = 30   # 이 횟수 이상 실패하면 카메라 신호 없는 것으로 판단하고 종료
last_action_time = 0.0  # 마지막으로 pick_and_place를 실행한 시각 (쿨다운 계산용)

# 시스템 시작 메시지 및 현재 설정값 출력
print("스마트 품질 검사 시스템 가동 (ver6 - 전체 수정본)")
print(f"  밝기 임계값: {BRIGHT_THRESHOLD}")
print(f"  타원 비율 기준: < {ELLIPSE_RATIO_THRESHOLD}")
print(f"  밝기 std 기준(비활성): > {BRIGHTNESS_STD_THRESHOLD}  ← 실측 후 활성화")
print(f"  엣지 기준(보조): > {DENT_THRESHOLD}px  |  쿨다운: {COOLDOWN_SEC}s")

# try/finally: 정상 종료(q키)든 오류 종료든 finally 블록이 반드시 실행되어
# 카메라·DB·시리얼 포트가 항상 안전하게 닫힘
try:
    while True:  # 메인 루프: 카메라 프레임을 계속 읽으며 공 감지

        # keep_alive_sleep 내에서 q키가 눌렸으면 루프 탈출
        if should_quit:
            break

        # 카메라에서 프레임 읽기
        ret, frame = cap.read()

        # 프레임 읽기 실패 처리
        if not ret:
            fail_count += 1
            print(f"[경고] 프레임 읽기 실패 ({fail_count}/{FAIL_LIMIT})")
            if fail_count >= FAIL_LIMIT:  # 30번 연속 실패 시 종료
                print("카메라 신호 없음. 종료합니다.")
                break
            cv2.waitKey(100)  # 100ms 대기 후 재시도
            continue
        fail_count = 0  # 읽기 성공 시 카운터 초기화

        # 카메라가 거꾸로 장착된 경우를 위해 상하좌우 반전 (-1: 둘 다 반전)
        frame = cv2.flip(frame, -1)

        # 3×3 그리드 구분선 노란색으로 화면에 그리기
        for xp in [grid_w, grid_w * 2]:  # 세로선 2개
            cv2.line(frame, (xp, 0), (xp, frame_height), (255, 255, 0), 1)
        for yp in [grid_h, grid_h * 2]:  # 가로선 2개
            cv2.line(frame, (0, yp), (frame_width, yp), (255, 255, 0), 1)

        # 이미지 전처리
        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 컬러 → 흑백 변환
        gray_enhanced = clahe.apply(gray)                        # CLAHE로 대비 향상
        blurred_edge  = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)  # 가우시안 블러 (엣지 노이즈 감소)
        edges         = cv2.Canny(blurred_edge, 40, 100)         # Canny 엣지 검출 (임계값 40~100)
        cv2.imshow("X-Ray (Edges)", edges)  # 엣지 이미지를 별도 창에 표시 (디버그용)

        # 쿨다운 중이면 공 감지 스킵, 화면만 갱신
        now = time.time()
        if now - last_action_time < COOLDOWN_SEC:
            remaining = COOLDOWN_SEC - (now - last_action_time)  # 남은 쿨다운 시간
            # 화면 좌상단에 남은 쿨다운 시간 표시
            cv2.putText(frame, f"Cooldown {remaining:.1f}s",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.imshow("Smart Quality Control System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # 이번 프레임 처리 건너뛰기

        # CLAHE 강화 이미지로 탁구공 후보 윤곽선 검출
        contours = find_bright_ball_candidates(gray_enhanced)

        # 검출된 모든 윤곽선에 대해 검사
        for cnt in contours:
            # 1차 필터: 탁구공 형태인지 확인
            valid, info = is_valid_ball(cnt)
            if not valid:
                continue  # 탁구공이 아니면 다음 윤곽선으로

            # 유효한 공의 중심, 반지름, 원형도 추출
            cx, cy, radius, circle_ratio = info

            # 불량 판별: 3가지 검사 종합
            is_defective, defect_reason, dbg = judge_defect(
                cnt, gray_enhanced, edges, cx, cy, radius, circle_ratio
            )

            # 화면에 표시할 불량 원인 메시지 구성
            if is_defective:
                defect_msg = (
                    f"Shape (ellipse={dbg['ellipse_ratio']:.2f})"  # 형상 불량
                    if defect_reason == 'Shape'
                    else f"Dent (edge={dbg['edge_px']}px)"          # 흠집 불량
                )
            else:
                defect_msg = ""  # 정상품은 메시지 없음

            # 공의 그리드 위치 계산 (픽셀 좌표 → 행/열 인덱스)
            row = min(int(cy) // grid_h, 2)  # 행 (0, 1, 2), 범위 초과 방지
            col = min(int(cx) // grid_w, 2)  # 열 (0, 1, 2), 범위 초과 방지

            # 화면 텍스트 표시 위치 (공 바운딩박스 좌상단 기준)
            x = int(cx) - int(radius)
            y = int(cy) - int(radius)

            # 불량: 빨간색, 정상: 초록색
            color = (0, 0, 255) if is_defective else (0, 255, 0)
            label = f"Damaged: {defect_msg}" if is_defective else "Normal Pass"

            # 탐지된 공 주위에 원 그리기
            cv2.circle(frame, (int(cx), int(cy)), int(radius), color, 3)

            # 공 위에 판정 라벨 텍스트 표시 (화면 위로 잘리지 않도록 y 최솟값 15 보정)
            label_y = max(y - 10, 15)
            cv2.putText(frame, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 공 아래에 디버그 수치 표시 (화면 아래로 잘리지 않도록 y 최댓값 보정)
            debug_text = (f"ellipse={dbg['ellipse_ratio']:.2f} "
                          f"std={dbg['std_dev']:.0f} "
                          f"edge={dbg['edge_px']}")
            debug_y = min(y + int(radius) * 2 + 20, frame_height - 10)
            cv2.putText(frame, debug_text, (x, debug_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 현재 프레임 화면 갱신
            cv2.imshow("Smart Quality Control System", frame)
            cv2.waitKey(1)

            # 1초 대기 메시지 출력 후 대기 (사용자가 화면을 확인할 시간)
            wait_msg = (" 불량 탁구공 발견 1초 뒤 수거를 시작합니다..."
                        if is_defective else " 정상 탁구공 확인 1초 뒤 이동을 시작합니다...")
            print(wait_msg)
            keep_alive_sleep(1.0, cap)

            # DB에 탐지 결과 저장 시도
            log_id = None  # insert 실패 시 None 유지 → 이후 update 건너뜀
            try:
                log_id = db.insert_detection(
                    status        = "Damaged" if is_defective else "Normal",  # 불량/정상
                    action        = "B"       if is_defective else "A",       # 동작 코드
                    defect_reason = defect_reason,                            # 불량 원인
                    circle_ratio  = round(float(circle_ratio), 4),            # 원형도
                    radius_px     = int(radius),                              # 반지름(픽셀)
                    edge_px       = int(dbg['edge_px']),                      # 엣지 픽셀 수
                    grid_row      = int(row),                                 # 그리드 행
                    grid_col      = int(col),                                 # 그리드 열
                    pixel_cx      = int(cx),                                  # 중심 x 좌표
                    pixel_cy      = int(cy),                                  # 중심 y 좌표
                )
            except Exception as db_err:
                print(f"[DB 오류] insert_detection 실패: {db_err}")

            # 로봇팔 동작 실행 (공 집어서 분류함에 내려놓기)
            action_result = pick_and_place(row, col, is_defective=is_defective, cap=cap)

            # 쿨다운 시작 시각 갱신 (pick_and_place 완료 후 갱신해야 8초 쿨다운이 정상 작동)
            # pick_and_place가 ~9~10초 걸리므로 시작 전에 찍으면
            # 완료 시점에 이미 8초가 지나 쿨다운이 즉시 풀려버림
            last_action_time = time.time()

            # DB에 로봇 동작 결과 업데이트 (insert가 성공해 log_id가 있을 때만)
            if log_id is not None:
                try:
                    db.update_action_result(log_id, **action_result)
                except Exception as db_err:
                    print(f"[DB 오류] update_action_result 실패: {db_err}")

            break  # 한 프레임에서 공을 여러 개 감지해도 첫 번째 하나만 처리

        # 매 프레임 결과 화면 갱신
        cv2.imshow("Smart Quality Control System", frame)

        # q키 누르면 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ==========================================
# 9. 종료 처리 (try/finally로 항상 실행 보장)
# ==========================================
finally:
    # 카메라 장치 해제
    cap.release()

    # 열린 모든 OpenCV 창 닫기
    cv2.destroyAllWindows()

    # DB 연결 종료 (실패해도 나머지 정리 계속)
    try:
        db.close()
    except Exception:
        pass

    # 아두이노 시리얼 포트 닫기 (연결돼 있을 때만)
    if ser:
        try:
            ser.close()
        except Exception:
            pass

    print("시스템이 정상적으로 종료되었습니다.")