# ============================================================
# 스마트 품질 검사 시스템 (auto5.py)
# 흰색 탁구공 감지 -> 정상/불량 판정 -> 로봇팔 자동 분류
#
# auto3.py에서 개선한 점
#   1. 인식용 원본 영상과 표시용 영상을 분리한다.
#   2. 흰색 HSV 마스크와 Canny 윤곽선을 함께 사용한다.
#   3. 화면의 3x3 안내선은 탐지가 끝난 뒤에만 그린다.
#   4. 같은 공에서 나온 중복 윤곽선을 제거한다.
#   5. 실행할 때 카메라 번호를 입력할 수 있다.
#   6. 학습된 SVM 모델로 정상/불량을 최종 판정한다.
#   7. 공을 놓고 SPACE를 눌러야만 한 번의 검사와 로봇 동작을 허용한다.
#
# 조작법
#   자동 분류: 공을 놓고 손을 치운 뒤 영상 창에서 SPACE, 종료는 Q 또는 Esc
#   피드백 수집: 실제 정상은 N, 실제 불량은 D, 종료는 Q 또는 Esc
# ============================================================

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import serial


# ==========================================
# 0. 경로 및 공통 설정
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PARENT_DIR)

from db_manager import DBManager
from ball_quality_ml import BallQualityClassifier, crop_ball_square


MAIN_WINDOW = "Smart Quality Control System"
MASK_WINDOW = "White Ball Mask (HSV)"
EDGE_WINDOW = "X-Ray (Edges)"

DEFAULT_CAMERA_INDEX = 0
DEFAULT_SERIAL_PORT = "COM3"
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "models", "ball_quality_svm_v3.xml")
)
DEFAULT_FEEDBACK_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "training_data", "live_feedback")
)

ser = None
db = None
should_quit = False


# ==========================================
# 1. 로봇팔 제어 파라미터
# ==========================================

GRIPPER_OPEN = 90
GRIPPER_CLOSE = 150

HOME_J1 = 90
HOME_J2 = 120
HOME_J3 = 90

HOVER_J2 = 120
HOVER_J3 = 90

DISCARD_RIGHT_J1 = 180
NORMAL_LEFT_J1 = 0

DROP_J2 = 180
DROP_J3 = 90


def load_lookup_table():
    lookup_json_path = os.path.join(BASE_DIR, "lookup_table.json")

    try:
        with open(lookup_json_path, "r", encoding="utf-8") as file:
            raw = json.load(file)

        table = {
            tuple(int(value) for value in key.split(",")): tuple(angles)
            for key, angles in raw["grid"].items()
        }
        print(f"[룩업 테이블] 로드 완료 ({len(table)}칸)")
        return table
    except Exception as error:
        print(f"[룩업 테이블 경고] 기본값을 사용합니다: {error}")
        return {
            (0, 0): (160, 60, 190, 60),
            (0, 1): (160, 95, 200, 65),
            (0, 2): (160, 130, 200, 65),
            (1, 0): (160, 60, 180, 50),
            (1, 1): (160, 95, 180, 50),
            (1, 2): (160, 130, 185, 50),
            (2, 0): (160, 60, 165, 30),
            (2, 1): (160, 95, 165, 30),
            (2, 2): (160, 130, 170, 30),
        }


LOOKUP_TABLE = load_lookup_table()


# ==========================================
# 2. 비전 검사 기준값
# ==========================================

# 흰색은 HSV에서 채도(S)가 낮고 밝기(V)가 높은 영역이다.
# 밝은 회색 바닥이 흰색으로 잡히지 않도록 V 하한을 높였다.
LOWER_WHITE = np.array([0, 0, 220], dtype=np.uint8)
UPPER_WHITE = np.array([179, 90, 255], dtype=np.uint8)

MIN_BALL_AREA = 1500
MAX_BALL_AREA = 30000
MIN_ASPECT_RATIO = 0.50
MAX_ASPECT_RATIO = 1.50
CIRCLE_RATIO_MIN = 0.40
FRAME_EDGE_MARGIN = 2
MAX_WHITE_MASK_COVERAGE = 0.18
STABLE_FRAMES_REQUIRED = 5
CLEAR_FRAMES_REQUIRED = 5
MANUAL_ARM_SETTLE_SEC = 0.8

ELLIPSE_RATIO_THRESHOLD = 0.94
DENT_THRESHOLD = 50
COOLDOWN_SEC = 7.0
FAIL_LIMIT = 30

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ==========================================
# 3. 실행 옵션과 장치 연결
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="흰색 탁구공을 검사하고 로봇팔로 분류합니다."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="카메라 번호. 생략하면 실행할 때 입력합니다.",
    )
    parser.add_argument(
        "--serial-port",
        default=DEFAULT_SERIAL_PORT,
        help=f"아두이노 포트 (기본값: {DEFAULT_SERIAL_PORT})",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="아두이노에 연결하지 않고 화면과 판정만 시험합니다.",
    )
    parser.add_argument(
        "--flip",
        choices=("both", "horizontal", "vertical", "none"),
        default="both",
        help="카메라 화면 뒤집기 방식 (기본값: both)",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=-7.0,
        help="수동 노출값 (기본값: -7)",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="카메라 자동 노출을 사용합니다.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"정상/불량 SVM 모델 경로 (기본값: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="로봇을 움직이지 않고 N/D 키로 실시간 오판 학습 사진을 수집합니다.",
    )
    parser.add_argument(
        "--feedback-dir",
        default=DEFAULT_FEEDBACK_DIR,
        help=f"피드백 사진 저장 폴더 (기본값: {DEFAULT_FEEDBACK_DIR})",
    )
    return parser.parse_args()


def choose_feedback_mode(argument_value):
    if argument_value:
        return True

    try:
        answer = input(
            "실행 모드를 선택하세요 (1: 자동 분류, 2: ML 피드백 사진 수집) "
            "[기본 1]: "
        ).strip()
    except EOFError:
        answer = ""

    return answer == "2"


def choose_camera_index(argument_value):
    if argument_value is not None:
        return argument_value

    try:
        answer = input(
            f"카메라 번호를 입력하세요 (노트북 웹캠 0, 외장 카메라 1인 경우가 많음) "
            f"[기본 {DEFAULT_CAMERA_INDEX}]: "
        ).strip()
        return DEFAULT_CAMERA_INDEX if answer == "" else int(answer)
    except (EOFError, ValueError):
        print(f"[카메라] 입력을 확인할 수 없어 {DEFAULT_CAMERA_INDEX}번을 사용합니다.")
        return DEFAULT_CAMERA_INDEX


def open_camera(camera_index, auto_exposure, exposure):
    # Windows에서는 DirectShow를 먼저 사용하고, 실패하면 기본 방식으로 재시도한다.
    if os.name == "nt":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        return None

    if not auto_exposure:
        # 지원하지 않는 카메라는 아래 설정을 자동으로 무시한다.
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        capture.set(cv2.CAP_PROP_EXPOSURE, exposure)

    return capture


def open_serial(serial_port, simulation):
    if simulation:
        print("[시뮬레이션] 아두이노 명령을 전송하지 않습니다.")
        return None

    try:
        connection = serial.Serial(
            serial_port,
            9600,
            timeout=1,
            write_timeout=2,
        )
        time.sleep(2)
        print(f"[아두이노] {serial_port} 연결 성공")
        return connection
    except Exception as error:
        print(f"[아두이노] 연결 실패, 시뮬레이션 모드로 계속합니다: {error}")
        return None


def flip_frame(frame, mode):
    flip_codes = {
        "both": -1,
        "horizontal": 1,
        "vertical": 0,
    }
    return cv2.flip(frame, flip_codes[mode]) if mode in flip_codes else frame


# ==========================================
# 4. 로봇팔 명령 함수
# ==========================================

def send_robot_command(instruction, ee, j1, j2, j3, move_time=1000):
    if ser is None:
        return

    command = (
        f"<{instruction},{int(ee)},{int(j1)},{int(j2)},{int(j3)},"
        f"{move_time},{move_time},{move_time},{move_time}>"
    )
    ser.write(command.encode())
    print(f"명령 전송됨: {command}")


def reset_to_home():
    if ser is None:
        print("[홈 초기화] 시뮬레이션 모드 - 명령 전송 생략")
        return

    print("[홈 초기화] 로봇팔을 홈 포지션으로 이동 중...")
    try:
        send_robot_command(
            "M",
            GRIPPER_OPEN,
            HOME_J1,
            HOME_J2,
            HOME_J3,
            move_time=2000,
        )
        time.sleep(2.5)
        print("[홈 초기화] 완료")
    except Exception as error:
        print(f"[홈 초기화 실패] 자세를 직접 확인하세요: {error}")


def keep_alive_sleep(seconds, cap=None, flip_mode="both"):
    global should_quit

    end_time = time.time() + seconds
    while time.time() < end_time:
        if cap is not None:
            success, live_frame = cap.read()
            if success:
                live_frame = flip_frame(live_frame, flip_mode)
                cv2.putText(
                    live_frame,
                    "Robot Moving...",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 165, 255),
                    3,
                )
                cv2.imshow(MAIN_WINDOW, live_frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            should_quit = True
            break


def pick_and_place(row, col, is_defective, cap=None, flip_mode="both"):
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "target_j1": None,
        "command_text": None,
        "action_started": started,
        "action_finished": None,
        "simulation": 1 if ser is None else 0,
        "success": 0,
        "error_msg": None,
    }

    target_angles = LOOKUP_TABLE.get((row, col))
    if not target_angles:
        result["error_msg"] = f"invalid grid ({row},{col})"
        result["action_finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result

    _, j1, j2_down, j3_down = target_angles
    drop_j1 = DISCARD_RIGHT_J1 if is_defective else NORMAL_LEFT_J1
    result["target_j1"] = drop_j1

    destination = (
        f"오른쪽({drop_j1}도) 불량 바구니"
        if is_defective
        else f"왼쪽({drop_j1}도) 정상 바구니"
    )
    quality = "불량품" if is_defective else "정상품"
    print(f"[{row},{col}] {quality} 수거 시작 -> {destination}")

    def wait(seconds):
        keep_alive_sleep(seconds, cap, flip_mode)

    try:
        send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
        wait(1.5)

        send_robot_command("M", GRIPPER_OPEN, j1, j2_down, j3_down, 1000)
        wait(1.5)

        send_robot_command("M", GRIPPER_CLOSE, j1, j2_down, j3_down, 500)
        wait(1.0)

        send_robot_command("M", GRIPPER_CLOSE, j1, HOVER_J2, HOVER_J3, 1000)
        wait(1.5)

        send_robot_command("M", GRIPPER_CLOSE, drop_j1, HOVER_J2, HOVER_J3, 1000)
        wait(1.5)

        send_robot_command("M", GRIPPER_CLOSE, drop_j1, DROP_J2, DROP_J3, 1000)
        wait(1.5)

        send_robot_command("M", GRIPPER_OPEN, drop_j1, DROP_J2, DROP_J3, 500)
        wait(1.0)

        send_robot_command("M", GRIPPER_OPEN, drop_j1, HOVER_J2, HOVER_J3, 1000)
        wait(1.5)

        result["command_text"] = (
            f"<M,{GRIPPER_OPEN},{drop_j1},{HOVER_J2},{HOVER_J3},"
            "1000,1000,1000,1000>"
        )
        result["success"] = 1
        print("분류 작업 완료")

    except Exception as error:
        result["error_msg"] = str(error)
        print(f"[로봇 동작 오류] {error}")

        try:
            print("[복구] 안전 위치로 복귀를 시도합니다.")
            send_robot_command("M", GRIPPER_OPEN, j1, HOVER_J2, HOVER_J3, 1000)
            wait(1.5)
        except Exception as recovery_error:
            print(f"[복구 실패] 수동 개입이 필요합니다: {recovery_error}")

    result["action_finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


# ==========================================
# 5. 영상 처리 함수
# ==========================================

def make_white_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((15, 15), dtype=np.uint8),
    )
    return mask


def make_edge_images(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_enhanced = CLAHE.apply(gray)
    blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 100)

    # 후보 탐색용 마스크다. 화면 표시용 원본에는 절대 그리지 않는다.
    expanded = cv2.dilate(
        edges,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )
    candidate_mask = cv2.morphologyEx(
        expanded,
        cv2.MORPH_CLOSE,
        np.ones((15, 15), dtype=np.uint8),
    )
    return gray_enhanced, edges, candidate_mask


def is_valid_ball(contour, frame_shape):
    area = cv2.contourArea(contour)
    if not (MIN_BALL_AREA < area < MAX_BALL_AREA):
        return False, None

    _, _, width, height = cv2.boundingRect(contour)
    if height == 0:
        return False, None

    aspect_ratio = float(width) / height
    if not (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
        return False, None

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    if radius <= 0:
        return False, None

    # 원의 일부가 영상 밖으로 잘린 후보는 크기와 모양을 신뢰할 수 없다.
    frame_height, frame_width = frame_shape[:2]
    if (
        center_x - radius <= FRAME_EDGE_MARGIN
        or center_y - radius <= FRAME_EDGE_MARGIN
        or center_x + radius >= frame_width - FRAME_EDGE_MARGIN
        or center_y + radius >= frame_height - FRAME_EDGE_MARGIN
    ):
        return False, None

    circle_ratio = area / (math.pi * radius**2)
    if circle_ratio < CIRCLE_RATIO_MIN:
        return False, None

    return True, (center_x, center_y, radius, circle_ratio)


def contour_roundness_score(contour, circle_ratio):
    """같은 공에서 나온 여러 윤곽선 중 가장 원에 가까운 것을 고르는 점수."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = 4.0 * math.pi * area / (perimeter**2)

    _, _, width, height = cv2.boundingRect(contour)
    aspect_score = 0.0
    if width > 0 and height > 0:
        aspect_score = min(width, height) / max(width, height)

    circle_score = min(max(circle_ratio, 0.0), 1.0)
    circularity_score = min(max(circularity, 0.0), 1.0)
    return (
        0.50 * circle_score
        + 0.30 * circularity_score
        + 0.20 * aspect_score
    )


def same_ball(first, second):
    distance = math.hypot(
        first["center_x"] - second["center_x"],
        first["center_y"] - second["center_y"],
    )
    center_tolerance = max(12.0, min(first["radius"], second["radius"]) * 0.45)
    return distance < center_tolerance


def find_ball_candidates(white_mask, edge_candidate_mask):
    # RETR_LIST를 사용해 바닥 선 안쪽에 있는 공 윤곽도 버리지 않는다.
    edge_contours, _ = cv2.findContours(
        edge_candidate_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    white_contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_candidates = [
        (contour, "CANNY") for contour in edge_contours
    ] + [
        (contour, "WHITE") for contour in white_contours
    ]

    valid_candidates = []
    for contour, source in raw_candidates:
        valid, info = is_valid_ball(contour, white_mask.shape)
        if not valid:
            continue

        center_x, center_y, radius, circle_ratio = info
        valid_candidates.append({
            "contour": contour,
            "source": source,
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
            "circle_ratio": circle_ratio,
            "roundness_score": contour_roundness_score(contour, circle_ratio),
        })

    # 같은 공에서 생긴 안쪽/바깥쪽 윤곽선을 한 그룹으로 묶는다.
    groups = []
    for candidate in valid_candidates:
        matching_group = next(
            (
                group
                for group in groups
                if any(same_ball(candidate, saved) for saved in group)
            ),
            None,
        )
        if matching_group is None:
            groups.append([candidate])
        else:
            matching_group.append(candidate)

    unique_candidates = []
    for group in groups:
        canny_candidates = [item for item in group if item["source"] == "CANNY"]
        white_candidates = [item for item in group if item["source"] == "WHITE"]

        # 가장 큰 윤곽선이 아니라 원형 점수가 가장 높은 윤곽선을 고른다.
        best_canny = (
            max(canny_candidates, key=lambda item: item["roundness_score"])
            if canny_candidates
            else None
        )
        best_white = (
            max(white_candidates, key=lambda item: item["roundness_score"])
            if white_candidates
            else None
        )

        # 안전장치: 흰색 마스크에서 공이 확인되지 않은 Canny 전용 후보는
        # 절대로 공으로 인정하지 않는다. Canny는 위치 보정에만 사용한다.
        if best_white is None:
            continue

        detection_candidate = best_canny or best_white
        classification_candidate = best_white
        if detection_candidate is None:
            continue

        merged_candidate = dict(detection_candidate)
        merged_candidate["classification_contour"] = classification_candidate["contour"]
        merged_candidate["classification_source"] = classification_candidate["source"]
        unique_candidates.append(merged_candidate)

    return unique_candidates


def check_shape_defect(contour):
    if len(contour) < 5:
        return False, 1.0

    try:
        _, (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)
        if major <= 0:
            return False, 1.0

        ellipse_ratio = minor / major
        return ellipse_ratio < ELLIPSE_RATIO_THRESHOLD, round(ellipse_ratio, 4)
    except cv2.error:
        return False, 1.0


def count_inner_edges(edges, center_x, center_y, radius):
    inner_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(
        inner_mask,
        (int(center_x), int(center_y)),
        int(radius * 0.65),
        255,
        -1,
    )
    return int(np.sum((edges == 255) & (inner_mask == 255)))


def judge_defect(candidate, edges, raw_frame, quality_classifier):
    shape_bad, ellipse_ratio = check_shape_defect(
        candidate["classification_contour"]
    )
    edge_pixels = count_inner_edges(
        edges,
        candidate["center_x"],
        candidate["center_y"],
        candidate["radius"],
    )
    edge_bad = edge_pixels > DENT_THRESHOLD

    ball_crop = crop_ball_square(
        raw_frame,
        candidate["center_x"],
        candidate["center_y"],
        candidate["radius"],
    )
    ml_result = quality_classifier.predict(ball_crop)

    debug_info = {
        "ellipse_ratio": ellipse_ratio,
        "edge_px": edge_pixels,
        "circle_ratio": candidate["circle_ratio"],
        "source": f"{candidate['source']}->{candidate['classification_source']}",
        "classic_shape_bad": shape_bad,
        "classic_edge_bad": edge_bad,
        "ml_label": ml_result["label"],
        "ml_margin": ml_result["margin"],
    }

    if ml_result["is_defective"]:
        return True, "ML", debug_info
    return False, None, debug_info


def analyze_frame(raw_frame, grid_width, grid_height, quality_classifier):
    white_mask = make_white_mask(raw_frame)
    _, edges, edge_candidate_mask = make_edge_images(raw_frame)

    white_coverage = float(np.count_nonzero(white_mask)) / white_mask.size
    if white_coverage > MAX_WHITE_MASK_COVERAGE:
        error_message = f"BLOCKED: white mask {white_coverage * 100:.1f}%"
        return [], white_mask, edges, error_message

    candidates = find_ball_candidates(white_mask, edge_candidate_mask)

    balls = []
    for candidate in candidates:
        try:
            defective, reason, debug = judge_defect(
                candidate,
                edges,
                raw_frame,
                quality_classifier,
            )
        except (cv2.error, RuntimeError, ValueError) as error:
            print(f"[ML 판정 오류] {error}")
            return [], white_mask, edges, "BLOCKED: ML prediction failed"
        row = max(0, min(int(candidate["center_y"]) // grid_height, 2))
        col = max(0, min(int(candidate["center_x"]) // grid_width, 2))

        balls.append(
            {
                **candidate,
                "is_defective": defective,
                "defect_reason": reason,
                "debug": debug,
                "row": row,
                "col": col,
            }
        )

    balls.sort(key=lambda ball: ball["radius"], reverse=True)
    return balls, white_mask, edges, None


def is_same_stable_ball(previous, current):
    if previous is None:
        return False

    if (
        previous["row"] != current["row"]
        or previous["col"] != current["col"]
        or previous["is_defective"] != current["is_defective"]
    ):
        return False

    center_distance = math.hypot(
        previous["center_x"] - current["center_x"],
        previous["center_y"] - current["center_y"],
    )
    center_tolerance = max(
        10.0,
        min(previous["radius"], current["radius"]) * 0.25,
    )
    radius_difference = abs(previous["radius"] - current["radius"])
    radius_tolerance = max(6.0, current["radius"] * 0.20)

    return (
        center_distance <= center_tolerance
        and radius_difference <= radius_tolerance
    )


# ==========================================
# 6. 화면 표시 함수
# ==========================================

def draw_grid(frame, grid_width, grid_height):
    height, width = frame.shape[:2]

    for x_position in (grid_width, grid_width * 2):
        cv2.line(frame, (x_position, 0), (x_position, height), (255, 255, 0), 1)
    for y_position in (grid_height, grid_height * 2):
        cv2.line(frame, (0, y_position), (width, y_position), (255, 255, 0), 1)


def draw_ball(frame, ball):
    center_x = int(ball["center_x"])
    center_y = int(ball["center_y"])
    radius = int(ball["radius"])
    debug = ball["debug"]

    color = (0, 0, 255) if ball["is_defective"] else (0, 255, 0)
    if ball["is_defective"]:
        label = f"Damaged: ML margin={debug['ml_margin']:.2f}"
    else:
        label = f"Normal: ML margin={debug['ml_margin']:.2f}"

    cv2.circle(frame, (center_x, center_y), radius, color, 3)
    cv2.putText(
        frame,
        label,
        (center_x - radius, max(center_y - radius - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    debug_text = (
        f"ml={debug['ml_label']} src={debug['source']} "
        f"ellipse={debug['ellipse_ratio']:.2f} edge={debug['edge_px']} "
        f"circle={debug['circle_ratio']:.2f}"
    )
    debug_y = min(center_y + radius + 22, frame.shape[0] - 10)
    cv2.putText(
        frame,
        debug_text,
        (center_x - radius, debug_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
    )


def draw_detection_screen(raw_frame, balls, grid_width, grid_height):
    # 중요: 안내선과 글씨는 인식이 모두 끝난 복사본에만 그린다.
    display_frame = raw_frame.copy()
    draw_grid(display_frame, grid_width, grid_height)

    for ball in balls:
        draw_ball(display_frame, ball)

    cv2.putText(
        display_frame,
        f"Balls: {len(balls)}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    return display_frame


# ==========================================
# 7. DB 기록 및 메인 반복
# ==========================================

def print_detection(ball):
    debug = ball["debug"]
    result = "Damaged" if ball["is_defective"] else "Normal"
    print(
        f"[판정] {result} | ML={debug['ml_label']} "
        f"margin={debug['ml_margin']:.3f} | source={debug['source']} | "
        f"grid=({ball['row']},{ball['col']}) | "
        f"ellipse={debug['ellipse_ratio']:.3f} | "
        f"edge={debug['edge_px']} | circle={debug['circle_ratio']:.3f}"
    )


def save_detection_to_db(ball):
    if db is None:
        return None

    try:
        return db.insert_detection(
            status="Damaged" if ball["is_defective"] else "Normal",
            action="B" if ball["is_defective"] else "A",
            defect_reason=ball["defect_reason"],
            circle_ratio=round(float(ball["circle_ratio"]), 4),
            radius_px=int(ball["radius"]),
            edge_px=int(ball["debug"]["edge_px"]),
            grid_row=int(ball["row"]),
            grid_col=int(ball["col"]),
            pixel_cx=int(ball["center_x"]),
            pixel_cy=int(ball["center_y"]),
        )
    except Exception as error:
        print(f"[DB 오류] 판정 기록 실패: {error}")
        return None


def update_action_in_db(log_id, action_result):
    if db is None or log_id is None:
        return

    try:
        db.update_action_result(log_id, **action_result)
    except Exception as error:
        print(f"[DB 오류] 동작 결과 기록 실패: {error}")


def feedback_counts(feedback_dir):
    return {
        label: len(
            [
                name
                for name in os.listdir(os.path.join(feedback_dir, "crops", label))
                if name.lower().endswith(".jpg")
            ]
        )
        for label in ("normal", "damaged")
    }


def prepare_feedback_dirs(feedback_dir):
    for image_type in ("raw", "crops"):
        for label in ("normal", "damaged"):
            os.makedirs(
                os.path.join(feedback_dir, image_type, label),
                exist_ok=True,
            )


def save_feedback_sample(raw_frame, ball, label, feedback_dir, camera_index):
    captured_at = datetime.now()
    filename = f"{label}_{captured_at.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    raw_path = os.path.join(feedback_dir, "raw", label, filename)
    crop_path = os.path.join(feedback_dir, "crops", label, filename)
    metadata_path = os.path.join(feedback_dir, "metadata.csv")

    ball_crop = crop_ball_square(
        raw_frame,
        ball["center_x"],
        ball["center_y"],
        ball["radius"],
    )
    if not cv2.imwrite(raw_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"원본 프레임 저장 실패: {raw_path}")
    if not cv2.imwrite(crop_path, ball_crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"공 이미지 저장 실패: {crop_path}")

    new_file = not os.path.exists(metadata_path)
    fields = (
        "captured_at",
        "label",
        "raw_path",
        "crop_path",
        "ml_prediction",
        "ml_margin",
        "ellipse_ratio",
        "edge_px",
        "circle_ratio",
        "grid_row",
        "grid_col",
        "center_x",
        "center_y",
        "radius",
        "camera_index",
    )
    with open(metadata_path, "a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "captured_at": captured_at.isoformat(timespec="milliseconds"),
                "label": label,
                "raw_path": raw_path,
                "crop_path": crop_path,
                "ml_prediction": ball["debug"]["ml_label"],
                "ml_margin": round(float(ball["debug"]["ml_margin"]), 5),
                "ellipse_ratio": ball["debug"]["ellipse_ratio"],
                "edge_px": ball["debug"]["edge_px"],
                "circle_ratio": round(float(ball["circle_ratio"]), 5),
                "grid_row": ball["row"],
                "grid_col": ball["col"],
                "center_x": round(float(ball["center_x"]), 2),
                "center_y": round(float(ball["center_y"]), 2),
                "radius": round(float(ball["radius"]), 2),
                "camera_index": camera_index,
            }
        )

    return crop_path


def run_system(args):
    global db, ser, should_quit

    feedback_mode = choose_feedback_mode(args.feedback)
    camera_index = choose_camera_index(args.camera)
    cap = None
    should_quit = False

    try:
        try:
            quality_classifier = BallQualityClassifier(args.model)
        except (FileNotFoundError, RuntimeError, cv2.error) as error:
            print(f"[ML 모델 오류] {error}")
            print("안전을 위해 로봇팔을 연결하거나 움직이지 않고 종료합니다.")
            return 1

        if feedback_mode:
            db = None
            ser = None
            prepare_feedback_dirs(args.feedback_dir)
            saved_counts = feedback_counts(args.feedback_dir)
        else:
            db_path = os.path.join(PARENT_DIR, "robot_arm.db")
            db = DBManager(db_path)
            ser = open_serial(args.serial_port, args.simulation)
        cap = open_camera(camera_index, args.auto_exposure, args.exposure)

        if cap is None:
            print(f"[카메라 오류] {camera_index}번 카메라를 열 수 없습니다.")
            return 1

        success, probe_frame = cap.read()
        if not success:
            print(f"[카메라 오류] {camera_index}번 카메라에서 영상을 읽을 수 없습니다.")
            return 1

        probe_frame = flip_frame(probe_frame, args.flip)
        frame_height, frame_width = probe_frame.shape[:2]
        grid_width = max(frame_width // 3, 1)
        grid_height = max(frame_height // 3, 1)

        if not feedback_mode:
            reset_to_home()

        if feedback_mode:
            print("ML 피드백 사진 수집 모드 가동 (로봇/아두이노/DB 사용 안 함)")
        else:
            print("스마트 품질 검사 시스템 가동 (auto5 - 흰색 공 ML 판정)")
        print(f"  카메라: {camera_index} | 해상도: {frame_width}x{frame_height}")
        print(f"  ML 모델: {quality_classifier.model_path}")
        print(f"  흰색 HSV: {LOWER_WHITE.tolist()} ~ {UPPER_WHITE.tolist()}")
        print(f"  타원 비율 기준: < {ELLIPSE_RATIO_THRESHOLD}")
        print(f"  내부 엣지 기준: > {DENT_THRESHOLD}px")
        print(f"  안전 확인: 같은 공 {STABLE_FRAMES_REQUIRED}프레임 연속 감지")
        if not feedback_mode:
            print("  수동 작동 승인: 공을 놓고 손을 치운 뒤 영상 창에서 SPACE")
            print(f"  승인 후 대기: {MANUAL_ARM_SETTLE_SEC:.1f}초")
        print(f"  흰색 마스크 차단 기준: 화면의 {MAX_WHITE_MASK_COVERAGE * 100:.0f}% 초과")
        if feedback_mode:
            print(f"  저장 위치: {os.path.abspath(args.feedback_dir)}")
            print("  N: 실제 정상 저장 | D: 실제 불량 저장 | Q: 종료")
        print("  종료: 영상 창에서 Q")

        fail_count = 0
        last_action_time = 0.0
        pending_frame = probe_frame
        stable_ball = None
        stable_count = 0
        inspection_armed = False
        armed_at = 0.0
        waiting_for_clear = False
        clear_count = 0
        feedback_status = "Place exactly one ball, then press N or D"
        feedback_status_color = (0, 255, 255)
        feedback_status_until = 0.0
        last_feedback_save = 0.0

        while not should_quit:
            if pending_frame is not None:
                raw_frame = pending_frame
                pending_frame = None
                success = True
            else:
                success, raw_frame = cap.read()
                if success:
                    raw_frame = flip_frame(raw_frame, args.flip)

            if not success:
                fail_count += 1
                print(f"[카메라 경고] 프레임 읽기 실패 ({fail_count}/{FAIL_LIMIT})")
                if fail_count >= FAIL_LIMIT:
                    print("카메라 신호가 없어 종료합니다.")
                    break
                cv2.waitKey(100)
                continue

            fail_count = 0

            # raw_frame에는 안내선이나 글씨를 절대로 그리지 않는다.
            balls, white_mask, edges, vision_error = analyze_frame(
                raw_frame,
                grid_width,
                grid_height,
                quality_classifier,
            )
            display_frame = draw_detection_screen(
                raw_frame,
                balls,
                grid_width,
                grid_height,
            )

            if feedback_mode:
                now = time.time()
                if now >= feedback_status_until:
                    if vision_error is not None:
                        feedback_status = vision_error
                        feedback_status_color = (0, 0, 255)
                    elif len(balls) != 1:
                        feedback_status = "Need exactly one detected ball"
                        feedback_status_color = (0, 200, 255)
                    else:
                        feedback_status = (
                            f"ML={balls[0]['debug']['ml_label']} "
                            f"margin={balls[0]['debug']['ml_margin']:.2f}"
                        )
                        feedback_status_color = (255, 255, 255)

                cv2.rectangle(
                    display_frame,
                    (0, 0),
                    (display_frame.shape[1], 120),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    display_frame,
                    "FEEDBACK MODE - ROBOT DISABLED",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "N: save NORMAL   D: save DAMAGED   Q: quit",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    display_frame,
                    (
                        f"Saved normal={saved_counts['normal']} "
                        f"damaged={saved_counts['damaged']}"
                    ),
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (120, 255, 120),
                    1,
                )
                cv2.putText(
                    display_frame,
                    feedback_status,
                    (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    feedback_status_color,
                    2,
                )

                cv2.imshow(MASK_WINDOW, white_mask)
                cv2.imshow(EDGE_WINDOW, edges)
                cv2.imshow(MAIN_WINDOW, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

                label = None
                if key in (ord("n"), ord("N")):
                    label = "normal"
                elif key in (ord("d"), ord("D")):
                    label = "damaged"

                if label is not None:
                    if vision_error is not None or len(balls) != 1:
                        feedback_status = "NOT SAVED: need exactly one valid ball"
                        feedback_status_color = (0, 0, 255)
                        feedback_status_until = now + 1.5
                    elif now - last_feedback_save >= 0.35:
                        try:
                            crop_path = save_feedback_sample(
                                raw_frame,
                                balls[0],
                                label,
                                args.feedback_dir,
                                camera_index,
                            )
                            saved_counts[label] += 1
                            last_feedback_save = now
                            feedback_status = f"Saved {label}: {os.path.basename(crop_path)}"
                            feedback_status_color = (0, 255, 0)
                            feedback_status_until = now + 1.2
                            print(
                                f"[피드백 저장] 실제={label} | "
                                f"ML={balls[0]['debug']['ml_label']} | "
                                f"margin={balls[0]['debug']['ml_margin']:.3f} | "
                                f"{crop_path}"
                            )
                        except (OSError, cv2.error) as error:
                            feedback_status = "SAVE FAILED"
                            feedback_status_color = (0, 0, 255)
                            feedback_status_until = now + 1.5
                            print(f"[피드백 저장 실패] {error}")
                continue

            now = time.time()
            cooldown_remaining = COOLDOWN_SEC - (now - last_action_time)
            action_ball = None
            safety_message = None
            safety_color = (0, 200, 255)

            if cooldown_remaining > 0:
                stable_ball = None
                stable_count = 0
                cv2.putText(
                    display_frame,
                    f"Cooldown {cooldown_remaining:.1f}s",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )
            elif vision_error is not None:
                stable_ball = None
                stable_count = 0
                clear_count = 0
                safety_message = vision_error
                safety_color = (0, 0, 255)
            elif waiting_for_clear:
                stable_ball = None
                stable_count = 0
                inspection_armed = False

                if len(balls) == 0:
                    clear_count += 1
                else:
                    clear_count = 0

                if clear_count >= CLEAR_FRAMES_REQUIRED:
                    waiting_for_clear = False
                    clear_count = 0
                    safety_message = "Ready for next ball"
                    safety_color = (0, 255, 0)
                else:
                    safety_message = (
                        f"BLOCKED: remove ball {clear_count}/{CLEAR_FRAMES_REQUIRED}"
                    )
                    safety_color = (0, 0, 255)
            elif not inspection_armed:
                stable_ball = None
                stable_count = 0
                safety_message = "PAUSED: place ball, remove hand, press SPACE"
                safety_color = (0, 200, 255)
            elif now - armed_at < MANUAL_ARM_SETTLE_SEC:
                stable_ball = None
                stable_count = 0
                settle_remaining = MANUAL_ARM_SETTLE_SEC - (now - armed_at)
                safety_message = f"ARMED: settling {settle_remaining:.1f}s"
                safety_color = (0, 200, 255)
            elif len(balls) == 0:
                stable_ball = None
                stable_count = 0
                safety_message = "Waiting for one white ball"
            elif len(balls) > 1:
                stable_ball = None
                stable_count = 0
                safety_message = f"BLOCKED: {len(balls)} candidates"
                safety_color = (0, 0, 255)
            else:
                current_ball = balls[0]
                if is_same_stable_ball(stable_ball, current_ball):
                    stable_count += 1
                else:
                    stable_count = 1

                stable_ball = current_ball
                if stable_count >= STABLE_FRAMES_REQUIRED:
                    action_ball = current_ball
                    safety_message = (
                        f"Confirmed {STABLE_FRAMES_REQUIRED}/{STABLE_FRAMES_REQUIRED}"
                    )
                    safety_color = (0, 255, 0)
                else:
                    safety_message = (
                        f"Checking {stable_count}/{STABLE_FRAMES_REQUIRED}"
                    )

            if safety_message is not None:
                cv2.putText(
                    display_frame,
                    safety_message,
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    safety_color,
                    2,
                )

            cv2.imshow(MASK_WINDOW, white_mask)
            cv2.imshow(EDGE_WINDOW, edges)
            cv2.imshow(MAIN_WINDOW, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if key == ord(" ") and not waiting_for_clear and cooldown_remaining <= 0:
                inspection_armed = not inspection_armed
                stable_ball = None
                stable_count = 0
                action_ball = None
                if inspection_armed:
                    armed_at = time.time()
                    print(
                        "[수동 승인] 검사를 시작합니다. "
                        "손을 화면 밖에 두세요."
                    )
                else:
                    print("[수동 정지] 검사를 취소했습니다.")

            if action_ball is None:
                continue

            target_ball = action_ball
            print_detection(target_ball)

            wait_message = (
                "불량 탁구공 발견 - 1초 뒤 수거를 시작합니다."
                if target_ball["is_defective"]
                else "정상 탁구공 확인 - 1초 뒤 이동을 시작합니다."
            )
            print(wait_message)
            keep_alive_sleep(1.0, cap, args.flip)

            if should_quit:
                break

            log_id = save_detection_to_db(target_ball)
            action_result = pick_and_place(
                target_ball["row"],
                target_ball["col"],
                target_ball["is_defective"],
                cap,
                args.flip,
            )
            last_action_time = time.time()
            update_action_in_db(log_id, action_result)
            waiting_for_clear = True
            clear_count = 0
            inspection_armed = False
            stable_ball = None
            stable_count = 0

        if feedback_mode:
            print("\nML 피드백 사진 수집을 종료했습니다.")
            print(
                f"정상 {saved_counts['normal']}장 / "
                f"불량 {saved_counts['damaged']}장"
            )
            print(f"저장 위치: {os.path.abspath(args.feedback_dir)}")
        return 0

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        if db is not None:
            try:
                db.close()
            except Exception:
                pass

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

        print("시스템이 정상적으로 종료되었습니다.")


if __name__ == "__main__":
    sys.exit(run_system(parse_args()))
