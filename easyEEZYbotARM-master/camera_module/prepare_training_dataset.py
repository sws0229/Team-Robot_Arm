"""수집한 사진에서 공을 잘라 기계학습용 데이터로 준비한다.

원본은 변경하지 않는다. 정상·불량 사진에서 흰 공 하나를 찾아 정사각형으로
저장하고, 공을 못 찾거나 모양이 애매한 사진은 검토 폴더로 분리한다.
별도 키 조작 없이 실행하면 자동으로 처리된다.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "training_data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "training_data" / "prepared"

LOWER_WHITE = np.array([0, 0, 220], dtype=np.uint8)
UPPER_WHITE = np.array([179, 90, 255], dtype=np.uint8)

MIN_BALL_AREA = 1500
MAX_BALL_AREA = 30000
MIN_ASPECT_RATIO = 0.50
MAX_ASPECT_RATIO = 1.50
MIN_CIRCLE_RATIO = 0.40
MAX_WHITE_COVERAGE = 0.18
FRAME_EDGE_MARGIN = 2

# 아래 기준은 사람의 추가 검토가 필요한 사진만 고르며 라벨을 바꾸지 않는다.
NORMAL_MIN_ELLIPSE_RATIO = 0.94
DAMAGED_LOOKS_NORMAL_ELLIPSE = 0.975
DAMAGED_LOOKS_NORMAL_SOLIDITY = 0.975
DAMAGED_LOOKS_NORMAL_RADIAL_CV = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="수집한 탁구공 사진을 자르고 애매한 사진을 검토 대상으로 분리합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"원본 사진 폴더 (기본값: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"처리 결과 폴더 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="자를 정사각형 사진의 픽셀 크기 (기본값: 224).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.35,
        help="공 주변에 포함할 여백의 비율.",
    )
    return parser.parse_args()


def make_white_mask(frame: np.ndarray) -> np.ndarray:
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


def contour_metrics(contour: np.ndarray, frame_shape) -> dict[str, float] | None:
    area = float(cv2.contourArea(contour))
    if not (MIN_BALL_AREA < area < MAX_BALL_AREA):
        return None

    x, y, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return None

    aspect_ratio = float(width) / height
    if not (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
        return None

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    if radius <= 0:
        return None

    frame_height, frame_width = frame_shape[:2]
    touches_edge = (
        center_x - radius <= FRAME_EDGE_MARGIN
        or center_y - radius <= FRAME_EDGE_MARGIN
        or center_x + radius >= frame_width - FRAME_EDGE_MARGIN
        or center_y + radius >= frame_height - FRAME_EDGE_MARGIN
    )
    if touches_edge:
        return None

    circle_ratio = area / (math.pi * radius**2)
    if circle_ratio < MIN_CIRCLE_RATIO:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    circularity = 0.0
    if perimeter > 0:
        circularity = 4.0 * math.pi * area / (perimeter**2)

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0

    ellipse_ratio = 0.0
    if len(contour) >= 5:
        _, (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        longest_axis = max(axis_a, axis_b)
        if longest_axis > 0:
            ellipse_ratio = min(axis_a, axis_b) / longest_axis

    moments = cv2.moments(contour)
    if moments["m00"]:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    else:
        centroid_x, centroid_y = center_x, center_y

    points = contour.reshape(-1, 2).astype(np.float32)
    distances = np.hypot(points[:, 0] - centroid_x, points[:, 1] - centroid_y)
    mean_distance = float(np.mean(distances)) if distances.size else 0.0
    radial_cv = (
        float(np.std(distances)) / mean_distance if mean_distance > 0 else 0.0
    )

    return {
        "area": area,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "radius": float(radius),
        "aspect_ratio": aspect_ratio,
        "circle_ratio": circle_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "ellipse_ratio": ellipse_ratio,
        "radial_cv": radial_cv,
    }


def find_ball(frame: np.ndarray) -> tuple[dict[str, float] | None, np.ndarray, str]:
    mask = make_white_mask(frame)
    white_coverage = float(cv2.countNonZero(mask)) / mask.size
    if white_coverage > MAX_WHITE_COVERAGE:
        return None, mask, "white_mask_too_large"

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    candidates = []
    for contour in contours:
        metrics = contour_metrics(contour, frame.shape)
        if metrics is not None:
            metrics["contour"] = contour
            candidates.append(metrics)

    if not candidates:
        return None, mask, "ball_not_found"
    if len(candidates) > 1:
        return None, mask, "multiple_balls"
    return candidates[0], mask, "ok"


def square_crop(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    diameter: float,
    padding: float,
    output_size: int,
) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    side = max(8, int(round(diameter * (1.0 + 2.0 * padding))))
    side = min(side, frame_width, frame_height)

    # 공이 화면 가장자리 근처에 있어도 검은 패딩을 만들지 않는다. 원하는
    # 정사각형을 영상 안쪽으로 이동시켜 실제 배경만 포함한다.
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(left, frame_width - side))
    top = max(0, min(top, frame_height - side))
    crop = frame[top : top + side, left : left + side].copy()

    interpolation = cv2.INTER_AREA if side >= output_size else cv2.INTER_CUBIC
    return cv2.resize(crop, (output_size, output_size), interpolation=interpolation)


def review_reason(label: str, metrics: dict[str, float]) -> str:
    if label == "normal" and metrics["ellipse_ratio"] < NORMAL_MIN_ELLIPSE_RATIO:
        return "normal_shape_irregular"

    if (
        label == "damaged"
        and metrics["ellipse_ratio"] >= DAMAGED_LOOKS_NORMAL_ELLIPSE
        and metrics["solidity"] >= DAMAGED_LOOKS_NORMAL_SOLIDITY
        and metrics["radial_cv"] <= DAMAGED_LOOKS_NORMAL_RADIAL_CV
    ):
        return "damaged_shape_looks_normal"

    return ""


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"이미지를 저장할 수 없습니다: {path}")


def process_ball_image(
    image_path: Path,
    label: str,
    output_dir: Path,
    output_size: int,
    padding: float,
) -> dict[str, str | float]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {
            "source_path": str(image_path),
            "label": label,
            "status": "review",
            "reason": "image_read_failed",
            "output_path": "",
        }

    metrics, _, detection_reason = find_ball(frame)
    if metrics is None:
        review_path = output_dir / "review" / label / image_path.name
        save_image(review_path, frame)
        return {
            "source_path": str(image_path),
            "label": label,
            "status": "review",
            "reason": detection_reason,
            "output_path": str(review_path),
        }

    crop = square_crop(
        frame,
        metrics["center_x"],
        metrics["center_y"],
        metrics["radius"] * 2.0,
        padding,
        output_size,
    )
    reason = review_reason(label, metrics)
    status = "review" if reason else "accepted"
    output_path = output_dir / status / label / image_path.name
    save_image(output_path, crop)

    row: dict[str, str | float] = {
        "source_path": str(image_path),
        "label": label,
        "status": status,
        "reason": reason or "ok",
        "output_path": str(output_path),
    }
    for name in (
        "area",
        "center_x",
        "center_y",
        "radius",
        "ellipse_ratio",
        "circle_ratio",
        "circularity",
        "solidity",
        "radial_cv",
    ):
        row[name] = round(float(metrics[name]), 5)
    return row


def process_empty_image(image_path: Path, output_dir: Path) -> dict[str, str | float]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {
            "source_path": str(image_path),
            "label": "empty",
            "status": "review",
            "reason": "image_read_failed",
            "output_path": "",
        }

    metrics, _, detection_reason = find_ball(frame)
    if metrics is None and detection_reason == "ball_not_found":
        status = "accepted"
        reason = "empty_ok"
    else:
        status = "review"
        reason = "ball_found_in_empty" if metrics is not None else detection_reason

    output_path = output_dir / status / "empty" / image_path.name
    save_image(output_path, frame)
    return {
        "source_path": str(image_path),
        "label": "empty",
        "status": status,
        "reason": reason,
        "output_path": str(output_path),
    }


def write_manifest(rows: list[dict[str, str | float]], path: Path) -> None:
    fieldnames = (
        "source_path",
        "label",
        "status",
        "reason",
        "output_path",
        "area",
        "center_x",
        "center_y",
        "radius",
        "ellipse_ratio",
        "circle_ratio",
        "circularity",
        "solidity",
        "radial_cv",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if args.size < 64:
        raise SystemExit("--size는 64 이상이어야 합니다.")
    if not (0.0 <= args.padding <= 1.0):
        raise SystemExit("--padding은 0.0에서 1.0 사이여야 합니다.")

    rows: list[dict[str, str | float]] = []
    for label in ("normal", "damaged"):
        label_dir = input_dir / label
        for image_path in sorted(label_dir.glob("*.jpg")):
            rows.append(
                process_ball_image(
                    image_path,
                    label,
                    output_dir,
                    args.size,
                    args.padding,
                )
            )

    for image_path in sorted((input_dir / "empty").glob("*.jpg")):
        rows.append(process_empty_image(image_path, output_dir))

    if not rows:
        raise SystemExit(f"처리할 JPG 사진이 없습니다: {input_dir}")

    manifest_path = output_dir / "preparation.csv"
    write_manifest(rows, manifest_path)

    counts = Counter((str(row["label"]), str(row["status"])) for row in rows)
    reasons = Counter(str(row["reason"]) for row in rows if row["status"] == "review")

    print("\n학습 사진 준비가 끝났습니다. 원본 사진은 변경하지 않았습니다.")
    for label in ("normal", "damaged", "empty"):
        print(
            f"  {label:<7} 승인 {counts[(label, 'accepted')]}장 / "
            f"검토 필요 {counts[(label, 'review')]}장"
        )
    if reasons:
        print("\n검토 사유:")
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}장")
    print(f"\n결과 폴더: {output_dir}")
    print(f"검토 목록: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
