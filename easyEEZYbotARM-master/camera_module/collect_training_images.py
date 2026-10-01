"""카메라로 탁구공 기계학습용 원본 사진을 수집한다.

로봇팔과 DB는 사용하지 않으며 사용자가 키를 눌러 실제 상태를 지정한다.

조작법
N: 정상 공 저장
D: 불량 공 저장
E: 공이 없는 배경 저장
Q 또는 Esc: 종료
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "training_data" / "raw"

LABEL_KEYS = {
    ord("n"): "normal",
    ord("d"): "damaged",
    ord("e"): "empty",
}

DISPLAY_COLORS = {
    "normal": (0, 220, 0),
    "damaged": (0, 0, 255),
    "empty": (0, 220, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="정상 공, 불량 공, 빈 배경 사진을 수집합니다."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="카메라 번호. 노트북 웹캠은 보통 0, USB 카메라는 보통 1입니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"사진 저장 폴더 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--flip",
        choices=("both", "horizontal", "vertical", "none"),
        default="both",
        help="화면 뒤집기 방식 (기본값: both).",
    )
    parser.add_argument("--width", type=int, help="요청할 카메라 영상 너비.")
    parser.add_argument("--height", type=int, help="요청할 카메라 영상 높이.")
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="수동 노출값. USB 카메라는 -5와 같은 값을 사용할 수 있습니다.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        choices=range(70, 101),
        metavar="70-100",
        help="JPEG 저장 품질 (기본값: 95).",
    )
    parser.add_argument(
        "--min-save-interval",
        type=float,
        default=0.4,
        help="키를 길게 눌렀을 때 중복 저장을 막는 최소 시간.",
    )
    return parser.parse_args()


def choose_camera_index(value: int | None) -> int:
    if value is not None:
        return value

    try:
        entered = input("카메라 번호를 입력하세요 (노트북 웹캠은 보통 0, 외부 카메라는 보통 1) [0]: ").strip()
    except EOFError:
        entered = ""

    if not entered:
        return 0

    try:
        return int(entered)
    except ValueError as exc:
        raise SystemExit("카메라 번호는 0 또는 1 같은 정수로 입력해야 합니다.") from exc


def open_camera(camera_index: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_ANY]
    if platform.system() == "Windows":
        backends.insert(0, cv2.CAP_DSHOW)

    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            return cap
        cap.release()

    raise RuntimeError(
        f"카메라 {camera_index}번을 열 수 없습니다. 다른 번호로 다시 실행해 주세요."
    )


def apply_flip(frame, mode: str):
    flip_codes = {
        "both": -1,
        "horizontal": 1,
        "vertical": 0,
    }
    if mode == "none":
        return frame
    return cv2.flip(frame, flip_codes[mode])


def ensure_dataset_dirs(output_dir: Path) -> None:
    for label in LABEL_KEYS.values():
        (output_dir / label).mkdir(parents=True, exist_ok=True)


def existing_counts(output_dir: Path) -> dict[str, int]:
    return {
        label: sum(1 for _ in (output_dir / label).glob("*.jpg"))
        for label in LABEL_KEYS.values()
    }


def save_labeled_frame(
    frame,
    label: str,
    output_dir: Path,
    metadata_path: Path,
    camera_index: int,
    jpeg_quality: int,
) -> Path:
    captured_at = datetime.now()
    filename = f"{label}_{captured_at.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    image_path = output_dir / label / filename

    saved = cv2.imwrite(
        str(image_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not saved:
        raise OSError(f"사진 저장에 실패했습니다: {image_path}")

    height, width = frame.shape[:2]
    new_file = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "captured_at",
                "label",
                "relative_path",
                "camera_index",
                "width",
                "height",
            ),
        )
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "captured_at": captured_at.isoformat(timespec="milliseconds"),
                "label": label,
                "relative_path": image_path.relative_to(output_dir).as_posix(),
                "camera_index": camera_index,
                "width": width,
                "height": height,
            }
        )

    return image_path


def draw_overlay(frame, counts: dict[str, int], status_text: str, status_color) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 112), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame,
        "N: Normal   D: Damaged   E: Empty   Q/Esc: Quit",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"Normal {counts['normal']}   Damaged {counts['damaged']}   "
            f"Empty {counts['empty']}"
        ),
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if status_text:
        cv2.putText(
            frame,
            status_text,
            (12, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            status_color,
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    camera_index = choose_camera_index(args.camera)
    output_dir = args.output.expanduser().resolve()
    metadata_path = output_dir / "metadata.csv"

    ensure_dataset_dirs(output_dir)
    counts = existing_counts(output_dir)

    try:
        cap = open_camera(camera_index)
    except RuntimeError as exc:
        print(f"[오류] {exc}")
        return 1

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)

    window_name = "Ping-pong Dataset Collector"
    last_save_at = 0.0
    status_text = "Ready"
    status_color = (255, 255, 255)
    status_until = 0.0
    failed_reads = 0

    print("\n탁구공 학습 사진 수집을 시작합니다.")
    print("  N: 정상 공  |  D: 불량 공  |  E: 공 없는 화면  |  Q 또는 Esc: 종료")
    print(f"  카메라: {camera_index}")
    print(f"  저장 위치: {output_dir}")
    print("공과 카메라를 조금씩 움직이고 조명 조건도 바꾸면서 촬영하세요.\n")

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 30:
                    print("[오류] 카메라 프레임을 연속 30회 읽지 못해 종료합니다.")
                    return 1
                time.sleep(0.05)
                continue

            failed_reads = 0
            clean_frame = apply_flip(raw_frame, args.flip)
            display_frame = clean_frame.copy()

            if time.monotonic() > status_until:
                status_text = "Ready"
                status_color = (255, 255, 255)
            draw_overlay(display_frame, counts, status_text, status_color)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            normalized_key = ord(chr(key).lower()) if key < 128 else key
            label = LABEL_KEYS.get(normalized_key)
            if label is None:
                continue

            now = time.monotonic()
            if now - last_save_at < args.min_save_interval:
                continue

            try:
                image_path = save_labeled_frame(
                    clean_frame,
                    label,
                    output_dir,
                    metadata_path,
                    camera_index,
                    args.jpeg_quality,
                )
            except (OSError, cv2.error) as exc:
                status_text = "SAVE FAILED"
                status_color = (0, 0, 255)
                status_until = now + 2.0
                print(f"[저장 실패] {exc}")
                continue

            counts[label] += 1
            last_save_at = now
            status_text = f"Saved: {label} #{counts[label]}"
            status_color = DISPLAY_COLORS[label]
            status_until = now + 1.2
            print(f"[저장] {label:<7} -> {image_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    total = sum(counts.values())
    print("\n사진 수집을 종료했습니다.")
    print(
        f"정상 {counts['normal']}장 / 불량 {counts['damaged']}장 / "
        f"빈 화면 {counts['empty']}장 / 전체 {total}장"
    )
    print(f"메타데이터: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
