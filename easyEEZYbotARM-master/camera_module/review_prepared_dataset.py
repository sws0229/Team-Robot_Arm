"""모양이 애매한 사진을 직접 검토하고 최종 학습 목록을 만든다.

먼저 prepare_training_dataset.py를 실행해야 한다. 공 전체를 찾지 못한 사진은
자동 제외하고, 사람이 판단해야 하는 사진만 차례로 보여준다.

조작법
K 또는 Space: 학습 사진으로 사용
X: 학습에서 제외
S: 나중에 판단
Q 또는 Esc: 현재 판단을 저장하고 종료
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_PREPARED_DIR = PROJECT_DIR / "training_data" / "prepared"

DECISION_FIELDS = ("source_path", "label", "reason", "decision")
TRAINING_FIELDS = (
    "label",
    "image_path",
    "source_path",
    "preparation_status",
    "preparation_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="애매한 공 사진을 검토하고 최종 학습 목록을 만듭니다."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=DEFAULT_PREPARED_DIR,
        help=f"준비된 학습 사진 폴더 (기본값: {DEFAULT_PREPARED_DIR})",
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        help="이미 판단한 사진도 다시 표시합니다.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    return {row["source_path"]: row for row in read_csv(path)}


def save_decisions(path: Path, decisions: dict[str, dict[str, str]]) -> None:
    rows = sorted(decisions.values(), key=lambda row: row["source_path"])
    write_csv(path, rows, DECISION_FIELDS)


def make_review_display(row: dict[str, str], image, index: int, total: int):
    preview = cv2.resize(image, (448, 448), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(
        preview,
        112,
        36,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )
    filename = Path(row["source_path"]).name
    lines = (
        f"Review {index}/{total}  label={row['label']}",
        f"reason={row['reason']}",
        filename,
        "K/Space: KEEP   X: EXCLUDE   S: SKIP   Q: QUIT",
    )
    colors = ((0, 230, 255), (0, 180, 255), (230, 230, 230), (80, 255, 80))
    for line_index, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(
            canvas,
            line,
            (10, 25 + line_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_training_manifest(
    preparation_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
    path: Path,
) -> tuple[int, int]:
    training_rows = []
    counts = {"normal": 0, "damaged": 0}

    for row in preparation_rows:
        if row["label"] not in counts:
            continue

        include = row["status"] == "accepted"
        if row["status"] == "review":
            decision = decisions.get(row["source_path"], {}).get("decision", "")
            include = decision == "keep"
        if not include:
            continue

        training_rows.append(
            {
                "label": row["label"],
                "image_path": row["output_path"],
                "source_path": row["source_path"],
                "preparation_status": row["status"],
                "preparation_reason": row["reason"],
            }
        )
        counts[row["label"]] += 1

    write_csv(path, training_rows, TRAINING_FIELDS)
    return counts["normal"], counts["damaged"]


def main() -> int:
    args = parse_args()
    prepared_dir = args.prepared.expanduser().resolve()
    preparation_path = prepared_dir / "preparation.csv"
    decisions_path = prepared_dir / "review_decisions.csv"
    training_manifest_path = prepared_dir / "training_manifest.csv"

    preparation_rows = read_csv(preparation_path)
    if not preparation_rows:
        raise SystemExit(
            "preparation.csv가 없습니다. prepare_training_dataset.py를 먼저 실행하세요."
        )

    decisions = load_decisions(decisions_path)
    review_rows = [row for row in preparation_rows if row["status"] == "review"]

    # 공 전체가 잘린 사진이 없으므로 학습에서는 제외하지만 원본은 보존한다.
    for row in review_rows:
        if row["reason"] in {
            "ball_not_found",
            "multiple_balls",
            "white_mask_too_large",
            "image_read_failed",
            "ball_found_in_empty",
        }:
            decisions[row["source_path"]] = {
                "source_path": row["source_path"],
                "label": row["label"],
                "reason": row["reason"],
                "decision": "exclude",
            }

    manual_rows = [
        row
        for row in review_rows
        if row["reason"] in {
            "normal_shape_irregular",
            "damaged_shape_looks_normal",
        }
        and (args.review_all or row["source_path"] not in decisions)
    ]

    window_name = "Training Image Review"
    try:
        for index, row in enumerate(manual_rows, start=1):
            image = cv2.imread(row["output_path"])
            if image is None:
                decision = "exclude"
            else:
                display = make_review_display(row, image, index, len(manual_rows))
                cv2.imshow(window_name, display)
                decision = ""
                while not decision:
                    key = cv2.waitKey(0) & 0xFF
                    if key in (ord("k"), ord("K"), ord(" ")):
                        decision = "keep"
                    elif key in (ord("x"), ord("X")):
                        decision = "exclude"
                    elif key in (ord("s"), ord("S")):
                        decision = "skip"
                    elif key in (ord("q"), ord("Q"), 27):
                        save_decisions(decisions_path, decisions)
                        normal_count, damaged_count = build_training_manifest(
                            preparation_rows,
                            decisions,
                            training_manifest_path,
                        )
                        print("\n검토를 중간 저장하고 종료했습니다.")
                        print(f"현재 학습 목록: 정상 {normal_count}장 / 불량 {damaged_count}장")
                        return 0

            if decision != "skip":
                decisions[row["source_path"]] = {
                    "source_path": row["source_path"],
                    "label": row["label"],
                    "reason": row["reason"],
                    "decision": decision,
                }
                save_decisions(decisions_path, decisions)
    finally:
        cv2.destroyAllWindows()

    save_decisions(decisions_path, decisions)
    normal_count, damaged_count = build_training_manifest(
        preparation_rows,
        decisions,
        training_manifest_path,
    )
    undecided = sum(
        1
        for row in review_rows
        if row["reason"] in {"normal_shape_irregular", "damaged_shape_looks_normal"}
        and row["source_path"] not in decisions
    )

    print("\n학습 사진 검토 결과를 저장했습니다.")
    print(f"  정상 {normal_count}장 / 불량 {damaged_count}장")
    print(f"  아직 판단하지 않은 사진: {undecided}장")
    print(f"  판단 기록: {decisions_path}")
    print(f"  최종 학습 목록: {training_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
