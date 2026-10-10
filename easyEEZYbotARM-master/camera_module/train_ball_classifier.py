"""OpenCV로 정상·불량 탁구공 SVM 분류기를 학습한다.

HOG·명암·엣지·외곽 특징을 사용한다. 기본값은 기존 승인 사진과 실시간
피드백 사진을 합치며, ``--feedback-only``를 지정하면 현재 촬영 환경의
피드백 사진만 사용한다. 평가 후 최종 모델은 선택된 전체 사진으로 다시 학습한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PYTHON_PACKAGES_DIR = PROJECT_DIR / "python_packages"
if str(PYTHON_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGES_DIR))

from easyEEZYbotARM.ball_quality_ml import (  # noqa: E402
    IMAGE_SIZE,
    extract_ball_features as extract_features,
)

DEFAULT_PREPARED_DIR = PROJECT_DIR / "training_data" / "prepared"
DEFAULT_MANIFEST = DEFAULT_PREPARED_DIR / "training_manifest.csv"
DEFAULT_FEEDBACK_DIR = PROJECT_DIR / "training_data" / "live_feedback" / "crops"
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_MODEL = DEFAULT_MODEL_DIR / "ball_quality_svm_v3.xml"

LABEL_TO_NUMBER = {"normal": 0, "damaged": 1}
NUMBER_TO_LABEL = {value: key for key, value in LABEL_TO_NUMBER.items()}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCV SVM 탁구공 정상·불량 분류기를 학습하고 평가합니다."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"기존 승인 사진 목록 (기본값: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"저장할 SVM 모델 경로 (기본값: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=DEFAULT_FEEDBACK_DIR,
        help=f"실시간 피드백 사진 폴더 (기본값: {DEFAULT_FEEDBACK_DIR})",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="실시간 피드백 사진을 빼고 기존 승인 사진만 학습합니다.",
    )
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="기존 승인 사진을 제외하고 현재 피드백 사진만 학습합니다.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.25,
        help="평가용으로 분리할 원본 사진 비율 (기본값: 0.25).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="회전·반전·밝기 변경을 통한 데이터 증강 없이 학습합니다.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"학습 목록이 없습니다: {path}\n"
            "review_prepared_dataset.py를 먼저 끝까지 실행하세요."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = []
        for row in csv.DictReader(csv_file):
            if row.get("label") not in LABEL_TO_NUMBER or not row.get("image_path"):
                continue
            row["dataset_source"] = "prepared"
            rows.append(row)
    if not rows:
        raise SystemExit(f"학습할 사진이 없습니다: {path}")
    return rows


def load_feedback_rows(feedback_dir: Path) -> list[dict[str, str]]:
    if not feedback_dir.exists():
        raise SystemExit(
            f"피드백 사진 폴더가 없습니다: {feedback_dir}\n"
            "auto5.py의 2번 ML 피드백 모드로 사진을 먼저 촬영하세요."
        )

    rows = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for label in LABEL_TO_NUMBER:
        label_dir = feedback_dir / label
        if not label_dir.exists():
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in extensions:
                rows.append(
                    {
                        "label": label,
                        "image_path": str(image_path.resolve()),
                        "source_path": str(image_path.resolve()),
                        "dataset_source": "live_feedback",
                    }
                )

    if not rows:
        raise SystemExit(f"피드백 폴더에 학습할 사진이 없습니다: {feedback_dir}")
    return rows


def resolve_image_path(row: dict[str, str], manifest_path: Path) -> Path:
    image_path = Path(row["image_path"])
    if image_path.exists():
        return image_path

    if not image_path.is_absolute():
        relative_path = manifest_path.parent / image_path
        if relative_path.exists():
            return relative_path

    # 다른 컴퓨터의 절대경로가 저장된 목록은 현재 컴퓨터의 승인 폴더에서 찾는다.
    fallback = manifest_path.parent / "accepted" / row["label"] / image_path.name
    if fallback.exists():
        return fallback
    return image_path


def load_images(
    rows: list[dict[str, str]],
    manifest_path: Path,
) -> list[dict[str, object]]:
    samples = []
    for row in rows:
        image_path = resolve_image_path(row, manifest_path)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[건너뜀] 사진을 읽을 수 없습니다: {image_path}")
            continue
        samples.append(
            {
                "label": row["label"],
                "label_number": LABEL_TO_NUMBER[row["label"]],
                "image_path": str(image_path),
                "source_path": row.get("source_path", ""),
                "dataset_source": row.get("dataset_source", "prepared"),
                "image": image,
            }
        )
    return samples


def stratified_split(
    samples: list[dict[str, object]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_samples = []
    test_samples = []

    for label_number in sorted(NUMBER_TO_LABEL):
        label_samples = [
            sample for sample in samples if sample["label_number"] == label_number
        ]
        random.Random(seed + label_number).shuffle(label_samples)
        test_count = max(1, int(round(len(label_samples) * test_ratio)))
        test_samples.extend(label_samples[:test_count])
        train_samples.extend(label_samples[test_count:])

    random.Random(seed).shuffle(train_samples)
    random.Random(seed + 100).shuffle(test_samples)
    return train_samples, test_samples


def feedback_holdout_split(
    samples: list[dict[str, object]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """기존 승인 사진은 모두 학습하고 피드백 사진 일부만 평가용으로 분리한다."""
    prepared_samples = [
        sample for sample in samples if sample["dataset_source"] == "prepared"
    ]
    feedback_samples = [
        sample for sample in samples if sample["dataset_source"] == "live_feedback"
    ]
    feedback_train, feedback_test = stratified_split(
        feedback_samples,
        test_ratio,
        seed,
    )
    train_samples = prepared_samples + feedback_train
    random.Random(seed).shuffle(train_samples)
    return train_samples, feedback_test


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def change_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment(image: np.ndarray) -> list[np.ndarray]:
    return [
        image,
        rotate_image(image, -12.0),
        rotate_image(image, 12.0),
        cv2.flip(image, 1),
        cv2.flip(image, 0),
        change_brightness(image, 0.85),
        change_brightness(image, 1.15),
    ]


def build_training_arrays(
    samples: list[dict[str, object]],
    use_augmentation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for sample in samples:
        images = augment(sample["image"]) if use_augmentation else [sample["image"]]
        for image in images:
            features.append(extract_features(image))
            labels.append(int(sample["label_number"]))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def train_svm(features: np.ndarray, labels: np.ndarray):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria(
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10_000, 1e-6)
    )
    if not svm.train(features, cv2.ml.ROW_SAMPLE, labels):
        raise RuntimeError("SVM 학습에 실패했습니다.")
    return svm


def evaluate(svm, samples: list[dict[str, object]]):
    report_rows = []
    confusion = np.zeros((2, 2), dtype=np.int32)

    for sample in samples:
        features = extract_features(sample["image"]).reshape(1, -1)
        _, prediction = svm.predict(features)
        predicted_number = int(round(float(prediction[0, 0])))
        actual_number = int(sample["label_number"])
        confusion[actual_number, predicted_number] += 1
        report_rows.append(
            {
                "image_path": sample["image_path"],
                "source_path": sample["source_path"],
                "dataset_source": sample["dataset_source"],
                "actual": NUMBER_TO_LABEL[actual_number],
                "predicted": NUMBER_TO_LABEL[predicted_number],
                "correct": actual_number == predicted_number,
            }
        )

    correct = int(np.trace(confusion))
    total = int(np.sum(confusion))
    accuracy = correct / total if total else 0.0
    return accuracy, confusion, report_rows


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "image_path",
                "source_path",
                "dataset_source",
                "actual",
                "predicted",
                "correct",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def class_recall(confusion: np.ndarray, label_number: int) -> float:
    total = int(np.sum(confusion[label_number]))
    return float(confusion[label_number, label_number]) / total if total else 0.0


def main() -> int:
    args = parse_args()
    if args.no_feedback and args.feedback_only:
        raise SystemExit("--no-feedback과 --feedback-only는 함께 사용할 수 없습니다.")
    if not (0.10 <= args.test_ratio <= 0.40):
        raise SystemExit("--test-ratio는 0.10에서 0.40 사이여야 합니다.")

    manifest_path = args.manifest.expanduser().resolve()
    feedback_dir = args.feedback_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    report_path = manifest_path.parent / f"{model_path.stem}_report.csv"
    metadata_path = model_path.with_suffix(".json")

    rows = [] if args.feedback_only else load_manifest(manifest_path)
    if not args.no_feedback:
        rows.extend(load_feedback_rows(feedback_dir))
    samples = load_images(rows, manifest_path)
    label_counts = {
        label: sum(1 for sample in samples if sample["label"] == label)
        for label in LABEL_TO_NUMBER
    }
    if min(label_counts.values()) < 10:
        raise SystemExit(f"클래스별 사진이 너무 적습니다: {label_counts}")

    source_counts = {
        source: {
            label: sum(
                1
                for sample in samples
                if sample["dataset_source"] == source and sample["label"] == label
            )
            for label in LABEL_TO_NUMBER
        }
        for source in ("prepared", "live_feedback")
    }

    if not args.no_feedback:
        feedback_counts = source_counts["live_feedback"]
        if min(feedback_counts.values()) < 4:
            raise SystemExit(
                "피드백 사진이 클래스별로 최소 4장 필요합니다: "
                f"{feedback_counts}"
            )
        train_samples, test_samples = feedback_holdout_split(
            samples,
            args.test_ratio,
            args.seed,
        )
        split_strategy = (
            "feedback_only_holdout" if args.feedback_only else "live_feedback_holdout"
        )
    else:
        train_samples, test_samples = stratified_split(
            samples,
            args.test_ratio,
            args.seed,
        )
        split_strategy = "stratified_random"
    use_augmentation = not args.no_augmentation
    train_features, train_labels = build_training_arrays(
        train_samples,
        use_augmentation,
    )

    print("\n탁구공 정상·불량 분류기 학습을 시작합니다.")
    print(f"  전체 원본: 정상 {label_counts['normal']}장 / 불량 {label_counts['damaged']}장")
    if args.feedback_only:
        print("  기존 승인 사진: 제외")
    else:
        print(
            "  기존 승인 사진: "
            f"정상 {source_counts['prepared']['normal']}장 / "
            f"불량 {source_counts['prepared']['damaged']}장"
        )
    if not args.no_feedback:
        print(
            "  새 피드백 사진: "
            f"정상 {source_counts['live_feedback']['normal']}장 / "
            f"불량 {source_counts['live_feedback']['damaged']}장"
        )
        print(
            "  시험 방식: 현재 피드백 사진 중 "
            f"{args.test_ratio * 100:.0f}%를 학습에서 제외하여 평가"
        )
    print(f"  평가용 학습 원본: {len(train_samples)}장")
    print(f"  평가용 학습 입력(증강 포함): {len(train_labels)}장")
    print(f"  시험 원본: {len(test_samples)}장")

    evaluation_svm = train_svm(train_features, train_labels)
    accuracy, confusion, report_rows = evaluate(evaluation_svm, test_samples)

    # 분리한 사진으로 먼저 평가하고 실제 사용할 모델은 승인 사진 전체로 학습한다.
    deployment_features, deployment_labels = build_training_arrays(
        samples,
        use_augmentation,
    )
    deployment_svm = train_svm(deployment_features, deployment_labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    deployment_svm.save(str(model_path))
    write_report(report_path, report_rows)

    metadata = {
        "model_type": "OpenCV linear SVM",
        "feature_type": (
            "filled-ball mask + focused HOG + grayscale appearance + "
            "Canny edge map + contour mask"
        ),
        "image_size": IMAGE_SIZE,
        "label_to_number": LABEL_TO_NUMBER,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "augmentation": use_augmentation,
        "dataset_counts": label_counts,
        "dataset_source_counts": source_counts,
        "split_strategy": split_strategy,
        "evaluation_training_original_count": len(train_samples),
        "evaluation_training_input_count": len(train_labels),
        "test_count": len(test_samples),
        "test_accuracy": round(accuracy, 6),
        "confusion_matrix": confusion.tolist(),
        "training_original_count": len(samples),
        "training_input_count": len(deployment_labels),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    normal_recall = class_recall(confusion, LABEL_TO_NUMBER["normal"])
    damaged_recall = class_recall(confusion, LABEL_TO_NUMBER["damaged"])
    mistakes = [row for row in report_rows if not row["correct"]]

    print("\n시험 결과")
    print(f"  전체 정확도: {accuracy * 100:.1f}% ({len(test_samples) - len(mistakes)}/{len(test_samples)})")
    print(f"  정상 재현율: {normal_recall * 100:.1f}%")
    print(f"  불량 재현율: {damaged_recall * 100:.1f}%")
    print("  혼동 행렬 (행=실제, 열=예측)")
    print("                 예측 정상  예측 불량")
    print(f"    실제 정상       {confusion[0, 0]:>3}       {confusion[0, 1]:>3}")
    print(f"    실제 불량       {confusion[1, 0]:>3}       {confusion[1, 1]:>3}")

    if mistakes:
        print("\n틀린 시험 사진:")
        for row in mistakes:
            print(
                f"  {Path(str(row['image_path'])).name}: "
                f"실제 {row['actual']} -> 예측 {row['predicted']}"
            )
    else:
        print("\n시험 사진을 모두 맞혔습니다.")

    print(
        "\n최종 배포 모델 학습: "
        f"원본 {len(samples)}장 / 증강 포함 {len(deployment_labels)}장"
    )
    print(f"\n모델: {model_path}")
    print(f"모델 정보: {metadata_path}")
    print(f"시험 상세 기록: {report_path}")
    if not args.no_feedback:
        print("\n참고: 시험 점수는 현재 피드백 사진의 홀드아웃 결과입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
