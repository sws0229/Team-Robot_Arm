"""SVM 학습과 실시간 판정에서 함께 사용하는 특징 추출 모듈이다.

공 사진을 같은 크기로 정리하고 HOG·명암·엣지·외곽 특징을 추출한다.
직접 실행하지 않고 학습 코드와 auto5.py에서 불러서 사용한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = 64
LABEL_TO_NUMBER = {"normal": 0, "damaged": 1}
NUMBER_TO_LABEL = {value: key for key, value in LABEL_TO_NUMBER.items()}

HOG = cv2.HOGDescriptor(
    (IMAGE_SIZE, IMAGE_SIZE),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def make_filled_ball_mask(resized: np.ndarray) -> np.ndarray:
    """밝은 공의 외곽을 찾아 찌그러진 내부 선까지 포함해 채운다."""
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 210], dtype=np.uint8),
        np.array([179, 110, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        fallback = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.circle(
            fallback,
            (IMAGE_SIZE // 2, IMAGE_SIZE // 2),
            int(IMAGE_SIZE * 0.30),
            255,
            -1,
        )
        return fallback

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return filled


def extract_ball_features(image: np.ndarray) -> np.ndarray:
    """학습과 실시간 판정에 동일하게 사용할 특징값을 만든다."""
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    enhanced = CLAHE.apply(gray)

    # 공 바깥의 바닥 무늬와 그림자는 제외하고 공 내부의 찌그러진 선은 남긴다.
    ball_mask = make_filled_ball_mask(resized)
    focused = np.full_like(enhanced, 127)
    focused[ball_mask > 0] = enhanced[ball_mask > 0]

    hog_features = HOG.compute(focused).reshape(-1).astype(np.float32)
    appearance = cv2.resize(focused, (16, 16), interpolation=cv2.INTER_AREA)
    appearance_features = appearance.reshape(-1).astype(np.float32) / 255.0

    edges = cv2.Canny(focused, 40, 100)
    edge_map = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA)
    edge_features = edge_map.reshape(-1).astype(np.float32) / 255.0

    mask_map = cv2.resize(ball_mask, (16, 16), interpolation=cv2.INTER_AREA)
    mask_features = mask_map.reshape(-1).astype(np.float32) / 255.0

    return np.concatenate(
        (hog_features, appearance_features, edge_features, mask_features)
    ).astype(np.float32)


def crop_ball_square(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    output_size: int = 224,
    padding: float = 0.35,
) -> np.ndarray:
    """검은 여백을 만들지 않고 공 전체를 정사각형으로 잘라낸다."""
    frame_height, frame_width = frame.shape[:2]
    diameter = max(2.0, radius * 2.0)
    side = max(8, int(round(diameter * (1.0 + 2.0 * padding))))
    side = min(side, frame_width, frame_height)

    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(left, frame_width - side))
    top = max(0, min(top, frame_height - side))
    crop = frame[top : top + side, left : left + side].copy()

    interpolation = cv2.INTER_AREA if side >= output_size else cv2.INTER_CUBIC
    return cv2.resize(crop, (output_size, output_size), interpolation=interpolation)


class BallQualityClassifier:
    """OpenCV SVM 모델을 불러와 공 사진을 정상 또는 불량으로 판정한다."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"기계학습 모델이 없습니다: {self.model_path}")

        self.svm = cv2.ml.SVM_load(str(self.model_path))
        if self.svm is None or self.svm.empty():
            raise RuntimeError(f"기계학습 모델을 불러올 수 없습니다: {self.model_path}")

        expected_count = len(
            extract_ball_features(np.zeros((224, 224, 3), dtype=np.uint8))
        )
        model_count = int(self.svm.getVarCount())
        if model_count != expected_count:
            raise RuntimeError(
                "모델과 특징 추출 코드의 크기가 다릅니다: "
                f"model={model_count}, code={expected_count}"
            )

    def predict(self, image: np.ndarray) -> dict[str, object]:
        if image is None or image.size == 0:
            raise ValueError("분류할 공 이미지가 비어 있습니다.")

        features = extract_ball_features(image).reshape(1, -1)
        _, prediction = self.svm.predict(features)
        predicted_number = int(round(float(prediction[0, 0])))
        if predicted_number not in NUMBER_TO_LABEL:
            raise RuntimeError(f"알 수 없는 모델 출력값입니다: {predicted_number}")

        _, raw_output = self.svm.predict(
            features,
            flags=cv2.ml.STAT_MODEL_RAW_OUTPUT,
        )
        margin = abs(float(raw_output[0, 0]))
        label = NUMBER_TO_LABEL[predicted_number]
        return {
            "label": label,
            "label_number": predicted_number,
            "is_defective": label == "damaged",
            "margin": margin,
        }
