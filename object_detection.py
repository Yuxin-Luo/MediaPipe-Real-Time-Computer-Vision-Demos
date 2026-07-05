"""Real-time object detection demo (MediaPipe Tasks API).

Migrated from ``mp.solutions.object_detection`` to
``mediapipe.tasks.vision.ObjectDetector`` using the EfficientDet-Lite0 model
that the legacy demo mapped to ``min_detection_confidence=0.5``.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Mirror toggle ----------------------------------------------------------
# By default webcams show a raw (non-mirrored) video: when you raise your LEFT
# hand, the figure on screen also raises their LEFT hand — counter-intuitive for
# a selfie-style webcam where users expect the on-screen image to follow their
# own movements as a mirror. Flip the video horizontally so the on-screen image
# is a self-mirror of the user.
#
# Set FLIP = False to pass the raw camera matrix through unchanged (useful for
# debugging hand/pose correspondence, where inference should match what the
# camera sensor actually saw).
FLIP = True
# ---------------------------------------------------------------------------

MODEL_PATH = "models/efficientdet_lite0.tflite"

WINDOW_NAME = "Object Detection"
SCORE_THRESHOLD = 0.5


def draw_detections(frame, detections, category_lookup):
    """Draw bounding boxes + category labels on a BGR frame."""
    h, w = frame.shape[:2]
    for det in detections:
        score = det.categories[0].score if det.categories else 0.0
        if score < SCORE_THRESHOLD:
            continue
        cat = det.categories[0]
        name = category_lookup.get(cat.index, f"class_{cat.index}")
        bbox = det.bounding_box
        x, y, ww, hh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        label = f"{name} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw, y), (0, 255, 0), -1)
        cv2.putText(
            frame, label, (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        max_results=5,
        score_threshold=SCORE_THRESHOLD,
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # EfficientDet-Lite0 COCO labels (80 classes). Same set the old demo
    # implicitly relied on via the bundled legacy model.
    category_lookup = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
        24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
        28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
        32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
        36: "skateboard", 37: "surfboard", 38: "tennis racket",
        39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
        44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
        49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
        59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
        64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
        68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
        72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
        76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if FLIP:
                frame = cv2.flip(frame, 1)  # horizontal mirror — selfie view.
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, timestamp_ms)
            draw_detections(frame, result.detections, category_lookup)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
