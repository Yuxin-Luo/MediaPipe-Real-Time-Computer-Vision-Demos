"""Real-time full-body pose estimation demo (MediaPipe Tasks API).

Migrated from ``mp.solutions.pose`` to ``mediapipe.tasks.vision.PoseLandmarker``.
``POSE_CONNECTIONS`` only existed in the legacy namespace, so we list the
canonical 33-landmark skeleton pairs that match the original output.
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

MODEL_PATH = "models/pose_landmarker_lite.task"

WINDOW_NAME = "Pose Estimation"

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),                 # face-left
    (0, 4), (4, 5), (5, 6), (6, 8),                 # face-right
    (9, 10),                                       # mouth
    (11, 12), (11, 13), (13, 15),                   # left arm
    (12, 14), (14, 16),                             # right arm
    (11, 23), (12, 24),                             # torso-top
    (23, 24),                                      # shoulders
    (23, 25), (25, 27), (27, 29), (27, 31),         # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),         # right leg
    (15, 17), (15, 19), (15, 21), (17, 19),         # left hand
    (16, 18), (16, 20), (16, 22), (18, 20),         # right hand
]


def draw_pose(frame, pose_landmarks):
    """Render the 33-landmark pose skeleton on a BGR frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 128, 255), -1)


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

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
            for pose_landmarks in result.pose_landmarks:
                draw_pose(frame, pose_landmarks)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
