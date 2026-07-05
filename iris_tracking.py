"""Real-time face-mesh + iris tracking demo (MediaPipe Tasks API).

Migrated from ``mp.solutions.face_mesh`` to
``mediapipe.tasks.vision.FaceLandmarker`` with ``output_face_blendshapes``
disabled for performance. The Tasks API equivalent of ``refine_landmarks=True``
(the legacy switch that exposed iris points 468..477) is the
``FaceLandmarkerOptions`` itself — by default the .task bundle includes the
attention mesh / iris refinement.
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

MODEL_PATH = "models/face_landmarker.task"

WINDOW_NAME = "Iris Tracking"

# Triangulation subset of FACEMESH_IRISES — pairs that keep the eye + iris
# regions visible (the full ~240-pair tessellation is overkill for a demo).
IRIS_EDGES = [
    # left eye contour
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133), (33, 133),
    # right eye contour
    (362, 263), (263, 249), (249, 390), (390, 373), (373, 374),
    (374, 380), (380, 381), (381, 382), (382, 362), (362, 382),
    # left iris
    (468, 469), (469, 470), (470, 471), (471, 472),
    # right iris
    (473, 474), (474, 475), (475, 476), (476, 477),
]


def draw_face_mesh(frame, face_landmarks):
    """Render the eye / iris sub-tessellation on a BGR frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
    for a, b in IRIS_EDGES:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 2, (0, 128, 255), -1)


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

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
            for face_landmarks in result.face_landmarks:
                draw_face_mesh(frame, face_landmarks)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
