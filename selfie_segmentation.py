"""Real-time selfie segmentation demo (MediaPipe Tasks API).

Migrated from ``mp.solutions.selfie_segmentation`` to
``mediapipe.tasks.vision.ImageSegmenter``.

Earlier revisions of this demo referenced ``selfie_multiclass_256x256.tflite``
which Google has since 404'd on storage.googleapis.com. The demo uses the
binary ``selfie_segmenter_landscape`` instead — same binary semantics as the
legacy ``SelfieSegmentation(model_selection=0)``:

    category_mask == 1 → person (keep original pixel)
    category_mask == 0 → background (replace with BG_COLOR)
"""

import cv2
import numpy as np
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

MODEL_PATH = "models/selfie_segmenter_landscape.tflite"

WINDOW_NAME = "Selfie Segmentation"
BG_COLOR = (0, 255, 0)  # Green background, matching original demo.


def compose(frame_bgr, category_mask):
    """Replace background pixels in ``frame_bgr`` with solid ``BG_COLOR``.

    ``category_mask`` is HxW for the multiclass model and HxWx1 for the
    binary landscape one — squeeze the trailing axis if present.
    """
    if category_mask.ndim == 3:
        category_mask = category_mask[..., 0]
    condition = category_mask[..., None] == 1  # HxW -> HxWx1, True where person.
    background = np.full(frame_bgr.shape, BG_COLOR, dtype=np.uint8)
    return np.where(condition, frame_bgr, background)


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)

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
            result = segmenter.segment_for_video(mp_image, timestamp_ms)
            mask = result.category_mask.numpy_view()
            if mask.ndim in (2, 3):
                output = compose(frame, mask)
                cv2.imshow(WINDOW_NAME, output)
            else:
                cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        segmenter.close()


if __name__ == "__main__":
    main()
