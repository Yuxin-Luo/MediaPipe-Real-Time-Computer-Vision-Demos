# 🧠 MediaPipe Real-Time Computer Vision Demos

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0%2B-orange)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-blue)](https://opencv.org/)

A collection of **real-time computer vision demos** built with **MediaPipe Tasks API** and **OpenCV**. Each script showcases one MediaPipe task — face detection, hand tracking, pose estimation, iris / face-mesh, object detection, hair & selfie segmentation, and a composite holistic tracker. Designed for learning, prototyping, and rapid capability checks against a webcam feed.

## 📦 Contents

| File Name                  | Tasks API Class Used        | Description |
|----------------------------|------------------------------|-------------|
| `face_detection.py`        | `vision.FaceDetector`        | Detects human faces; draws bounding boxes + 6 keypoints. |
| `hand_tracking.py`         | `vision.HandLandmarker`      | Tracks up to two hands, visualizing 21 landmarks with skeleton connections. |
| `pose_estimation.py`       | `vision.PoseLandmarker`      | Estimates a 33-keypoint full-body skeleton. |
| `iris_tracking.py`         | `vision.FaceLandmarker`      | Tracks face mesh (478 landmarks) with iris sub-tessellation. |
| `object_detection.py`      | `vision.ObjectDetector`      | Runs EfficientDet-Lite0, draws boxes + COCO labels + scores. |
| `hair_segmentation.py`     | `vision.ImageSegmenter`      | Uses selfie landscape model as a hair proxy. |
| `selfie_segmentation.py`   | `vision.ImageSegmenter`      | Replaces the background with a solid color (green). |
| `holistic_tracking.py`     | Face + Pose + Hand Landmarker| Composes face, pose, and hand landmarks in one window. |

## 🎯 Features

- 🧑‍🦰 **Face Detection** — bounding boxes with keypoints
- 🖐️ **Hand Tracking** — 21-point skeleton per hand
- 🕺 **Pose Estimation** — full-body 33-keypoint skeleton
- 👁️ **Iris Tracking** — face mesh + iris landmarks (gaze, attention)
- 🖼️ **Hair Segmentation** — uses selfie model as proxy for hair region
- 🎥 **Selfie Segmentation** — solid-color virtual backgrounds
- 🔁 **Holistic Tracking** — face + pose + hands composed in one pipeline

## ⚙️ Installation

### Prerequisites
- **Python**: 3.9 or higher.
- **Hardware**: a webcam, or modify the scripts to point at a video file
  (`cv2.VideoCapture('video.mp4')`).
- **OS**: Windows / macOS / Linux.

### Setup

```bash
# Activate the target conda env (example uses the user's `video2txt` env)
conda activate video2txt

pip install -r requirements.txt
```

Or manually:

```bash
pip install "mediapipe>=0.10.0" opencv-python numpy
```

The demo scripts load model files from the local `./models/` directory — the
`.task` / `.tflite` files were pre-downloaded into this folder at build time
because MediaPipe's bundled libcurl HTTP client cannot reach Google's public
model CDN on some Linux conda environments, even when Python `curl`/`urllib`
work fine. See `dev_doc/development-history.md` (v4) for full diagnosis.
Net result: **the demos run fully offline** after `pip install`.

> ⚠️ **`./models/` is `.gitignore`d** — the 6 model files (~25 MB) sit only on
> the local checkout. Recreate them once via:
>
> ```bash
> mkdir -p models && cd models && \
>   curl -sS -O https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite && \
>   curl -sS -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task && \
>   curl -sS -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task && \
>   curl -sS -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task && \
>   curl -sS -O https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite && \
>   curl -sS -O https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite
> ```

## 🚀 Usage

```bash
python face_detection.py
```

Replace `face_detection.py` with any other script (e.g. `hand_tracking.py`,
`pose_estimation.py`). Each opens a window named after its task; press **q**
to quit.

### 🪞 Mirror toggle

Every script exposes a top-of-file constant:

```python
FLIP = True   # mirror the webcam horizontally (selfie view, default)
FLIP = False  # raw camera matrix, no horizontal flip
```

By default the scripts mirror the frame so that raising your **left** hand
shows the figure on screen raising **their left hand too** — the natural
selfie-view convention. Set `FLIP = False` if you want the unmirrored
camera matrix (useful when debugging whether an inference result was
computed against the frame you actually see).

## 🧠 API Migration Note

The 8 demos in this folder originally targeted MediaPipe's legacy
`mp.solutions.*` namespace, which was removed in MediaPipe 0.10. They have
been rewritten on top of `mediapipe.tasks.vision.*` so they run unmodified
on MediaPipe >= 0.10:

- `mp.solutions.face_detection` → `vision.FaceDetector`
- `mp.solutions.hands`           → `vision.HandLandmarker`
- `mp.solutions.pose`            → `vision.PoseLandmarker`
- `mp.solutions.face_mesh`       → `vision.FaceLandmarker`
- `mp.solutions.object_detection` → `vision.ObjectDetector`
- `mp.solutions.selfie_segmentation` → `vision.ImageSegmenter`
- `mp.solutions.holistic`        → composed: Face + Pose + Hand Landmarker

`HAND_CONNECTIONS` / `POSE_CONNECTIONS` / `FACEMESH_TESSELATION` constants
only existed in the legacy namespace, so the new scripts carry inline tables
of the standard connection pairs. Full historical rationale lives in
[`dev_doc/development-history.md`](dev_doc/development-history.md).

## 🛠 Troubleshooting

- **Webcam unavailable** — try a different index (`cv2.VideoCapture(1)`) or a video file path.
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`** —
  you are running the *old* versions of these scripts. Pull the migrated
  versions from this folder.
- **First run is slow** — MediaPipe downloads the model file on first use;
  subsequent runs use the local cache.
- **Hair quality** — `hair_segmentation.py` is a proxy (no dedicated hair model in MediaPipe); switch to `selfie_segmentation.py` for cleaner body extraction.

## 📚 References

- [MediaPipe Tasks — Python vision API](https://developers.google.com/mediapipe/solutions/vision/mediapipe_vision_sdk)
- [MediaPipe Studio (live model playground)](https://mediapipe-studio.webapps.google.com/)
- [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [MediaPipe GitHub repository](https://github.com/google-ai-edge/mediapipe)

## 📝 License

MIT. See [LICENSE](LICENSE) if present, otherwise the standard MIT terms.
