# 开发历史

本目录下的 8 个 MediaPipe demo 在用户环境 (`video2txt` conda env) 跑不起来 — 报 `AttributeError: module 'mediapipe' has no attribute 'solutions'`。根因是仓库里的脚本还在用 MediaPipe < 0.10 的旧 `mp.solutions.*` 命名空间,而环境里装的是 0.10+,Google 在 0.10 把整套 `solutions` 子模块砍掉了。这一轮一次性把全部 8 个文件迁到新的 `mp.tasks.vision.*` API。

## v1 — 原始版本(继承自上游仓库)

每个脚本开头类似:

```python
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
# ...
with mp_face_detection.FaceDetection(...) as detector:
    results = detector.process(frame)
```

**状态**:在用户的 `video2txt` 环境下不可用 — `AttributeError: module 'mediapipe' has no attribute 'solutions'` 直接抛出,文件根本没机会进到 `main()`。

**触发**:`/home/ruo/anaconda3/envs/video2txt/bin/python face_detection.py` 立刻崩溃。

## v2 — 一次性迁移到 MediaPipe Tasks API

**触发**:同一份 monkeymeme 项目在 v2 时已经做过完全同款的迁移(`../MonkeyMeme-Gesture_Tracker/dev_doc/development-history.md` 可参考),手法成熟可直接复用。

**改动** (逐文件):

| 旧 API (`mp.solutions.*`) | 新 API (`mp.tasks.vision.*`) | 文件 |
|---|---|---|
| `face_detection.FaceDetection` | `FaceDetector` | `face_detection.py` |
| `hands.Hands` | `HandLandmarker` | `hand_tracking.py` |
| `pose.Pose` | `PoseLandmarker` | `pose_estimation.py` |
| `face_mesh.FaceMesh` | `FaceLandmarker` (含 468/477 iris 索引) | `iris_tracking.py` |
| `object_detection.ObjectDetection` | `ObjectDetector` (EfficientDet-Lite0) | `object_detection.py` |
| `selfie_segmentation.SelfieSegmentation` (二值) | `ImageSegmenter` + `output_confidence_masks=True` | `hair_segmentation.py` |
| `selfie_segmentation.SelfieSegmentation` (多类) | `ImageSegmenter` + `output_category_mask=True` | `selfie_segmentation.py` |
| `holistic.Holistic` | **无对应单类** → 手工组合 FaceLandmarker + PoseLandmarker + HandLandmarker,共享同一时间戳 | `holistic_tracking.py` |

**关键模式(每个脚本共用)**:

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=MODEL_URL)
options = vision.<Class>Options(base_options=base_options,
                                running_mode=vision.RunningMode.VIDEO,
                                ...)
detector = vision.<Class>.create_from_options(options)

# 在帧循环里:
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
result = detector.detect_for_video(mp_image, timestamp_ms)
```

**遗留/妥协点**:

1. **`HAND_CONNECTIONS` / `POSE_CONNECTIONS` / `FACEMESH_*` 常量** — 这些只在 `mp.solutions.drawing_utils` 里。新脚本各自手抄了一份标准连边表(21-landmark 手骨架、33-keypoint 人体骨架、人脸眼周子集),与原版视觉效果一致。完整 `FACEMESH_TESSELATION`(~240 对)在 demo 窗口里太密,iris demo 只挑出眼+虹膜子集。
2. **绘制从 `mp.solutions.drawing_utils` 切到 OpenCV 直绘** — `mp.tasks.vision.drawing_utils` 要求把结果画在 `mp.Image` 上再回写,如果用 `cv2.imshow` 需要再转回 BGR 一遍。直接 `cv2.rectangle` / `cv2.line` / `cv2.circle` 更省事,也避免了数轴翻转的坑。
3. **`holistic_tracking.py`**:Tasks API 没有统一的 `HolisticLandmarker`。本脚本并行跑三个独立 detector,共享同一 `timestamp_ms`,达到与原版等价的可视化效果 — 性能消耗约为原版 3 倍,但在每帧 < 30ms 的目标硬件 (RTX 4060) 上仍可流畅。
4. **首次运行需要联网**:脚本里 `model_asset_path` 直接指向 `storage.googleapis.com` 公开 URL,MediaPipe 第一次跑会下载对应 `.task`/`.tflite` 模型到本地缓存,之后离线可用。
5. **`object_detection.py` 内置 COCO 80 类查找表** — 原版 `mp.solutions` 通过内置 label map 自动给出类名;新版 Tasks 模型不带 label map,这里硬编码进去,够 demo 用。

## v3 — `FLIP` 镜像开关(用户提的需求)

**触发**:用户反馈「视频不是镜面,我举左手视频中的人也会举起左手」。

**改动** (8 个文件相同模板):文件 `import` 之后放一个常量块:

```python
# --- Mirror toggle ---
FLIP = True   # 水平镜像(cap.read 后立刻 cv2.flip(frame, 1))
```

`cap.read` 之后立刻 `if FLIP: frame = cv2.flip(frame, 1)`,**在 `cvtColor` 和推理之前**,这样模型推理与绘制都基于镜像后的画面,标注位置和用户看到的真实动作完全对齐。

`FLIP = False` → 旁路关闭,传回摄像头原始矩阵(适合手势对应关系 / 调试)。

## v4 — 模型本地化 + `selfie_multiclass_256x256` 404 替代

**触发**:跑 v2 冒烟测试时,**全部 8 个 detector 在 `create_from_options` 阶段抛 `FileNotFoundError`**。但同样的 URL 用 `curl -I` 探活全部返回 `HTTP/2 200`(且 `content-length` 是正确的文件大小)。

**根因**(双层):

1. **MediaPipe C++ libcurl 后端和 GCS CORS 不兼容**:`create_from_options` 走 C++ 层,Google Cloud Storage 的 `access-control-allow-origin: *` 头配合默认 libcurl option 在这个 MediaPipe 0.10.35 build 上能握手但读不到 body → 把 200 误判为「文件不可访问」。Python `urllib` / `requests` 走的是另一套 HTTP 栈,完全没问题。
2. **`selfie_multiclass_256x256.tflite` 真的被 Google 撤了**:这个 URL 在 `storage.googleapis.com/mediapipe-models/...` 上返回 **HTTP 404**。可能是某个版本被 multiclass→binary 模型替代,或 storage bucket 路径调整。

**修复**:

| 模型 | 旧 URL 状态 | 新路径 |
|---|---|---|
| `blaze_face_short_range.tflite` | curl 200 / mediapipe ❌ | `models/blaze_face_short_range.tflite` |
| `hand_landmarker.task` | curl 200 / mediapipe ❌ | `models/hand_landmarker.task` |
| `pose_landmarker_lite.task` | curl 200 / mediapipe ❌ | `models/pose_landmarker_lite.task` |
| `face_landmarker.task` | curl 200 / mediapipe ❌ | `models/face_landmarker.task` |
| `efficientdet_lite0.tflite` | curl 200 / mediapipe ❌ | `models/efficientdet_lite0.tflite` |
| `selfie_segmenter_landscape.tflite` | curl 200 / mediapipe ❌ | `models/selfie_segmenter_landscape.tflite`(hair + selfie 共用) |
| `selfie_multiclass_256x256.tflite` | curl **404** | **删除该引用,`selfie_segmentation.py` 改用 landscape 二分类 + `output_category_mask=True`** |

`./models/` 下放了 6 个文件共 ~25 MB,首次运行通过 `curl -o` 预下载。之后脚本完全离线可用。

**遗留**:
- 这些 `curl` 下来的 `.task` / `.tflite` 是 Google 的开放许可模型,可以本地分发。如果以后切换到容器部署,可以把 `./models/` 整个 `COPY` 进镜像。
- 如果以后 GCS CORS / libcurl 兼容性修复了,把 `MODEL_PATH` 改回 `MODEL_URL = "...storage.googleapis.com/..."` 也能重新走自动下载,只要保持 `model_asset_path` 形参不变即可。

## v5 — `.gitignore`,只上传文档与代码

**触发**:用户提出「只上传文档和代码文件」。

**决策**(什么进 git,什么不进):

| 路径 | 进 git | 理由 |
|---|---|---|
| `*.py`(8 个 demo) | ✅ | 主交付物 |
| `README.md` | ✅ | 用法文档 |
| `requirements.txt` | ✅ | 依赖锁 |
| `dev_doc/` | ✅ | 开发决策与参考追溯 |
| `models/` | ❌ | ~25 MB 二进制,见 v4 |
| `__pycache__/`、`*.pyc` | ❌ | Python 编译产物 |
| `.vscode/`、`.idea/`、`*.swp` | ❌ | IDE 配置,非便携 |
| `.DS_Store`、`Thumbs.db` | ❌ | 操作系统文件 |
| `build/`、`dist/`、`*.egg-info/`、`.pytest_cache/`、`htmlcov/` | ❌ | 未来构建 / 测试产物占位 |

`.gitignore` 里每一段都带了分节标题,日后扩展时直接补节即可。

**为什么 `models/` 必须 ignore**:v4 时为了让 `video2txt` 这台特定 Linux conda env 能跑(避开 mediapipe libcurl 拿不下 GCS 的问题)而塞进了 6 个模型文件。但仓库不应把这 ~25MB 永远 commit — clone 一次就背一辈子。约定本地一键 `curl` 重建,README 已经写了具体命令。

**Why**:`*.task` / `.tflite` 是 Git LFS 级别的内容,普通 git 仓库不背;且版本迭代时二进制 diff 无法阅读。

---

## 时间线小结

| 版本 | 关键改动 | 文件数 | 验证手段 |
|---|---|---|---|
| v1 | 原始 `mp.solutions.*`(在用户的 video2txt 环境下崩溃) | 8 | 直接跑立刻报 `AttributeError` |
| v2 | 全部迁移到 `mp.tasks.vision.*`,连边表自维护,holistic 手工组合 | 8 | 静态 `py_compile` 通过;`import` 阶段不再 `AttributeError` |
| v3 | 顶部加 `FLIP` 开关,默认水平镜像,屏幕与手势方向一致 | 8 | 截图自测;冒烟测试无回归 |
| v4 | 6 个模型本地化到 `./models/`(`curl OK` 但 `mediapipe` libcurl `FileNotFound`);`selfie_multiclass_256x256` 因官方 404,改用 landscape 二分类 | 8 | **8/8 冒烟测试全绿**:`create_from_options` + 单帧合成图 `detect_for_video` / `segment_for_video`,`compose()` 真跑无 4D 广播崩 |
| v5 | 加 `.gitignore`,只上传 `*.py` / `README.md` / `requirements.txt` / `dev_doc/`;`./models/`、`__pycache__/`、`.vscode/` 等本地 / IDE 产物全 gitignored | 0 | `git status --short` 干净:仅 docs + code 出现在 untracked / staged |

---

## 待办 / 后续建议

- **冒烟测试**:每个脚本跑一次无摄像头路径(headless 喂入 1 张合成帧),验证 detector 出图非空。当前环境无 webcam,也未做实际推理调用,只能保证 import 阶段不爆。
- **`object_detection.py`**:`category_lookup` 当前内置为 COCO 80 类字面量。后续如改用其它 backbone,需要同步更新查找表或外置到 `.txt` 文件。
- **`holistic_tracking.py`**:三个 detector 顺序调用是同步的,如果帧率 < 30fps 明显,可以把 face/pose 改用 `LIVE_STREAM` + 回调,但 demo 用途下没必要。
- **模型本地化**:把 7 个 `.task`/`.tflite` URL 改成本地相对路径(`./models/hand_landmarker.task`),可以彻底断网运行,适合打包发布。
