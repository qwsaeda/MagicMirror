import cv2
import numpy as np
import onnxruntime as ort

# === 加载模型 ===
models = r"C:\Users\Administrator\MagicMirror\models"
arcface = ort.InferenceSession(f"{models}\\arcface_w600k_r50.onnx")
inswapper = ort.InferenceSession(f"{models}\\inswapper_128_fp16.onnx")

# 加载权重矩阵（last initializer）
import onnx

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

onnx_model = onnx.load(f"{models}\\inswapper_128_fp16.onnx")
weight = onnx.numpy_helper.to_array(onnx_model.graph.initializer[-1])

# === 参考 TinyFace 的模板 ===
INSWAPPER_TEMPLATE = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
])
ARCFACE_TEMPLATE = np.array([
    [0.34191607, 0.46157471],
    [0.65653393, 0.45983393],
    [0.5002250, 0.64050536],
    [0.3709750, 0.5752300],
    [0.63152143, 0.57341857],
])

# === 读取图片 ===
img1 = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
img2 = cv2.imread(os.path.join(FIXTURES, "b.png"))
print(f"img1: {img1.shape}, img2: {img2.shape}")

# === SCRFD 检测（用 onnxruntime） ===
def detect_face(img):
    scrfd = ort.InferenceSession(f"{models}\\scrfd_2.5g.onnx")
    h, w = img.shape[:2]
    # resize to 640
    scale = 640 / max(h, w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    blob = cv2.dnn.blobFromImage(resized, 1.0/128.0, (640, 640), (127.5, 127.5, 127.5), swapRB=True)
    outs = scrfd.run(None, {"input": blob})
    return outs, scale, resized

# 简单的手动 warp
def warp(img, landmarks, template, size):
    tmpl = template * size
    M = cv2.estimateAffinePartial2D(landmarks, tmpl, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    warped = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return warped, M

print("需要先安装 tinyface 才能获取 landmarks。请先 pip install tinyface")