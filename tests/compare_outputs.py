import cv2
import numpy as np
import onnxruntime as ort

M = r"C:\Users\Administrator\MagicMirror\models"

# 加载模型
scrfd = ort.InferenceSession(f"{M}\\scrfd_2.5g.onnx")
arcface = ort.InferenceSession(f"{M}\\arcface_w600k_r50.onnx")
swapper = ort.InferenceSession(f"{M}\\inswapper_128_fp16.onnx")

# 加载权重
import onnx

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

onnx_model = onnx.load(f"{M}\\inswapper_128_fp16.onnx")
weight = onnx.numpy_helper.to_array(onnx_model.graph.initializer[-1])

# 模板
WARP_TEMPLATE = np.array([[0.36167656,0.40387734],[0.63696719,0.40235469],[0.50019687,0.56044219],[0.38710391,0.72160547],[0.61507734,0.72034453]])
ARC_TEMPLATE = np.array([[0.34191607,0.46157471],[0.65653393,0.45983393],[0.500225,0.64050536],[0.370975,0.57523],[0.63152143,0.57341857]])

def detect(img):
    h, w = img.shape[:2]
    # 缩放
    scale = 640 / max(h, w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    blob = cv2.dnn.blobFromImage(resized, 1.0/128.0, (640,640), (127.5,127.5,127.5), swapRB=True)
    outs = scrfd.run(None, {"input": blob})
    return outs, scale

def warp_face(img, landmark, tmpl, size):
    tmpl = tmpl * size
    M = cv2.estimateAffinePartial2D(landmark, tmpl, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    warped = cv2.warpAffine(img, M, (size,size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return warped, M

def get_embedding(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0/127.5, (112,112), (127.5,127.5,127.5), swapRB=True)
    emb = arcface.run(None, {"input": blob})[0][0]
    norm = np.linalg.norm(emb)
    if norm > 0: emb /= norm
    return emb

# 读取图片
img1 = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
img2 = cv2.imread(os.path.join(FIXTURES, "b.png"))
print(f"img1: {img1.shape}, img2: {img2.shape}")

# 检测
outs1, sc1 = detect(img1)
outs2, sc2 = detect(img2)

# 提取分数最高的检测
# SCRFD 输出: 0-2 scores, 3-5 boxes, 6-8 landmarks
# 取第一个尺度的第一个检测
best_score = 0
best_ldm = None
best_box = None
for scale_idx in range(3):
    scores = outs1[scale_idx].flatten()
    boxes = outs1[scale_idx+3].reshape(-1, 4)
    ldms = outs1[scale_idx+6].reshape(-1, 10)
    for i in range(len(scores)):
        if scores[i] > best_score and scores[i] > 0.05:
            best_score = scores[i]
            # 恢复坐标
            x1, y1, x2, y2 = boxes[i] / sc1
            ldm = ldms[i].reshape(5,2) / sc1
            best_box = (x1, y1, x2, y2)
            best_ldm = ldm

if best_ldm is None:
    print("No face detected")
    exit()

print(f"Best score: {best_score:.4f}")
print(f"Landmark: {best_ldm}")

# 对齐人脸（inswapper 128x128）
warped, M1 = warp_face(img1, best_ldm.astype(np.float32), WARP_TEMPLATE, 128)
print(f"Warped: {warped.shape}")

# 对齐目标人脸（ArcFace 112x112）
# 先检测目标人脸
outs2, sc2 = detect(img2)
best_score2 = 0
best_ldm2 = None
for scale_idx in range(3):
    scores = outs2[scale_idx].flatten()
    ldms = outs2[scale_idx+6].reshape(-1, 10)
    for i in range(len(scores)):
        if scores[i] > best_score2 and scores[i] > 0.05:
            best_score2 = scores[i]
            best_ldm2 = ldms[i].reshape(5,2) / sc2

if best_ldm2 is None:
    print("No target face")
    exit()

warped2, M2 = warp_face(img2, best_ldm2.astype(np.float32), ARC_TEMPLATE, 112)
print(f"Target warped: {warped2.shape}")

# 提取嵌入
emb = get_embedding(warped2)
print(f"Embedding[:5]: {emb[:5]}")

# 变换嵌入
transformed_emb = emb @ weight / np.linalg.norm(emb)
print(f"Transformed emb[:5]: {transformed_emb[:5]}")

# 运行 swapper
# 输入: target = 源图 (RGB, [0,1]), source = 变换后的嵌入
input_target = warped[:,:,::-1].transpose(2,0,1)[np.newaxis,...].astype(np.float32) / 255.0
input_source = transformed_emb.reshape(1, -1).astype(np.float32)
result = swapper.run(None, {"target": input_target, "source": input_source})[0][0]

# 输出解码: CHW->HWC, clip(0,1), RGB->BGR * 255
result = result.transpose(1,2,0).clip(0,1)[:,:,::-1] * 255
result = result.astype(np.uint8)
print(f"Swapped face: {result.shape}")

# 粘贴回原图
invM = cv2.invertAffineTransform(M1)
h, w = img1.shape[:2]
paste = cv2.warpAffine(result, invM, (w, h), borderMode=cv2.BORDER_REPLICATE)

# 简单融合
mask = np.ones((128,128), dtype=np.float32)
mask = cv2.warpAffine(mask, invM, (w, h))
mask = mask.clip(0, 1)
out = (img1.astype(np.float32) * (1 - mask[:,:,np.newaxis]) + paste * mask[:,:,np.newaxis]).astype(np.uint8)

cv2.imwrite(os.path.join(FIXTURES, "output1.jpg"), out)
print(f"Python output saved!")