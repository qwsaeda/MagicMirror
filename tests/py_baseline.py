# -*- coding: utf-8 -*-
"""标准 face swap：把 b.jpg 的人脸换到 a.png 上"""
import cv2
import numpy as np
import onnxruntime as ort
import onnx

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


M = r"C:\Users\Administrator\MagicMirror\models"
SIZE = 640
SCORE_THRESH = 0.1
NMS_THRESH = 0.5

scrfd = ort.InferenceSession(f"{M}\\scrfd_2.5g.onnx")
arcface = ort.InferenceSession(f"{M}\\arcface_w600k_r50.onnx")
swapper = ort.InferenceSession(f"{M}\\inswapper_128_fp16.onnx")
onnx_model = onnx.load(f"{M}\\inswapper_128_fp16.onnx")
weight = onnx.numpy_helper.to_array(onnx_model.graph.initializer[-1])

INSWAPPER_TMPL = np.array([[0.36167656,0.40387734],[0.63696719,0.40235469],[0.50019687,0.56044219],[0.38710391,0.72160547],[0.61507734,0.72034453]])
ARCFACE_TMPL = np.array([[0.34191607,0.46157471],[0.65653393,0.45983393],[0.500225,0.64050536],[0.370975,0.57523],[0.63152143,0.57341857]])


def preprocess(img):
    h, w = img.shape[:2]
    scale = min(SIZE / w, SIZE / h)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), 127.5, dtype=np.uint8)
    pad_x = (SIZE - new_w) // 2
    pad_y = (SIZE - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    blob = canvas[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    return blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32), scale, pad_x, pad_y


def generate_anchors(stride, anchor_total):
    feat_h = SIZE // stride
    feat_w = SIZE // stride
    anchors = []
    for i in range(feat_h):
        cy = (feat_h - 1 - i) * stride
        for j in range(feat_w):
            cx = j * stride
            for _ in range(anchor_total):
                anchors.append((cx, cy))
    return np.array(anchors, dtype=np.float32)


def detect(img):
    blob, scale, pad_x, pad_y = preprocess(img)
    outs = scrfd.run(None, {"input": blob})
    all_boxes = []
    strides = [8, 16, 32]
    for si, stride in enumerate(strides):
        scores = outs[si].reshape(-1)
        boxes = outs[si+3].reshape(-1, 4)
        kps = outs[si+6].reshape(-1, 10)
        anchors = generate_anchors(stride, 2)
        for i in range(len(scores)):
            if scores[i] < SCORE_THRESH:
                continue
            cx, cy = anchors[i]
            l, t, r, b = boxes[i] * stride
            x1 = (cx - l - pad_x) / scale; y1 = (cy - t - pad_y) / scale
            x2 = (cx + r - pad_x) / scale; y2 = (cy + b - pad_y) / scale
            ldm = np.zeros((5, 2))
            for j in range(5):
                ldm[j, 0] = (cx + kps[i, j*2] * stride - pad_x) / scale
                ldm[j, 1] = (cy + kps[i, j*2+1] * stride - pad_y) / scale
            all_boxes.append((scores[i], x1, y1, x2, y2, ldm))
    all_boxes.sort(key=lambda x: -x[0])
    keep = []
    for box in all_boxes:
        ok = all(iou(box, k) <= NMS_THRESH for k in keep)
        if ok: keep.append(box)
    return keep


def iou(a, b):
    x1 = max(a[1], b[1]); y1 = max(a[2], b[2])
    x2 = min(a[3], b[3]); y2 = min(a[4], b[4])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[3]-a[1]) * (a[4]-a[2])
    area_b = (b[3]-b[1]) * (b[4]-b[2])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def warp_face(img, landmark, tmpl, size):
    tmpl_scaled = tmpl * size
    M = cv2.estimateAffinePartial2D(landmark.astype(np.float32), tmpl_scaled.astype(np.float32),
                                    method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    warped = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return warped, M


def get_embedding(face_img, landmark):
    warped, _ = warp_face(face_img, landmark, ARCFACE_TMPL, 112)
    blob = warped[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    emb = arcface.run(None, {"input": blob})[0][0]
    norm = np.linalg.norm(emb)
    if norm > 0: emb = emb / norm
    return emb


def paste_back(original, swapped, M):
    invM = cv2.invertAffineTransform(M)
    h, w = original.shape[:2]
    mask = np.ones((128, 128), dtype=np.float32)
    inv_mask = cv2.warpAffine(mask, invM, (w, h), borderMode=cv2.BORDER_REPLICATE)
    inv_face = cv2.warpAffine(swapped, invM, (w, h), borderMode=cv2.BORDER_REPLICATE)
    inv_mask = np.clip(inv_mask, 0, 1)
    result = original.astype(np.float32).copy()
    for c in range(3):
        result[:, :, c] = inv_mask * inv_face[:, :, c] + (1 - inv_mask) * original[:, :, c]
    return result.astype(np.uint8)


def main():
    a = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
    b = cv2.imread(os.path.join(FIXTURES, "b.png"))
    print(f"a: {a.shape}, b: {b.shape}")

    a_faces = detect(a)
    b_faces = detect(b)
    print(f"a_faces: {len(a_faces)}, b_faces: {len(b_faces)}")
    if not a_faces or not b_faces:
        print("No faces"); return
    a_face = a_faces[0]
    b_face = b_faces[0]
    print(f"a_face bbox=({a_face[1]:.0f},{a_face[2]:.0f},{a_face[3]:.0f},{a_face[4]:.0f})")
    print(f"b_face bbox=({b_face[1]:.0f},{b_face[2]:.0f},{b_face[3]:.0f},{b_face[4]:.0f})")

    # 1. 提取 b 脸 embedding（身份）
    b_emb = get_embedding(b, b_face[5])
    print(f"b embedding[:5]: {b_emb[:5]}")
    temb = b_emb @ weight / np.linalg.norm(b_emb)
    print(f"transformed[:5]: {temb[:5]}")

    # 2. warp a.png 用 a 脸 landmark 到 128x128 (被修改的人脸)
    warped_a, M = warp_face(a, a_face[5], INSWAPPER_TMPL, 128)
    print(f"warped_a: {warped_a.shape}")

    # 3. swapper: target=warped_a, source=b 脸 embedding
    input_target = warped_a[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
    input_source = temb.reshape(1, -1).astype(np.float32)
    result = swapper.run(None, {"target": input_target, "source": input_source})[0][0]
    swapped_face = result.transpose(1, 2, 0).clip(0, 1)[:, :, ::-1] * 255
    swapped_face = swapped_face.astype(np.uint8)

    # 4. paste back 到 a.png
    final = paste_back(a, swapped_face, M)
    out_path = os.path.join(FIXTURES, "py_tinyface_baseline.jpg")
    cv2.imwrite(out_path, final)
    import os
    print(f"Saved: {out_path}, size: {final.shape}, file: {os.path.getsize(out_path)} bytes")
    open(os.path.join(FIXTURES, "py_meta.txt"), "w").write(
        f"a_landmark={a_face[5].tolist()}\nb_landmark={b_face[5].tolist()}\n"
        f"b_embedding={b_emb[:10].tolist()}\ntransformed={temb[:10].tolist()}\n"
        f"swapped_face_mean={swapped_face.mean():.2f}"
    )


if __name__ == "__main__":
    main()