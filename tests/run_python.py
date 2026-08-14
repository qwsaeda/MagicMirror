import cv2
import numpy as np
import os, sys

# 指定本地模型路径，避免下载
os.environ["TINYFACE_MODELS_DIR"] = r"C:\Users\Administrator\MagicMirror\models"

from tinyface import TinyFace

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


tf = TinyFace()
tf.config.face_detector_model = r"C:\Users\Administrator\MagicMirror\models\scrfd_2.5g.onnx"
tf.config.face_embedder_model = r"C:\Users\Administrator\MagicMirror\models\arcface_w600k_r50.onnx"
tf.config.face_swapper_model = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"
# 不设置 enhancer 避免下载
tf.prepare()

# 读取用户图片
a = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
b = cv2.imread(os.path.join(FIXTURES, "b.png"))
print(f"a: {a.shape}, b: {b.shape}")

# 获取人脸
src_face = tf.get_one_face(a)
dst_face = tf.get_one_face(b)
print(f"Source face score: {src_face.score:.4f}, bbox: {src_face.bbox}")
print(f"Target face score: {dst_face.score:.4f}, bbox: {dst_face.bbox}")
print(f"Source landmark: {src_face.landmark_5}")

# 执行换脸（Python 标准版）
result = tf.swap_face(
    vision_frame=a,
    reference_face=src_face,
    destination_face=dst_face,
)

# 保存
out_path = os.path.join(FIXTURES, "output1.jpg")
cv2.imwrite(out_path, result)
print(f"Python output saved: {out_path}")
print(f"Output size: {result.shape}")
print(f"File size: {os.path.getsize(out_path)} bytes")