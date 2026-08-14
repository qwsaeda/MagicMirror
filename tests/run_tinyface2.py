"""直接用 tinyface 的核心组件，跳过 enhancer"""
import sys, os
import cv2, numpy as np
from tinyface import TinyFace
from tinyface.core.typing import Face

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# 创建 TinyFace 但不加载 enhancer
tf = TinyFace()
tf.config.face_detector_model = r"C:\Users\Administrator\MagicMirror\models\scrfd_2.5g.onnx"
tf.config.face_embedder_model = r"C:\Users\Administrator\MagicMirror\models\arcface_w600k_r50.onnx"
tf.config.face_swapper_model = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"
# 不设置 enhancer
tf.prepare(detection_only=False)  # 这会加载 enhancer

# 手动加载组件，跳过 enhancer
tf.detector.prepare()
tf.embedder.prepare()
tf.swapper.prepare()
# 跳过 enhancer

a = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
b = cv2.imread(os.path.join(FIXTURES, "b.png"))
print(f"a: {a.shape}, b: {b.shape}")

# 获取人脸
src_face = tf.get_one_face(a)
dst_face = tf.get_one_face(b)
print(f"src_face score={src_face.score:.3f}, bbox={src_face.bbox}")
print(f"dst_face score={dst_face.score:.3f}, bbox={dst_face.bbox}")
print(f"src_face landmark: {src_face.landmark_5}")
print(f"dst_face landmark: {dst_face.landmark_5}")

# 手动执行 swap_face（跳过 enhancer）
temp = a.copy()
# get_target_faces 在 a 中找相似人脸
# 因为 reference_face = src_face，vision_frame = a，所以返回 [src_face]
faces = [src_face]  # 直接使用源人脸（简化）

for face in faces:
    temp = tf.swapper.swap_face(temp, dst_face, face)
    # 跳过 enhancer

cv2.imwrite(os.path.join(FIXTURES, "py_tinyface_baseline.jpg"), temp)
print(f"Saved: py_tinyface_baseline.jpg, size: {temp.shape}, file: {os.path.getsize(os.path.join(FIXTURES, 'py_tinyface_baseline.jpg'))}")