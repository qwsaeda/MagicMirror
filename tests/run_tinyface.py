"""直接用 tinyface 库，不加载 enhancer，生成原 Python 效果"""
import os
os.environ["TINYFACE_MODELS_DIR"] = r"C:\Users\Administrator\MagicMirror\models"

from tinyface import TinyFace, config

# 配置模型路径，不设置 enhancer 避免下载
tf = TinyFace()
tf.config.face_detector_model = r"C:\Users\Administrator\MagicMirror\models\scrfd_2.5g.onnx"
tf.config.face_embedder_model = r"C:\Users\Administrator\MagicMirror\models\arcface_w600k_r50.onnx"
tf.config.face_swapper_model = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"
# 不设置 enhancer，避免下载
tf.prepare()

import cv2
a = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
b = cv2.imread(os.path.join(FIXTURES, "b.png"))
print(f"a: {a.shape}, b: {b.shape}")

# 用原 Python 的 face.py 相同的调用方式
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src-python"))
from magic.face import swap_face, _swap_face, _get_one_face, _read_image

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# 直接用原 Python 的 swap_face 函数
result = swap_face(os.path.join(FIXTURES, "a.jpg"), os.path.join(FIXTURES, "b.png"))
print(f"Result: {result}")

if result and os.path.exists(result):
    out = cv2.imread(result)
    print(f"Output shape: {out.shape}, file size: {os.path.getsize(result)}")
    # 复制到固定位置
    import shutil
    shutil.copy(result, os.path.join(FIXTURES, "py_tinyface_baseline.jpg"))
    print("Copied to py_baseline.jpg")
else:
    print("Failed, trying direct call...")
    # 直接调用
    import cv2, numpy as np
    from tinyface import TinyFace
    tf2 = TinyFace()
    tf2.config.face_detector_model = r"C:\Users\Administrator\MagicMirror\models\scrfd_2.5g.onnx"
    tf2.config.face_embedder_model = r"C:\Users\Administrator\MagicMirror\models\arcface_w600k_r50.onnx"
    tf2.config.face_swapper_model = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"
    tf2.prepare()
    
    src_face = tf2.get_one_face(a)
    dst_face = tf2.get_one_face(b)
    print(f"src_face score={src_face.score:.3f}, bbox={src_face.bbox}")
    print(f"dst_face score={dst_face.score:.3f}, bbox={dst_face.bbox}")
    
    out_img = tf2.swap_face(vision_frame=a, reference_face=src_face, destination_face=dst_face)
    cv2.imwrite(os.path.join(FIXTURES, "py_tinyface_baseline.jpg"), out_img)
    print(f"Output shape: {out_img.shape}, file size: {os.path.getsize(os.path.join(FIXTURES, 'py_tinyface_baseline.jpg'))}")