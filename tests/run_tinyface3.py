from tinyface import TinyFace
import cv2, numpy as np, os

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


tf = TinyFace()
tf.config.face_detector_model = r"C:\Users\Administrator\MagicMirror\models\scrfd_2.5g.onnx"
tf.config.face_embedder_model = r"C:\Users\Administrator\MagicMirror\models\arcface_w600k_r50.onnx"
tf.config.face_swapper_model = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"
if os.path.exists(r"C:\Users\Administrator\MagicMirror\models\gfpgan_1.4.onnx"):
    tf.config.face_enhancer_model = r"C:\Users\Administrator\MagicMirror\models\gfpgan_1.4.onnx"
tf.prepare()

# 真实文件名：a.jpg 和 b.png
a = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
b = cv2.imread(os.path.join(FIXTURES, "b.png"))
print(f"a: {a.shape}, b: {b.shape}")

src = tf.get_one_face(a)
dst = tf.get_one_face(b)
print(f"src score={src.score:.3f}, src_ldm={src.landmark_5}")
print(f"dst score={dst.score:.3f}, dst_ldm={dst.landmark_5}")
print(f"src embedding[:5]={src.embedding[:5]}")
print(f"dst embedding[:5]={dst.embedding[:5]}")

result = tf.swapper.swap_face(a, dst, src)
cv2.imwrite(os.path.join(FIXTURES, "py_tinyface_baseline.jpg"), result)
print(f"Saved baseline: {result.shape}, size={os.path.getsize(os.path.join(FIXTURES, 'py_tinyface_baseline.jpg'))} bytes")