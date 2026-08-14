import os, time
print("step0: start", flush=True)
t0 = time.time()
from tinyface import TinyFace
print(f"step1: import tinyface {(time.time()-t0)*1000:.0f}ms", flush=True)
t0 = time.time()
tf = TinyFace()
print(f"step2: TinyFace() {(time.time()-t0)*1000:.0f}ms", flush=True)
t0 = time.time()
tf.config.face_detector_model = r"C:\Users\Administrator\MagicMirror\models\scrfd_2.5g.onnx"
tf.config.face_embedder_model = r"C:\Users\Administrator\MagicMirror\models\arcface_w600k_r50.onnx"
tf.config.face_swapper_model = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"
print(f"step3: set config {(time.time()-t0)*1000:.0f}ms", flush=True)
t0 = time.time()
tf.prepare()
print(f"step4: prepare (load models) {(time.time()-t0)*1000:.0f}ms", flush=True)
t0 = time.time()
import cv2

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

a = cv2.imread(os.path.join(FIXTURES, "a.jpg"))
print(f"step5: read a.jpg {(time.time()-t0)*1000:.0f}ms", flush=True)
t0 = time.time()
face_a = tf.get_one_face(a)
print(f"step6: get_one_face(a) {(time.time()-t0)*1000:.0f}ms", flush=True)
t0 = time.time()
b = cv2.imread(os.path.join(FIXTURES, "b.png"))
face_b = tf.get_one_face(b)
print(f"step7: get_one_face(b) {(time.time()-t0)*1000:.0f}ms", flush=True)
t0 = time.time()
out = tf.swap_face(vision_frame=a, reference_face=face_a, destination_face=face_b)
print(f"step8: swap_face {(time.time()-t0)*1000:.0f}ms, shape={out.shape}", flush=True)
t0 = time.time()
cv2.imwrite(r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "dbg_py_server_out.jpg")", out)
print(f"step9: save {(time.time()-t0)*1000:.0f}ms", flush=True)
print("ALL DONE", flush=True)