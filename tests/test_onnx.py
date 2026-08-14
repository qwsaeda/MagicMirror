import onnxruntime as ort
import numpy as np

model_path = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"

print("Loading model...")
sess = ort.InferenceSession(model_path)

# Print input/output
for i, inp in enumerate(sess.get_inputs()):
    print(f"Input {i}: name={inp.name}, shape={inp.shape}, type={inp.type}")
for i, out in enumerate(sess.get_outputs()):
    print(f"Output {i}: name={out.name}, shape={out.shape}, type={out.type}")

# Test with two different inputs
target1 = np.random.rand(1, 3, 128, 128).astype(np.float32) * 2 - 1
source1 = np.random.rand(1, 512).astype(np.float32)
target2 = np.random.rand(1, 3, 128, 128).astype(np.float32) * 2 - 1
source2 = np.random.rand(1, 512).astype(np.float32)

r1 = sess.run(None, {"target": target1, "source": source1})[0]
r2 = sess.run(None, {"target": target2, "source": source2})[0]

print(f"\nResult1 shape: {r1.shape}, min={r1.min():.4f}, max={r1.max():.4f}")
print(f"Result2 shape: {r2.shape}, min={r2.min():.4f}, max={r2.max():.4f}")
diff = np.abs(r1 - r2).max()
print(f"Max diff: {diff:.6f}")
if abs(r1.max()) < 0.001 and abs(r1.min()) < 0.001:
    print("WARNING: Output is all zeros!")
else:
    print("Output looks valid")