import onnxruntime as ort
import numpy as np

# Test the converted model
model_path = r"C:\Users\Administrator\MagicMirror\models\inswapper_128.onnx"

print("Loading converted model...")
sess = ort.InferenceSession(model_path)

for i, inp in enumerate(sess.get_inputs()):
    print(f"Input {i}: name={inp.name}, shape={inp.shape}")
for i, out in enumerate(sess.get_outputs()):
    print(f"Output {i}: name={out.name}, shape={out.shape}")

target1 = np.random.rand(1, 3, 128, 128).astype(np.float32) * 2 - 1
source1 = np.random.rand(1, 512).astype(np.float32)
target2 = np.random.rand(1, 3, 128, 128).astype(np.float32) * 2 - 1
source2 = np.random.rand(1, 512).astype(np.float32)

r1 = sess.run(None, {"target": target1, "source": source1})[0]
r2 = sess.run(None, {"target": target2, "source": source2})[0]

print(f"\nResult1: min={r1.min():.6f}, max={r1.max():.6f}, mean={r1.mean():.6f}")
print(f"Result2: min={r2.min():.6f}, max={r2.max():.6f}, mean={r2.mean():.6f}")
diff = np.abs(r1 - r2).max()
print(f"Max diff between outputs: {diff:.6f}")

if abs(r1.max()) < 0.001 and abs(r1.min()) < 0.001:
    print("WARNING: Output is STILL all zeros!")
else:
    print("SUCCESS: Output is valid!")