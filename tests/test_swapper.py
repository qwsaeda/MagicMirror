import onnxruntime as ort
import numpy as np
import sys

model_path = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"

print("Loading model...")
sess = ort.InferenceSession(model_path)

# Print input/output details
for i, inp in enumerate(sess.get_inputs()):
    print(f"Input {i}: name={inp.name}, shape={inp.shape}, type={inp.type}")

for i, out in enumerate(sess.get_outputs()):
    print(f"Output {i}: name={out.name}, shape={out.shape}, type={out.type}")

# Create two different test inputs
target1 = np.random.randn(1, 3, 128, 128).astype(np.float32)
source1 = np.random.randn(1, 512).astype(np.float32)
target2 = np.random.randn(1, 3, 128, 128).astype(np.float32) * 2  
source2 = np.random.randn(1, 512).astype(np.float32) * 2

# Run with first input
result1 = sess.run(None, {"target": target1, "source": source1})[0]
print(f"\nResult1 shape: {result1.shape}")
print(f"Result1 min: {result1.min():.4f}, max: {result1.max():.4f}")
print(f"Result1 sample: {result1[0,0,0,0]:.4f}, {result1[0,0,0,1]:.4f}")

# Run with second input
result2 = sess.run(None, {"target": target2, "source": source2})[0]
print(f"\nResult2 shape: {result2.shape}")
print(f"Result2 min: {result2.min():.4f}, max: {result2.max():.4f}")

# Compare
diff = np.abs(result1 - result2).max()
print(f"\nMax difference: {diff:.6f}")
if diff < 0.001:
    print("OUTPUTS ARE ALMOST IDENTICAL - PROBLEM!")
else:
    print("Outputs are different - OK")