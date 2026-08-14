import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

model_path = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"
output_path = r"C:\Users\Administrator\MagicMirror\models\inswapper_128.onnx"

print("Loading model...")
model = onnx.load(model_path)

# Convert all float16 initializers to float32
fp16_count = 0
for init in model.graph.initializer:
    if init.data_type == TensorProto.FLOAT16:
        fp16_data = np.frombuffer(init.raw_data, dtype=np.float16)
        fp32_data = fp16_data.astype(np.float32)
        init.raw_data = fp32_data.tobytes()
        init.data_type = TensorProto.FLOAT
        fp16_count += 1
print(f"Converted {fp16_count} float16 tensors")

# For Cast nodes that convert to/from float16, change the target type to float
cast_count = 0
from_float16 = 0
for node in model.graph.node:
    if node.op_type == "Cast":
        for attr in node.attribute:
            if attr.name == "to":
                if attr.i == 10:  # FLOAT16 -> FLOAT
                    attr.i = 1
                    cast_count += 1
                elif attr.i == 1:  # Keep FLOAT as is
                    cast_count += 1
                else:
                    print(f"  Unknown cast type: {attr.i}")

print(f"Updated {cast_count} Cast nodes")

# Update value_info to use float instead of float16
for vi in model.graph.value_info:
    if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT

# Update graph input/output types
for vi in model.graph.input:
    if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT

for vi in model.graph.output:
    if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT

# Save
onnx.save(model, output_path)

# Verify
try:
    onnx.checker.check_model(output_path)
    print("Model check passed!")
except Exception as e:
    print(f"Model check: {e}")

# Test
import onnxruntime as ort
print("Testing with onnxruntime...")
sess = ort.InferenceSession(output_path)

target = np.random.rand(1, 3, 128, 128).astype(np.float32) * 2 - 1
source = np.random.rand(1, 512).astype(np.float32)

result = sess.run(None, {"target": target, "source": source})[0]
print(f"Result: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")

if abs(result.max()) < 0.001 and abs(result.min()) < 0.001:
    print("WARNING: Output is STILL all zeros!")
else:
    print("SUCCESS: Model produces valid output!")