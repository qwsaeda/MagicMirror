import onnxruntime as ort
import numpy as np

model_path = r"C:\Users\Administrator\MagicMirror\models\inswapper_128_fp16.onnx"

print("Available providers:", ort.get_available_providers())
print()

sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Try with different input ranges
for scale in [1.0, 255.0, 127.5]:
    target = np.random.rand(1, 3, 128, 128).astype(np.float32) * scale
    source = np.random.rand(1, 512).astype(np.float32) * scale
    result = sess.run(None, {"target": target, "source": source})[0]
    print(f"Scale {scale}: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")

# Try with zeros
target = np.zeros((1, 3, 128, 128), dtype=np.float32)
source = np.zeros((1, 512), dtype=np.float32)
result = sess.run(None, {"target": target, "source": source})[0]
print(f"Zeros: min={result.min():.6f}, max={result.max():.6f}")

# Check if the model graph has any output nodes
print(f"\nNumber of nodes: {len(sess._model.graph.node)}")
print(f"Number of outputs: {len(sess.get_outputs())}")

# Check the last few nodes
for i, node in enumerate(sess._model.graph.node):
    if node.op_type == "Cast" or i > len(sess._model.graph.node) - 5:
        print(f"Node {i}: {node.op_type}, inputs={node.input}, outputs={node.output}")