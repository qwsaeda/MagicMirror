import cv2
import numpy as np

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


py = cv2.imread(os.path.join(FIXTURES, "output1.jpg"))
rust = cv2.imread(r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src-server", "a_output.jpg")")

print(f"Python: {py.shape}")
print(f"Rust: {rust.shape}")

if py.shape == rust.shape:
    diff = np.abs(py.astype(np.float32) - rust.astype(np.float32))
    diff_mask = diff > 20
    pct = diff_mask.mean() * 100
    print(f"Max diff: {diff.max():.2f}")
    print(f"Mean diff: {diff.mean():.2f}")
    print(f"Pixels differing (>20): {pct:.1f}%")
    if pct < 5:
        print("✅ Outputs are SIMILAR!")
    else:
        print("❌ Outputs still differ significantly")
else:
    print(f"Different sizes: {py.shape} vs {rust.shape}")