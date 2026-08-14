import cv2
import numpy as np

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


py = cv2.imread(os.path.join(FIXTURES, "py_tinyface_baseline.jpg"))
rust = cv2.imread(r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src-server", "a_output.jpg")")

print(f"Python: {py.shape}, {py.nbytes} bytes, file: 16844 bytes")
print(f"Rust: {rust.shape}, {rust.nbytes} bytes, file: 8852 bytes")

if py.shape == rust.shape:
    diff = np.abs(py.astype(np.float32) - rust.astype(np.float32))
    print(f"Max diff: {diff.max():.1f}")
    print(f"Mean diff: {diff.mean():.2f}")
    # 不同像素比例
    dm = diff > 20
    sim = diff < 5
    print(f"Pixels similar (<5): {sim.mean()*100:.1f}%")
    print(f"Pixels differing (>20): {dm.mean()*100:.1f}%")
    
    # 保存差异可视化
    diff_vis = (diff / diff.max() * 255).astype(np.uint8)
    cv2.imwrite(r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "diff_visual.jpg")", diff_vis)
    print(f"Diff visualization saved: diff_visual.jpg")
else:
    print(f"Size mismatch: {py.shape} vs {rust.shape}")