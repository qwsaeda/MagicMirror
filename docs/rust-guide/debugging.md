# 调试方法与验证基线 (Debugging Methods & Validation Baseline)

> 本文档记录调试换脸管线的系统化方法，以及如何验证 Rust 实现的正确性。
>
> This document covers systematic debugging methods for the face swap pipeline and how to validate Rust implementation correctness.

---

## 1. 逐阶段对比法 / Stage-by-Stage Comparison

不要只对比最终图片，要逐阶段对比中间结果：

```
阶段 1: 人脸检测 (detector)
  ├─ 对比：bbox 坐标、landmarks 位置
  └─ 工具：画框截图对比

阶段 2: 仿射变换 (warp)
  ├─ 对比：affine matrix 数值
  └─ 工具：打印矩阵，与 Python 输出对比

阶段 3: 人脸嵌入 (embedder)
  ├─ 对比：embedding cos 相似度
  └─ 工具：cosine_similarity(emb_rust, emb_python)

阶段 4: 换脸结果 (swapper)
  ├─ 对比：mean abs diff
  └─ 工具：图像像素级对比

阶段 5: 增强结果 (enhancer)
  ├─ 对比：Laplacian variance（清晰度指标）
  └─ 工具：cv2.Laplacian(face, CV_64F).var()
```

---

## 2. 关键验证指标 / Key Validation Metrics

### 2.1 嵌入相似度 / Embedding Similarity

```python
# Python 计算 cos 相似度
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 期望 >0.99
```

### 2.2 图像相似度 / Image Similarity

```python
# Python 计算像素差
import cv2
import numpy as np

diff = cv2.absdiff(img_rust, img_python)
mean_diff = np.mean(diff)
pct_within_5 = np.mean(diff <= 5) * 100
pct_within_20 = np.mean(diff <= 20) * 100

# 期望指标：
# - >87% 像素差值 ≤5
# - <1% 像素差值 >20
# - 平均差 ~2.3
```

### 2.3 清晰度指标 / Sharpness Metric

```python
# Laplacian 方差（越大越清晰）
sharpness = cv2.Laplacian(face, cv2.CV_64F).var()

# 增强后期望提升 +50%
# Python baseline: ~103.9
# Rust optimized: ~159.6
```

---

## 3. 常见问题诊断 / Common Issue Diagnosis

### 问题 1: "No face swap"（输出等于原图）

**诊断步骤**：

```bash
# 1. 检查输入文件是否存在
cat srv_out.log | grep "Reading input image"

# 2. 检查 affine 矩阵
cat srv_out.log | grep "Affine matrix"

# 3. 检查 mask 是否为空
python -c "import cv2; img=cv2.imread('output.jpg'); print(cv2.mean(img))"
```

**可能原因**：
1. 输入路径错误 → 文件读取失败
2. affine 方向错误 → mask 为空
3. landmarks 翻转 → 对齐错误

### 问题 2: 颜色异常（红蓝对调）

**诊断**：
```bash
# 检查是否在某个阶段做了多余的 BGR↔RGB 转换
grep -n "bgr\|rgb\|swap" src-server/src/inference/*.rs
```

### 问题 3: 人脸位置错误

**诊断**：
```bash
# 检查 anchor Y 轴方向
grep -n "stride\|anchor" src-server/src/inference/detector.rs
```

---

## 4. 调试工具 / Debugging Tools

### 4.1 日志文件 / Log Files

| 文件 | 内容 | 用途 |
|------|------|------|
| `srv_out.log` | server stdout | 查看推理步骤、输入路径 |
| `srv_err.log` | server stderr | 查看错误信息、警告 |

### 4.2 测试脚本 / Test Scripts

```bash
# 运行换脸测试
python tests/compare_outputs.py --rust output_rust.jpg --python output_py.jpg

# 计算相似度
python tests/compare_diff.py --ref py_tinyface_baseline.jpg --test output_rust.jpg
```

### 4.3 可视化调试 / Visual Debugging

在代码中临时添加：
```rust
// 保存中间结果
image::save_as(&crop, "debug_warp.jpg").unwrap();
```

---

## 5. 性能基准 / Performance Benchmark

### 5.1 模型加载时间 / Model Load Time

| 模型 | 大小 | 加载时间 |
|------|------|---------|
| scrfd_2.5g.onnx | 3.3 MB | ~1s |
| arcface_w600k_r50.onnx | 166 MB | ~5s |
| inswapper_128_fp16.onnx | 265 MB | ~5s |
| gfpgan_1.4.onnx | 324 MB | ~15s |
| **合计** | **~760 MB** | **~30s** |

### 5.2 推理时间 / Inference Time

| 操作 | CPU | DirectML | CUDA |
|------|-----|----------|------|
| 人脸检测 | 80ms | 15ms | 8ms |
| 嵌入提取 | 50ms | 12ms | 5ms |
| 换脸 | 200ms | 50ms | 25ms |
| 增强 | 150ms | 40ms | 20ms |
| **总计** | **~480ms** | **~117ms** | **~58ms** |

---

## 6. 回归测试 / Regression Testing

确保改动不破坏已有功能：

```bash
# 运行所有测试
cd src-server
cargo test

# 对比输出
python tests/compare_outputs.py \
  --rust tests/fixtures/c_a_to_b.jpg \
  --python tests/fixtures/py_tinyface_baseline.jpg
```

**通过标准**：
- 单元测试全部通过
- 输出与基线相似度 >87%（像素差 ≤5）
- 平均像素差 <5

---

## 7. 快速排错流程 / Quick Troubleshooting Flow

```
用户反馈问题
    │
    ▼
查看 srv_out.log 和 srv_err.log
    │
    ├─ 文件不存在？ → 检查输入路径
    │
    ├─ affine 矩阵异常？ → 检查 warp.rs 实现
    │
    ├─ 颜色错误？ → 检查通道序处理
    │
    ├─ 没换脸？ → 检查 paste_back 仿射方向
    │
    └─ 仍然失败？ → 运行回归测试，对比 Python 基线
```
