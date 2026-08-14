# ort 使用经验与踩过的坑 (ONNX Runtime Pitfalls & Lessons)

> 本文档记录使用 `ort` Rust 库进行 ONNX 推理时的常见问题、已修复 bug 及调试经验。
>
> This document covers common issues, fixed bugs, and debugging lessons when using the `ort` Rust crate for ONNX inference.

---

## 1. 版本锁定 / Version Locking

### ndarray 版本必须匹配 ort 内部版本

**坑**：`ort` 内部使用 ndarray，若 Cargo.toml 中 ndarray 版本与 ort 期望不一致，会出现类型不兼容或 API 变化。

```toml
# ✅ 正确
ort = { version = "2.0.0-rc.13", features = ["directml"] }
ndarray = "0.17"  # 必须与 ort 内部版本一致

# ❌ 错误（升级后编译报错）
ndarray = "0.18"  # ort 2.0.0-rc.13 期望 0.17
```

---

## 2. Session::run 需要 &mut self

**坑**：早期代码用 `as_ref()` 拿 `&Session`，编译报错 "cannot borrow as mutable"。

```rust
// ✅ 正确
let session = self.session.as_mut().ok_or(InferenceError::NotLoaded)?;
let outputs = session.run(inputs!["input" => TensorRef::from_array_view(&input_tensor)?])?;

// ❌ 错误
let session = self.session.as_ref().unwrap();  // 编译失败
```

---

## 3. 图像预处理对齐 OpenCV

### 3.1 浮点转整型用 `.round()` 而非 `as u8`

**症状**：输出相对 Python 基线有 ~0.5 系统性偏移。

**根因**：cv2 用 `round()`，Rust `as u8` 是截断。

```rust
// ✅ 正确
let pixel: u8 = (float_value).round() as u8;

// ❌ 错误
let pixel: u8 = float_value as u8;  // 截断，差 0.5
```

### 3.2 JPEG 色度采样用 4:2:0

**症状**：输出文件比 Python 大 21%（87KB vs 197KB）。

**根因**：`image` crate 默认 4:4:4 色度采样，cv2 用 4:2:0。

```rust
use jpeg_encoder::{Encoder, SamplingFactor, ColorType};

let mut file = std::fs::File::create(&output_path)?;
let mut encoder = Encoder::new(&mut file, 95);
encoder.set_sampling_factor(SamplingFactor::R_4_2_0);  // 对齐 cv2
encoder.encode(rgb.as_raw(), w, h, ColorType::Rgb)?;
```

### 3.3 通道序处理

**症状**：换脸结果红蓝通道对调。

**根因**：inswapper 输出 RGB，但代码先转 BGR 又被 `paste_back` 当 RGB 处理。

```rust
// ✅ 正确：模型输出直接按 RGB 处理
let rgb_image = image::RgbImage::from_raw(width, height, swapped)?;

// ❌ 错误：额外转 BGR
let bgr = rgb_to_bgr(swapped)?;  // 多余转换导致通道错乱
```

---

## 4. 仿射变换方向

**症状**：换脸输出与输入完全一致（"没换成功"），因为 mask 全空。

**根因**：`paste_back` 用了**逆仿射** `transform_point(&inv, x, y)`，把原图像素映射到了模板外。

```rust
// ✅ 正确：正向仿射
transform_point(affine, x, y)

// ❌ 错误：逆仿射
transform_point(&inv, x, y)
```

**调试技巧**：打印 affine 矩阵，与 Python `cv2.estimateAffinePartial2D` 输出对比。

---

## 5. SCRFD Anchor Y 轴方向

**症状**：人脸对齐错误，landmarks Y 坐标翻转。

**根因**：`cy = (feat_h - 1 - i) * stride`（自下而上），而 SCRFD 输出是自上而下。

```rust
// ✅ 正确：自上而下
let cy = i * stride;

// ❌ 错误：自下而上
let cy = (feat_h - 1 - i) * stride;
```

---

## 6. 仿射求解器选择

**症状**：人脸旋转错误，姿态明显不对。

**根因**：手写 2x2 SVD 返回零旋转（r01=r10=0）。

```rust
// ✅ 正确：线性最小二乘，匹配 OpenCV estimateAffinePartial2D
fn estimate_similarity_transform(src: &[[f32; 2]], dst: &[[f32; 2]]) -> [[f32; 2]; 3] {
    // 实现线性求解器
}

// ❌ 错误：手写 SVD
fn bad_svd(...) -> ... {
    // 返回零旋转
}
```

---

## 7. GFPGAN 增强对象

**症状**：增强后结果被"拉回"原图，换脸效果不明显。

**根因**：GFPGAN 输入用的是**原图**而非换脸结果。

```rust
// ✅ 正确：增强换脸结果
let enhanced = enhancer.enhance(&swapped_result)?;
let final = blend(&swapped, &enhanced, 0.25, 0.75);

// ❌ 错误：增强原图
let enhanced = enhancer.enhance(&original)?;  // 会把结果拉回原图
```

---

## 8. ArcFace Warp 模板坐标

**坑**：ArcFace warp 模板的 landmarks 坐标写错（0.575/0.573 应为 0.824/0.823），导致 embedding 对齐错误。

```rust
// ✅ 正确模板坐标
const ARCFACE_WARP_TEMPLATE: [[f32; 2]; 5] = [
    [0.35478000, 0.25630000],   // left eye
    [0.64520000, 0.25630000],   // right eye
    [0.50000000, 0.40350000],   // nose
    [0.37913000, 0.55338000],   // left mouth
    [0.62087000, 0.55338000],   // right mouth
];

// ❌ 错误（下巴坐标错）
[0.575, 0.573]  // 应为 [0.824, 0.823]
```

---

## 9. 日志控制

```rust
// 抑制 ONNX Runtime 刷屏日志
std::env::set_var("ORT_LOGGING_LEVEL", "Error");

// 只记录 WARN/ERROR
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::WARN)
    .init();
```

---

## 10. 调试清单 / Debugging Checklist

遇到"换脸失败"时按顺序检查：

1. **输入路径是否正确？**
   - 检查 `srv_out.log` 中的 `Reading input image:` 行
   - 常见错误：文件名写错（a.jpg vs a.png）

2. **仿射矩阵是否合理？**
   - 打印 affine matrix，与 Python 输出对比
   - 确认 scale 接近 1.0，translation 在合理范围

3. **mask 是否非空？**
   - 如果 mean diff from input = 0.00，说明 mask 为空
   - 检查 paste_back 是否用了正向仿射

4. **通道序是否正确？**
   - 如果颜色异常（红蓝对调），检查 RGB/BGR 转换

5. **模型路径是否正确？**
   - 检查 models/ 目录是否有全部 5 个文件
   - 文件大小是否与预期一致

---

## 11. 验证基线 / Validation Baseline

Rust 输出 vs Python 基线期望相似度：

```
>87% 像素差值 ≤5
<1%  像素差值 >20
平均差 ~2.3
```

参考输出：`tests/fixtures/py_tinyface_baseline.jpg`
