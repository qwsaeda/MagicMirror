# Rust 推理服务实现指南 (Rust Server Implementation Guide)

> 本文档全面介绍 MagicMirror 的 Rust 推理服务（`magic-server`）实现，包括：为什么用 Rust、整体架构、ONNX Runtime (ort) 的使用方式、踩过的坑与经验教训。
>
> This document covers the Rust inference server (`magic-server`): why Rust, architecture, how to use ONNX Runtime via `ort`, pitfalls we hit, and hard-won lessons.

---

## 目录 (Table of Contents)

- [1. 为什么用 Rust / Why Rust](#1-为什么用-rust--why-rust)
- [2. 架构总览 / Architecture](#2-架构总览--architecture)
- [3. ONNX Runtime 使用方式 / ort Usage](#3-onnx-runtime-使用方式--ort-usage)
- [4. 踩过的坑 / Pitfalls](#4-踩过的坑--pitfalls)
- [5. 经验总结 / Lessons Learned](#5-经验总结--lessons-learned)
- [6. 常见问题 FAQ](#6-常见问题-faq)

---

## 1. 为什么用 Rust / Why Rust

### 中文

MagicMirror 原始版本使用 Python + Nuitka 打包推理服务。本 fork 将推理层整体重写为纯 Rust，原因如下：

| 维度 | Rust | Python |
|------|------|--------|
| **启动速度** | 单二进制，毫秒级启动 | 解释器 + 依赖加载，秒级 |
| **内存占用** | 常驻 ~300MB（含模型共享） | 更高，解释器开销 |
| **部署形态** | 单个 `server.exe`，无运行环境依赖 | 需打包 Python 运行时 + 依赖 |
| **包体积** | ~24MB 单文件 | Nuitka standalone 数百 MB |
| **性能** | 接近 C，推理管线零胶水开销 | 解释执行，图像处理慢 |
| **安全/稳定** | 内存安全，无 GIL，天然并发 | GIL 限制，多进程复杂度高 |

**核心优势总结：**
- **无环境依赖**：`server.exe` 是自包含二进制，配合 ONNX Runtime DirectML 后端，用户无需安装 Python、CUDA 等任何运行时。
- **进程模型简单**：Tauri 桌面端直接派生 `server.exe` 子进程并托管生命周期，无需复杂的 Python 环境检查。
- **开发体验**：`cargo build` 即产物，类型系统让图像管线的错误在编译期暴露（如维度不匹配、通道序错误）。

### English

The original MagicMirror used Python + Nuitka for the inference server. This fork rewrites the entire inference layer in pure Rust:

- **Zero runtime dependencies**: a single self-contained `server.exe`, no Python/CUDA needed.
- **Fast startup & low memory**: single binary, ~300MB resident with shared models.
- **Simpler process model**: Tauri spawns/kills `server.exe` directly as a child process.
- **Compile-time safety**: the type system catches tensor-shape and channel-order bugs before runtime.

---

## 2. 架构总览 / Architecture

### 2.1 目录结构 / Directory layout

```
src-server/
├── Cargo.toml          # 依赖清单
└── src/
    ├── main.rs         # Axum HTTP 入口 + 命令行参数 + CORS
    ├── lib.rs          # 库入口（暴露 inference 模块）
    ├── worker.rs       # Worker 池 + 自动并发数计算
    └── inference/      # 推理管线
        ├── mod.rs      # TinyFace 编排器（协调各模型）
        ├── detector.rs # SCRFD 人脸检测
        ├── embedder.rs # ArcFace 人脸嵌入
        ├── swapper.rs  # inswapper 人脸交换
        ├── enhancer.rs # GFPGAN 质量增强（可选）
        ├── warp.rs     # 仿射变换（对齐 OpenCV estimateAffinePartial2D）
        └── image_utils.rs # 图像工具函数
```

### 2.2 推理管线 / Inference pipeline

```
输入图 ──▶ SCRFD 检测 ──▶ 提取最大人脸框 + landmarks
                                │
目标图 ──▶ SCRFD 检测 ──▶ 提取目标人脸框 + landmarks
                                │
                                ▼
                    ArcFace 提取目标身份 embedding
                                │
                                ▼
                    inswapper 融合两张人脸（128x128 模板）
                                │
                                ▼
                    GFPGAN 增强换脸结果（可选，0.25*swap + 0.75*enhanced）
                                │
                                ▼
                    paste_back 正向仿射贴回原图 ──▶ 输出 JPG (4:2:0, 质量95)
```

### 2.3 Worker 并发模型 / Concurrency model

```
HTTP /task ──▶ mpsc::channel(10) ──▶ Worker(Arc<Mutex<TinyFace>>)
                                         │
                                    [ 推理 ]
                                         │
                                    mpsc::Sender<TaskResult> ──▶ HTTP 响应
```

- 模型只加载一次，所有 worker 共享（`Arc<Mutex<TinyFace>>`）。
- `ONNX Session::run` 需要 `&mut self`，因此用 Mutex 保证同一时刻只有一个推理在跑（单 worker 默认）。
- 并发数自动计算：`min(CPU核数, 可用内存/300MB)`，计算密集任务不能超过核数。

**命令行参数：**

```bash
server.exe                # 默认 1 worker
server.exe --workers 4    # 指定 4 个
server.exe --workers auto # 自动计算
```

### English

- Models load **once** and are shared via `Arc<Mutex<TinyFace>>`.
- `Session::run` needs `&mut self`, so a `Mutex` serializes inference.
- Auto worker count: `min(CPU cores, available_mem / 300MB)` — compute-bound, never exceed cores.

---

## 3. ONNX Runtime 使用方式 / ort Usage

> 本项目使用 `ort = { version = "2.0.0-rc.13", features = ["directml"] }`。
> We use `ort` 2.0.0-rc.13 with the `directml` feature.

### 3.1 加载模型 / Loading models

```rust
// 使用 OnnxModel trait 统一接口
trait OnnxModel {
    fn load(&mut self, model_path: impl AsRef<Path>) -> InferenceResult<()>;
    fn prepare(&self) -> InferenceResult<()>;
    fn session(&self) -> Option<&Session>;
}

// 每个模型组件持有一个 Session
pub struct Detector {
    session: Option<Session>,
    input_shape: [usize; 4], // [1, 3, H, W]
}
```

```rust
// 实际加载
let session = ort::Session::builder()
    .commit_from_file(model_path)?;   // rc.13 API
```

### 3.2 推理 / Running inference

```rust
// Session::run 需要 &mut self
let session = self.session.as_mut().ok_or(InferenceError::NotLoaded)?;

// 输入张量（ndarray）
let input_tensor = preprocess_image(img); // Array4<f32> [1,3,640,640]
let outputs = session.run(inputs!["input" => TensorRef::from_array_view(&input_tensor)?])?;

// 提取输出
let output = outputs.into_iter().next()
    .ok_or_else(|| InferenceError::Onnx(ort::Error::new("No outputs")))?;
let array: ndarray::ArrayViewD<'_, f32> = output.1.try_extract_array::<f32>()?;
```

### 3.3 模型输入输出规格 / Model I/O specs

| 模型 | 输入 | 输出 |
|------|------|------|
| SCRFD | `[1,3,640,640]` f32 (BGR) | scores / bboxes / kpss（多尺度） |
| ArcFace | `[1,3,112,112]` f32 (RGB) | `[1,512]` f32 embedding |
| inswapper | `[1,3,128,128]` f32 + embedding `[1,512]` | `[1,3,128,128]` f32 |
| GFPGAN | `[1,3,128,128]` f32 (RGB) | `[1,3,128,128]` f32 |

### 3.4 环境变量 / Environment variables

```rust
// main.rs 启动时设置
std::env::set_var("ORT_LOGGING_LEVEL", "Error");  // 抑制 ONNX 刷屏日志
std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");  // 减少 OpenMP 线程干扰
```

### English

- Use `ort` 2.0.0-rc.13 + `directml` feature (no GPU required).
- `Session::run` takes `&mut self`; keep `self.session.as_mut()`.
- Inputs via `TensorRef::from_array_view(&ndarray)`, outputs via `try_extract_array::<f32>()`.
- Set `ORT_LOGGING_LEVEL=Error` to silence ORT's verbose logs.

---

## 4. 踩过的坑 / Pitfalls

> 以下是我们实际遇到并修复的问题，均有明确症状与修复方案。
> Real bugs we hit, with symptoms and fixes.

### 4.1 `ort` 版本与 `ndarray` 版本必须匹配 / ndarray version lock

**坑**：`ort` 内部使用 ndarray，若 Cargo.toml 中 ndarray 版本与 ort 期望不一致，会出现类型不兼容或 API 变化导致的编译错误。

**修复**：锁定 `ndarray = "0.17"`（与 ort 2.0.0-rc.13 匹配），升级前必须检查 ort 兼容性。

### 4.2 `Session::run` 需要 `&mut self` / run needs &mut

**坑**：早期代码用 `as_ref()` 拿 `&Session`，编译报错 "cannot borrow as mutable"。

**修复**：统一用 `self.session.as_mut().ok_or(InferenceError::NotLoaded)?`。

### 4.3 仿射变换方向 / Affine direction (mask empty → output = original)

**症状**：换脸输出与输入完全一致（"没换成功"），因为 mask 全空。

**根因**：`paste_back` 用了**逆仿射** `transform_point(&inv, x, y)`，把原图像素映射到了模板外。

**修复**：用**正向仿射** `transform_point(affine, x, y)`。

### 4.4 SCRFD anchor Y 轴方向 / Anchor Y axis

**症状**：人脸对齐错误，landmarks 翻转。

**根因**：`cy = (feat_h - 1 - i) * stride`（自下而上），而 SCRFD 输出是自上而下。

**修复**：`cy = i * stride`。

### 4.5 仿射求解器 / Similarity transform solver

**症状**：人脸旋转错误。

**根因**：手写 2x2 SVD 返回零旋转（r01=r10=0）。

**修复**：改用线性最小二乘求解器，精确匹配 OpenCV `estimateAffinePartial2D`。

### 4.6 通道序 / Channel order (R/B swapped)

**症状**：换脸结果红蓝通道对调。

**根因**：inswapper 输出 RGB，但代码先转 BGR 又被 `paste_back` 当 RGB 处理。

**修复**：模型输出直接按 RGB 处理。

### 4.7 浮点转整型截断 / float→int truncation

**症状**：输出相对 Python 基线有 ~0.5 系统性偏移。

**根因**：cv2 用 `round()`，Rust `as u8` 是截断。

**修复**：所有浮点转像素用 `.round() as u8`。

### 4.8 JPEG 色度采样 / Chroma sampling

**症状**：输出文件比 Python 大 21%（87KB vs 197KB）。

**根因**：`image` crate 默认 4:4:4 色度采样，cv2 用 4:2:0。

**修复**：

```rust
use jpeg_encoder::{Encoder, SamplingFactor, ColorType};
let mut file = std::fs::File::create(&output_path)?;
let mut encoder = Encoder::new(&mut file, 95);
encoder.set_sampling_factor(SamplingFactor::R_4_2_0);
encoder.encode(rgb.as_raw(), w, h, ColorType::Rgb)?;
```

### 4.9 GFPGAN 增强对象 / GFPGAN must enhance the swapped result

**症状**：增强后结果被"拉回"原图。

**根因**：GFPGAN 输入用的是**原图**而非换脸结果。

**修复**：对 `paste_back` 后的**换脸结果**做 warp 再增强。

### 4.10 ARCFACE 模板坐标 / Warp template coordinates

**坑**：ArcFace warp 模板的 landmarks 坐标写错（0.575/0.573 应为 0.824/0.823），导致 embedding 对齐错误。

**修复**：使用标准 `ARCFACE_WARP_TEMPLATE` 值。

### English

Real bugs (all fixed, verified against a Python reference baseline):

1. **ndarray version lock** — keep `ndarray = "0.17"` matching `ort`.
2. **`Session::run` needs `&mut`** — use `as_mut()` not `as_ref()`.
3. **Affine direction** — forward transform, not inverse (mask empty bug).
4. **SCRFD anchor Y** — top-to-bottom (`cy = i * stride`).
5. **Similarity transform solver** — linear least-squares matching OpenCV, not hand-rolled SVD.
6. **Channel order** — treat inswapper output as RGB.
7. **`.round() as u8`** — match cv2 rounding, avoid systematic ~0.5 offset.
8. **JPEG 4:2:0** — use `jpeg-encoder` with `SamplingFactor::R_4_2_0`.
9. **GFPGAN input** — enhance the swapped result, not the original.
10. **Warp template** — use correct `ARCFACE_WARP_TEMPLATE` landmarks.

---

## 5. 经验总结 / Lessons Learned

### 5.1 以 Python 参考实现做逐阶段对比 / Stage-by-stage comparison

Rust 重写推理管线时，最容易出错的地方是"细节行为不一致"。我们的方法：

1. **逐阶段对比**：affine → warp crop → embedding cos → swapper 输出 → paste，每一阶段与 Python 参考输出做对比。
   - embedding 用 **cos 相似度**
   - 图像用 **mean abs diff**
2. **打印中间矩阵**：不只对比最终图，先对比 affine 矩阵本身。
3. **在 Python 里模拟 Rust**：用同样的 affine 公式、双线性 warp、box mask，隔离"Rust 偏离自身逻辑"还是"偏离 Python"。

### 5.2 验证基线 / Validation baseline

```
Rust vs Python 期望相似度：
  >87% 像素差值 ≤5
  <1%  像素差值 >20
  平均差 ~2.3
```

参考输出：`tests/fixtures/py_tinyface_baseline.jpg`（由真实 a.jpg/b.png 生成）。

### 5.3 调试"没换脸"的思路 / Debugging "no face swap"

如果输出看起来像原图：

1. **先确认输入路径**（最常见原因）：server 日志里 `Read input failed: 系统找不到指定的文件` 说明文件名写错。测试图片固定为 `a.jpg`（源）和 `b.png`（目标），不要用 `a.png`/`b.jpg`。
2. 检查 `paste_back` 仿射方向（正向）。
3. 检查 SCRFD anchor Y 轴（自上而下）。
4. 检查 `estimate_similarity_transform` 旋转（线性求解器）。

### 5.4 日志与可观测性 / Logging

```rust
// 只记录 WARN/ERROR，抑制 ONNX 刷屏
tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();
std::env::set_var("ORT_LOGGING_LEVEL", "Error");
```

- server 输出被 Tauri 重定向到 `srv_out.log` / `srv_err.log`。
- 排查问题先看这两个文件，靠近 `Reading input image:` 行附近找真实错误。

### English

- Compare **stage-by-stage** against a Python reference (cos sim for embeddings, mean-abs-diff for images).
- Print the **affine matrix**, not just the final image.
- **Simulate Rust in Python** to isolate deviations.
- Validate against the baseline: **>87% pixels diff ≤5, <1% >20, mean ~2.3**.
- "No face swap" ≈ wrong input path (check `srv_out.log` for the read error).

---

## 6. 常见问题 FAQ

| 问题 | 答案 |
|------|------|
| 需要 GPU 吗？ | 不需要。`ort` 使用 DirectML 后端，CPU 即可。 |
| 模型从哪来？ | 见 [模型清单与下载指南](./models.md)（含下载地址与存放方式）。 |
| 如何增加并发？ | `server.exe --workers N`。默认自动计算。 |
| 输出文件在哪？ | 输入图同目录，`{basename}_output.jpg`，重名加时间戳。 |
| ort 日志刷屏？ | `ORT_LOGGING_LEVEL=Error` 已内置抑制。 |
| 换脸没生效？ | 见上文 5.3 调试思路，先查日志输入路径。 |

### English FAQ

- **Need a GPU?** No — `ort` uses DirectML, runs on CPU.
- **Where are models?** See [architecture](./architecture.md) §5.2.
- **More concurrency?** `server.exe --workers N`.
- **Where is output?** Next to the input image, `{basename}_output.jpg`, timestamped on conflict.
- **No swap happened?** See §5.3 — almost always the wrong input path.