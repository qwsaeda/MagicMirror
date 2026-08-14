# 模型清单与下载指南 (Models Guide)

> 本文档介绍 MagicMirror 用到的全部 AI 模型：功能、输入输出规格、下载地址、存放目录与使用方式。
>
> This document covers all AI models used by MagicMirror: purpose, I/O specs, download links, location, and usage.

---

## 目录 (Table of Contents)

- [1. 模型总览 / Model Overview](#1-模型总览--model-overview)
- [2. 模型详细说明 / Per-Model Details](#2-模型详细说明--per-model-details)
- [3. 下载与存放 / Download & Location](#3-下载与存放--download--location)
- [4. 使用方式 / How It's Used](#4-使用方式--how-its-used)
- [5. 常见问题 / FAQ](#5-常见问题--faq)

---

## 1. 模型总览 / Model Overview

| 模型 / Model | 大小 / Size | 作用 / Purpose | 必需 / Required |
|--------------|-------------|----------------|-----------------|
| `scrfd_2.5g.onnx` | 3.3 MB | 人脸检测（SCRFD） | ✅ 是 |
| `arcface_w600k_r50.onnx` | 166 MB | 人脸身份嵌入（ArcFace） | ✅ 是 |
| `inswapper_128_fp16.onnx` | 265 MB | 人脸交换（inswapper, fp16） | ✅ 是 |
| `inswapper_weight.bin` | 1 MB | 交换权重矩阵 | ✅ 是 |
| `gfpgan_1.4.onnx` | 324 MB | 质量增强（GFPGAN） | ⭕ 可选（推荐） |
| **合计 / Total** | **~760 MB** | | |

### 推理管线中的角色 / Role in the pipeline

```
scrfd_2.5g ──▶ 检测人脸位置 + 5 点关键点
arcface_w600k_r50 ──▶ 提取目标身份 embedding (512 维)
inswapper_weight.bin ──▶ 变换 embedding（风格注入）
inswapper_128_fp16 ──▶ 融合两张人脸
gfpgan_1.4 ──▶ 提升换脸结果清晰度（可选）
```

---

## 2. 模型详细说明 / Per-Model Details

### 2.1 `scrfd_2.5g.onnx` — 人脸检测 / Face Detection

**功能 / Function**：检测图片中的所有人脸，输出人脸边界框与 5 点关键点（双眼、鼻尖、左右嘴角），用于后续对齐与裁剪。

**输入 / Input**：`[1, 3, 640, 640]` f32，BGR 通道，均值 `[127.5,127.5,127.5]`、缩放 `1/128`（与 Python `blobFromImage` 一致）。

**输出 / Output**：多尺度 scores / bboxes / kpss。

**说明 / Notes**：
- 推理管线优先取**面积最大**的人脸作为源脸。
- 关键点 Y 轴**自上而下**（`cy = i * stride`），反了会导致对齐错误。

---

### 2.2 `arcface_w600k_r50.onnx` — 人脸嵌入 / Face Embedding

**功能 / Function**：将裁剪对齐后的 112×112 人脸映射为 **512 维** 身份向量（embedding），用于表征"这是谁"。

**输入 / Input**：`[1, 3, 112, 112]` f32，RGB 通道，归一化 `[-1, 1]`。

**输出 / Output**：`[1, 512]` f32 归一化 embedding。

**说明 / Notes**：
- 输入人脸必须先用 `ARCFACE_WARP_TEMPLATE` 做仿射对齐，模板坐标错误会导致身份特征提取不准。
- 本模型固定用于提取**目标图（身份来源）**的 embedding。

---

### 2.3 `inswapper_128_fp16.onnx` — 人脸交换 / Face Swapping

**功能 / Function**：以目标身份 embedding 驱动，把源脸换成目标脸，输出 128×128 换脸结果。

**输入 / Input**：
- `source`: `[1, 3, 128, 128]` f32（源脸 crop）
- `target`: `[1, 512]` f32（目标身份 embedding）

**输出 / Output**：`[1, 3, 128, 128]` f32（RGB 通道）

**说明 / Notes**：
- 本项目使用 **fp16** 版本（265MB）；原版还有 fp32 版本（554MB，Python 基线用）。
- 输出**直接按 RGB 处理**，不要再转 BGR（历史上因此出过红蓝对调 bug）。

---

### 2.4 `inswapper_weight.bin` — 交换权重矩阵 / Swap Weight Matrix

**功能 / Function**：从 inswapper ONNX 模型导出的最后一层权重矩阵（`[512, 512]` f32，小端序），用于在推理前变换 source embedding，实现"风格注入"。

**输入 / Input**：无（推理前读取到内存）。

**处理 / Processing**：

```rust
// 从 ONNX 模型提取（一次性导出，存为 .bin）
// 运行期读取 512×512 个 f32（小端）到内存
let data = std::fs::read(weight_path)?;             // 大小必须 = 512*512*4 = 1,048,576 字节
let mut weight = vec![0.0f32; 512 * 512];
for (i, chunk) in data.chunks_exact(4).enumerate() {
    weight[i] = f32::from_le_bytes(chunk.try_into().unwrap());
}
// 使用：transformed_emb = emb.dot(&weight)
```

**说明 / Notes**：
- 缺失时交换会报错（`Weight matrix not loaded`）。
- 若模型是 fp32 版本，权重提取方式相同。

---

### 2.5 `gfpgan_1.4.onnx` — 质量增强 / Quality Enhancement (可选)

**功能 / Function**：对换脸后的 128×128 人脸做清晰度/细节增强，减少模糊感。

**输入 / Input**：`[1, 3, 128, 128]` f32（RGB）。

**输出 / Output**：`[1, 3, 128, 128]` f32（RGB）。

**说明 / Notes**：
- 增强对象是**换脸结果**，不是原图（否则增强会把结果拉回原图）。
- 与换脸结果按 `0.25 * swap + 0.75 * enhanced` 混合，提升清晰度（Laplacian 方差 +54%）。
- 文件缺失时跳过增强，不影响主流程，但画质略降。

---

## 3. 下载与存放 / Download & Location

### 3.1 官方下载地址 / Official download URL

所有模型打包发布在 TinyFace 项目 Release：

```
https://github.com/idootop/TinyFace/releases/download/models-1.0.0
```

逐个下载：

```bash
BASE_URL="https://github.com/idootop/TinyFace/releases/download/models-1.0.0"
curl -O -L "${BASE_URL}/scrfd_2.5g.onnx"
curl -O -L "${BASE_URL}/arcface_w600k_r50.onnx"
curl -O -L "${BASE_URL}/inswapper_128_fp16.onnx"
curl -O -L "${BASE_URL}/gfpgan_1.4.onnx"
```

> `inswapper_weight.bin` 需从 inswapper fp32 模型手动导出（见 4.2），或直接使用本项目仓库 `out/server/models/` 中已导出的文件。

### 3.2 存放目录 / Where to put them

`server.exe` 启动时按以下顺序查找 `models/` 目录：

```
1. {当前工作目录}/models      ← 优先（MagicMirror 派生 server 时 CWD = exe 目录）
2. {server.exe 所在目录}/models
3. {HOME}/MagicMirror/models
```

**便携版结构 / Portable layout:**

```
MagicMirror-Portable/
├── MagicMirror.exe
├── server.exe
└── models/                    ← 全部模型放这里
    ├── scrfd_2.5g.onnx
    ├── arcface_w600k_r50.onnx
    ├── inswapper_128_fp16.onnx
    ├── inswapper_weight.bin
    └── gfpgan_1.4.onnx
```

---

## 4. 使用方式 / How It's Used

### 4.1 加载顺序（Rust）/ Load order (Rust)

```rust
// src-server/src/inference/mod.rs
pub fn load_models(&mut self, models_dir: &Path) -> InferenceResult<()> {
    self.detector.load(models_dir.join("scrfd_2.5g.onnx"))?;
    self.embedder.load(models_dir.join("arcface_w600k_r50.onnx"))?;
    self.swapper.load(models_dir.join("inswapper_128_fp16.onnx"))?;

    let weight_path = models_dir.join("inswapper_weight.bin");
    if weight_path.exists() {
        self.swapper.load_weight(&weight_path)?;
    }

    let enhancer_path = models_dir.join("gfpgan_1.4.onnx");
    if enhancer_path.exists() {
        self.enhancer = Some(enhancer::Enhancer::new());
        self.enhancer.as_mut().unwrap().load(&enhancer_path)?;
    }
    Ok(())
}
```

- `scrfd` / `arcface` / `inswapper` 为**必需**模型，缺失直接报错。
- `gfpgan` 为**可选**，缺失则跳过增强。
- `inswapper_weight.bin` 为必需，缺失时交换步骤报错。

### 4.2 从 ONNX 导出 weight 矩阵 / Exporting weight from ONNX

`inswapper_weight.bin` 是 inswapper 模型最后一个 initializer（`[512,512]` f32）。可用 Python 导出：

```python
import onnx
m = onnx.load("inswapper_128.onnx")  # 用 fp32 版本
init = m.graph.initializer[-1]
weight = init.raw_data  # 已是小端 f32，直接写文件
with open("inswapper_weight.bin", "wb") as f:
    f.write(weight)
```

> 注意：确保导出的是**最后**一个 initializer（正是本项目所需矩阵）；fp16 模型不宜用于导出权重。

### 4.3 输出路径规则 / Output Path Rule

**输出文件保存到 targetFace（第二张图片/身份来源图）所在目录**，文件名 `{basename}_output.jpg`。

```rust
// worker.rs
let target_path = std::path::Path::new(&task.target_face_path);
let output_dir = target_path.parent().unwrap_or_else(|| Path::new("."));
let base_name = target_path.file_stem().unwrap_or("output");
let output_path = output_dir.join(format!("{base_name}_output.jpg"));
```

- 若文件已存在，追加毫秒时间戳 `{basename}_output_{timestamp}.jpg` 避免覆盖
- `id` 字段仅用于任务取消，不再影响输出路径

### 4.4 验证模型完整性 / Verify models

模型缺失的典型日志：

```
srv_err.log:
ERROR src-server\src\worker.rs:...: Worker pool error: ... No such file or directory
```

或在 `srv_out.log` 找不到 `Models loaded successfully`。此时检查 `models/` 目录是否包含上述 5 个文件且大小正确。

---

## 5. 常见问题 / FAQ

| 问题 | 答案 |
|------|------|
| 模型从哪里下载？ | TinyFace Release：`github.com/idootop/TinyFace/releases/download/models-1.0.0` |
| 没有 GPU 能用吗？ | 能。ONNX Runtime 用 DirectML，CPU 即可运行。 |
| fp16 和 fp32 的 inswapper 区别？ | fp16（265MB）Rust 用；fp32（554MB）Python 基线用，效果基本一致，fp16 更快。 |
| `inswapper_weight.bin` 哪来的？ | 从 inswapper ONNX 最后一个 initializer 导出（512×512 f32 小端）。 |
| 缺少 GFPGAN 会怎样？ | 跳过增强，换脸仍可工作，画质略降。 |
| 模型放哪？ | `server.exe` CWD 或 exe 目录下的 `models/` 文件夹。 |
| 模型加载多久？ | 合计约 30s（GFPGAN 占 15s），期间 `/status` 返回 `starting`。 |

### English FAQ

- **Where to download?** TinyFace Release `models-1.0.0` (link above).
- **No GPU?** Fine — DirectML backend, CPU only.
- **fp16 vs fp32 inswapper?** Rust uses fp16 (265MB); Python baseline uses fp32 (554MB). Same quality, fp16 faster.
- **Where's `inswapper_weight.bin` from?** Exported from the last ONNX initializer (512×512 f32 LE).
- **No GFPGAN?** Enhancement skipped; swap still works.
- **Where to place?** `models/` next to `server.exe` or its CWD.
- **Load time?** ~30s total (GFPGAN ~15s); `/status` shows `starting` meanwhile.
