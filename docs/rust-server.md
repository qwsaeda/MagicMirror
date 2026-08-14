# Rust 推理服务实现指南 (Rust Server Implementation Guide)

> 本文档是 MagicMirror Rust 推理服务的总览索引，详细技术内容见子文档。
>
> This is the overview index for MagicMirror's Rust inference server. See sub-documents for detailed technical content.

---

## 目录 / Table of Contents

| 文档 / Doc | 内容 / Content |
|-----------|---------------|
| [📐 系统架构](./architecture.md) | 整体架构、进程模型、Worker 系统、便携版打包 |
| [🦀 Rust Server 概览](#rust-server-overview) | 为什么用 Rust、目录结构、推理管线 |
| [🔄 子进程管理](./rust-guide/child-process.md) | spawn/kill 逻辑、残留清理、退出兜底 |
| [🎮 GPU 支持](./rust-guide/gpu-support.md) | DirectML/CUDA 配置、自动检测、国产卡适配 |
| [⚠️ ort 踩坑记录](./rust-guide/ort-pitfalls.md) | 已修复 bug、调试经验、验证基线 |
| [🐛 调试方法](./rust-guide/debugging.md) | 逐阶段对比、性能基准、快速排错 |
| [🧠 模型清单](./models.md) | 模型功能、输入输出规格、下载地址 |

---

## Rust Server 概览 / Rust Server Overview

### 为什么用 Rust / Why Rust

| 维度 | Rust | Python（原版） |
|------|------|---------------|
| **启动速度** | 毫秒级 | 秒级 |
| **内存占用** | ~300MB | 更高 |
| **部署形态** | 单二进制 | 需打包运行时 |
| **包体积** | ~24MB | 数百 MB |
| **GPU 支持** | DirectML/CUDA | DirectML/CUDA |

**核心优势**：
- 零环境依赖：单 `server.exe` 即可运行
- 进程模型简单：Tauri 直接托管子进程生命周期
- 编译期安全：类型系统防止维度/通道序错误

### 目录结构 / Directory Structure

```
src-server/
├── Cargo.toml           # 依赖清单
└── src/
    ├── main.rs          # HTTP 入口 + GPU 检测
    ├── worker.rs        # Worker 池 + 自动并发计算
    └── inference/       # 推理管线
        ├── mod.rs       # TinyFace 编排器
        ├── detector.rs  # SCRFD 人脸检测
        ├── embedder.rs  # ArcFace 嵌入
        ├── swapper.rs   # inswapper 交换
        ├── enhancer.rs  # GFPGAN 增强
        ├── warp.rs      # 仿射变换
        └── image_utils.rs
```

### 推理管线 / Inference Pipeline

```
输入图 ──▶ SCRFD 检测 ──▶ landmarks
                      │
目标图 ──▶ SCRFD 检测 ──▶ landmarks
                      │
                      ▼
              ArcFace embedding (512维)
                      │
                      ▼
              inswapper 交换 (128x128)
                      │
                      ▼
              GFPGAN 增强（可选）
                      │
                      ▼
              paste_back 贴回原图
                      │
                      ▼
              输出 JPG (4:2:0, Q95)
```

### API 端点 / API Endpoints

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/status` | GET | 运行状态 + worker 数 |
| `/prepare` | POST | 等待模型加载（180s 超时） |
| `/task` | POST | 提交换脸任务（同步等待，120s 超时） |
| `/task/{id}` | DELETE | 取消任务（返回 405） |

### 文件写入架构 / File Write Architecture

**关键设计决策：由 server.exe 写入文件，而非前端 MagicMirror**

```typescript
// 前端只发请求、收路径
const result = await Server.createTask({
  id: taskId,
  inputImage,
  targetFace,
});
setOutput(result);  // 显示，不写文件
```

```rust
// server 直接写入输出文件
let mut file = std::fs::File::create(&output_path)?;
let mut encoder = jpeg_encoder::Encoder::new(&mut file, 95);
encoder.encode(rgb.as_raw(), width, height, ColorType::Rgb)?;
```

**原因：**
1. Tauri 安全模型限制前端直接访问文件系统
2. server 是推理服务，天然持有文件 I/O 权限
3. 架构清晰：前端只管 UI，server 只管推理 + 文件操作

**输出路径规则（已修正）：**
- 输出保存到 **`targetFace` 所在目录**（第二张图片/身份来源图）
- 文件名 `{basename}_output.jpg`，重名加时间戳
- `id` 字段仅用于任务取消，不影响输出路径

---

## 快速开始 / Quick Start

### 本地构建 / Local Build

```bash
# 构建 Rust server
cd src-server
cargo build --release

# 构建 Tauri 桌面端
cd ..
pnpm tauri build
```

### 运行 / Run

```bash
# 方式 1: 双击 MagicMirror.exe（自动派生 server）
.\MagicMirror.exe

# 方式 2: 命令行启动 server
.\server.exe --workers auto

# 方式 3: 批量换脸
.\scripts\batch-swap.bat
```

---

## 相关链接 / Related Links

- [架构设计](./architecture.md) — 进程模型、Worker 系统
- [子进程管理](./rust-guide/child-process.md) — spawn/kill/清理逻辑
- [GPU 支持](./rust-guide/gpu-support.md) — DirectML/CUDA 配置
- [ort 踩坑](./rust-guide/ort-pitfalls.md) — 已修复 bug
- [调试方法](./rust-guide/debugging.md) — 排查技巧
- [模型清单](./models.md) — 下载与使用
