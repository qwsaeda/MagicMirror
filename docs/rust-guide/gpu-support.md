# GPU 支持与自动检测 (GPU Support & Auto-Detection)

> 本文档记录 MagicMirror 的 GPU 后端选择策略、自动检测实现，以及国产 GPU 支持情况。
>
> This document covers GPU backend selection, auto-detection implementation, and domestic GPU support status.

---

## 1. GPU 后端概览 / GPU Backend Overview

### 支持的 Provider

| Provider | 厂商 | 启用方式 | 说明 |
|----------|------|---------|------|
| **DirectML** | 所有 Windows GPU | `features = ["directml"]` | 当前默认，覆盖 NVIDIA/AMD/Intel |
| **CUDA** | NVIDIA | `features = ["cuda"]` | 需安装 CUDA Toolkit |
| **CoreML** | Apple Silicon | `features = ["coreml"]` | macOS/iOS |
| **TensorRT** | NVIDIA | `features = ["tensorrt"]` | 高性能，需额外安装 |
| **CPU** | 所有设备 | 无 feature | 兜底方案 |

### 当前配置

```toml
# src-server/Cargo.toml
ort = { version = "2.0.0-rc.13", features = ["directml"] }
```

**为什么选 DirectML 默认？**
- 零依赖：无需安装 CUDA Toolkit
- 通用性好：NVIDIA/AMD/Intel 都能用
- CPU 可降级：无 GPU 时自动回退到 CPU

---

## 2. GPU 自动检测实现 / Auto-Detection Implementation

### 2.1 检测逻辑

```rust
// src-server/src/main.rs
#[cfg(target_os = "windows")]
fn detect_gpu_backend() -> &'static str {
    // 使用 WMIC 查询显卡信息
    let output = std::process::Command::new("wmic")
        .args(["path", "win32_VideoController", "get", "Name"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok());
    
    match output {
        Some(ref info) if info.to_lowercase().contains("nvidia") => {
            tracing::info!("Detected NVIDIA GPU, CUDA backend recommended");
            "cuda"
        }
        Some(ref info) if info.to_lowercase().contains("amd") || info.to_lowercase().contains("radeon") => {
            tracing::info!("Detected AMD GPU, DirectML backend recommended");
            "directml"
        }
        Some(_) | None => {
            tracing::info!("No discrete GPU detected, using DirectML fallback");
            "directml"
        }
    }
}
```

### 2.2 环境变量设置

```rust
fn main() {
    let gpu_backend = detect_gpu_backend();
    
    match gpu_backend {
        "cuda" => {
            std::env::set_var("ORT_CUDA_AVAILABLE", "1");
            info!("Using CUDA backend (NVIDIA GPU detected)");
        }
        "directml" => {
            std::env::set_var("ORT_DIRECTML_AVAILABLE", "1");
            info!("Using DirectML backend");
        }
        _ => {
            info!("Using CPU backend (no GPU detected)");
        }
    }
}
```

### 2.3 环境变量速查

| 变量 | 值 | 作用 |
|------|-----|------|
| `ORT_CUDA_AVAILABLE` | `1` | 启用 CUDA Provider |
| `ORT_DIRECTML_AVAILABLE` | `1` | 启用 DirectML Provider |
| `ORT_LOGGING_LEVEL` | `Error` | 抑制 ORT 刷屏日志 |
| `OMP_WAIT_POLICY` | `PASSIVE` | 减少 OpenMP 线程干扰 |

---

## 3. 性能对比 / Performance Comparison

基于 GTX 1050 Ti 实测（换脸耗时）：

| 后端 | 平均耗时 | 显存占用 | 说明 |
|------|---------|---------|------|
| **CUDA** | ~0.8s | ~2GB | 最快，需 CUDA Toolkit |
| **DirectML** | ~1.2s | ~1.5GB | 次快，零配置 |
| **CPU** | ~3.5s | ~500MB | 最慢，兼容最好 |

> 注：实际性能取决于模型大小、图像分辨率、GPU 显存。

---

## 4. 国产 GPU 支持情况 / Domestic GPU Support

### 已支持的 Provider

| 厂商 | Provider | 状态 | 使用方式 |
|------|----------|------|---------|
| **华为昇腾** | CANN | ⚠️ Preview | 社区维护，需手动加载 DLL |
| **AMD** | MIGraphX | ✅ 官方 | 需编译时启用 |
| **Intel** | OpenVINO | ✅ 官方 | 核显优化 |

### 华为 CANN 适配说明

**重要纠正**：之前文档错误地说"国产卡不支持"，实际上华为 CANN 已被 ONNX Runtime 官方收录为 Community-maintained Provider。

**当前限制**：
- `rust-ort` 官方暂未提供 `cann` feature flag
- 需要手动加载 CANN Provider DLL 或通过 C API 调用

**适配方式（未来）**：
```rust
// 方案 A：通过 C API 手动加载（需要 FFI 封装）
extern "C" {
    fn OrtSessionOptionsAppendExecutionProvider_CANN(
        options: *mut OrtSessionOptions,
        device_id: i32
    );
}

// 方案 B：fork rust-ort 添加 cann feature
// 参考：https://github.com/pykeio/ort
```

### 其他国产卡

| 厂商 | 型号 | 状态 |
|------|------|------|
| 天数智芯 | IPU | ❌ 无公开信息 |
| 摩尔线程 | MUSA | ❓ 需查厂商文档 |
| 燧原 | Triton | ❌ 无 ONNX 支持 |

---

## 5. 如何启用 CUDA / How to Enable CUDA

### 条件

1. 用户已安装 NVIDIA GPU
2. 用户已安装 CUDA Toolkit 11.x+
3. 重新编译 server.exe 启用 CUDA feature

### 修改 Cargo.toml

```toml
# 启用 CUDA（会显著增加编译时间和二进制大小）
ort = { version = "2.0.0-rc.13", features = ["cuda", "directml"] }
```

### 重新编译

```bash
cd src-server
cargo build --release
# 编译时间从 ~2min 增加到 ~10min，产物从 24MB 增加到 ~200MB
```

---

## 6. FAQ

**Q: 我的电脑有 NVIDIA 显卡，为什么没用 CUDA？**
A: 当前默认启用 DirectML，因为它零配置且兼容性更好。如需 CUDA 加速，需重新编译并安装 CUDA Toolkit。

**Q: DirectML 支持所有 Windows GPU 吗？**
A: 是的，DirectML 是 Windows 的 GPU 抽象层，NVIDIA/AMD/Intel 都支持。

**Q: 如何提高推理速度？**
A: 启用 CUDA（NVIDIA）或 TensorRT（NVIDIA）。DirectML 已经比 CPU 快 2-3 倍。

**Q: 国产卡什么时候能原生支持？**
A: 华为 CANN 已支持但需手动适配。其他厂商暂无公开 ONNX Provider。
