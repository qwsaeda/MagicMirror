# MagicMirror

**Instant AI Face Swap — One click to a brand new you!** · 一键 AI 换脸，发现更美的自己 ✨

[![GitHub release](https://img.shields.io/github/v/release/idootop/MagicMirror.svg)](https://github.com/idootop/MagicMirror/releases) [![Build APP](https://github.com/idootop/MagicMirror/actions/workflows/build-app.yaml/badge.svg)](https://github.com/idootop/MagicMirror/actions/workflows/build-app.yaml) [![Build Server](https://github.com/idootop/MagicMirror/actions/workflows/build-server.yaml/badge.svg)](https://github.com/idootop/MagicMirror/actions/workflows/build-server.yaml)

![](./docs/assets/banner.jpg)

---

## 简介 / Introduction

MagicMirror 是一个**完全离线运行**的 AI 换脸桌面应用。把照片拖进镜子，即可一键换脸、试发型、试穿搭。所有推理都在本地完成，图片不会上传到任何服务器。

MagicMirror is a **fully offline** AI face-swap desktop app. Drag a photo into the mirror and instantly try on new faces, hairstyles, and outfits. All inference happens locally — your images never leave your device.

本仓库是原版 MagicMirror 的**深度改造 fork**：推理层由 Python 整体重写为**纯 Rust**，并新增了并发 Worker 系统、便携绿色版打包与完善的子进程生命周期管理。

This repo is a **deeply reworked fork**: the inference layer is fully rewritten in **pure Rust**, with a concurrent worker pool, portable "green" packaging, and robust child-process lifecycle management.

---

## 功能亮点 / Features

| 中 / CN | 英 / EN |
|---------|---------|
| 🪞 一键换脸 | **Where AI Meets Your Beauty** — one click to a brand new you |
| 🎯 超低门槛，无需 GPU | **Hardware Friendly** — runs on ordinary computers, no GPU needed |
| 🔒 完全离线，隐私安全 | **Privacy First** — 100% offline, images never uploaded |
| 🚀 极致精简，绿色便携版 | **Ultra-Lightweight** — single portable zip, unzip & run |
| 🧩 并发 Worker，自动调优 | **Concurrent Worker Pool** — auto-scaled by CPU & memory |

---

## 技术栈 / Tech Stack

本 fork 用**纯 Rust 重写了原版 Python 推理服务** / This fork replaces the original Python inference server with pure Rust:

- **桌面端 / Desktop**: [Tauri 2](https://tauri.app/) + React 18 + TypeScript + Vite
- **推理服务 / Inference**: [Axum](https://github.com/tokio-rs/axum) HTTP + [ort](https://github.com/pykeio/ort) (ONNX Runtime, DirectML — **无需 GPU / no GPU required**)
- **模型 / Models**: SCRFD (检测) + ArcFace (嵌入) + inswapper (交换) + GFPGAN (增强)
- **并发 / Concurrency**: Worker 池，模型内存共享，数量自动计算
- **进程模型 / Process**: 桌面端自动派生/托管 `server.exe` 子进程，无控制台窗口，退出自动清理

**Rust 实现优势 / Why Rust:** 单二进制零环境依赖、毫秒级启动、内存占用更低、编译期保证图像管线正确性。详见 [Rust 实现指南](./docs/rust-server.md)。

---

## 文档 / Documentation

> 所有文档均提供中英双语说明 / All docs are bilingual.

| 文档 / Doc | 说明 / Description |
|-----------|--------------------|
| [📐 架构设计 / Architecture](./docs/architecture.md) | 系统总览、进程模型、Worker 系统、REST API、便携版打包 |
| [🦀 Rust 实现指南 / Rust Server Guide](./docs/rust-server.md) | **总览索引**：架构、API、快速开始 |
| [📁 子进程管理 / Child Process](./docs/rust-guide/child-process.md) | spawn/kill 逻辑、残留清理、退出兜底 |
| [🎮 GPU 支持 / GPU Support](./docs/rust-guide/gpu-support.md) | DirectML/CUDA 配置、自动检测、国产卡适配 |
| [⚠️ ort 踩坑 / ort Pitfalls](./docs/rust-guide/ort-pitfalls.md) | 已修复 bug、调试经验、验证基线 |
| [🐛 调试方法 / Debugging](./docs/rust-guide/debugging.md) | 逐阶段对比、性能基准、快速排错 |
| [🧠 模型清单 / Models Guide](./docs/models.md) | 模型功能、输入输出规格、**下载地址**、存放目录 |
| [📦 安装 / Install (EN)](./docs/en/install.md) | 安装与系统要求 / Installation & system requirements |
| [🎮 使用 / Usage (EN)](./docs/en/usage.md) | 使用教程 / Usage guide |
| [❓ 常见问题 / FAQ (EN)](./docs/en/faq.md) | 常见问题 / Frequently asked questions |
| [📖 中文总览 / CN Readme](./docs/cn/readme.md) | 中文版项目介绍、安装、使用与 FAQ |

**开发者必读 / Developer essentials:**

- [Rust 服务端实现与 ort 经验](./docs/rust-server.md) — 如果你要改造推理管线，先读这一篇。
- [架构与进程模型](./docs/architecture.md) — 理解桌面端如何托管 server 子进程。

---

## 快速开始 / Get Started

> [👉 中文教程和下载地址请戳这里](./docs/cn/readme.md)

To get started with MagicMirror:

1. Follow the [Installation Guide](./docs/en/install.md)
2. Check out the [Usage Guide](./docs/en/usage.md)
3. See the [FAQ](./docs/en/faq.md) for common issues

If you have any questions, please [submit an issue](https://github.com/idootop/MagicMirror/issues).

> Note: MagicMirror only supports macOS 13 (Ventura) and Windows 10 and above.
>
> 注：仅支持 macOS 13 (Ventura) 与 Windows 10 及以上系统。

---

## 动机 / Motivation

![391785246-b3b52898-4d43-40db-8fbe-acbc00d78eec](https://github.com/user-attachments/assets/64ba0436-d7c2-4e81-bc4a-9ec00b5b7d7a)

Ever found yourself endlessly scrolling through hairstyles and outfits, wondering "How would this look on me?" As someone who loves exploring different styles but hates the hassle, I dreamed of an app that could instantly show me wearing any look I fancy.

While AI technology has advanced tremendously, most AI face applications either require complex setup, demand high-end GPU hardware, or rely on cloud processing that compromises privacy.

**The ideal solution should be as simple as taking a selfie** - just drag, drop, and transform. No technical expertise needed, no expensive hardware required, and no privacy concerns.

So, why not build one myself?

And that’s how MagicMirror came to life ✨

你是否也曾在众多发型和穿搭间纠结：这套造型放在我身上会是什么样？如果有一款应用，能把喜欢的发型或心动的穿搭直接"穿"到自己身上预览，那该多好。

现在的 AI 技术已经很成熟，但大多数换脸应用要么需要配置复杂参数和高性能 GPU，要么必须把图片上传到服务器，存在隐私风险。**理想的解决方案应该像自拍一样简单**——拖进去，就完成。无需专业知识，无需昂贵设备，无需担心隐私。

所以，为什么不自己做一个呢？

于是便有了 MagicMirror ✨

Enjoy! ;)

---

## 鸣谢 / Acknowledgments

MagicMirror builds upon several outstanding open-source projects:

- [TinyFace](https://github.com/idootop/TinyFace): The minimalist face swapping tool that just works.
- [FaceFusion](https://github.com/facefusion/facefusion): Industry leading face manipulation platform
- [InsightFace](https://github.com/deepinsight/insightface): State-of-the-art 2D and 3D Face Analysis Project
- [Nuitka](https://github.com/Nuitka/Nuitka): Nuitka is a Python compiler written in Python.
- [Tauri](https://github.com/tauri-apps/tauri): Build smaller, faster, and more secure desktop and mobile applications with a web frontend.

---

## 免责声明 / Disclaimer

MagicMirror is designed for personal entertainment and creative purposes only. Commercial use is prohibited. / MagicMirror 仅限个人娱乐与创意用途，严禁用于商业用途。

- **Ethical Usage / 道德使用**: This software must not be used for activities including, but not limited to: a) impersonating others with malicious intent, b) spreading misinformation, c) violating personal privacy or dignity, d) creating explicit or inappropriate content.
- **Content Rights / 内容版权**: Users are responsible for: a) obtaining necessary permissions for source images, b) respecting copyrights and intellectual property, c) complying with local laws and regulations on AI-generated content.
- **Limitation of Liability / 免责声明**: The software and its developers are not liable for any user-generated content. Users assume full responsibility for the use of the generated content and any consequences that may arise from its use.

By using MagicMirror, you agree to these terms and commit to using the software responsibly. / 使用 MagicMirror 即表示您已阅读并同意上述条款，并承诺负责任地使用本软件。

---

## License

MIT License © 2024-PRESENT [Del Wang](https://del.wang)
