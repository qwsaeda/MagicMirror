# MagicMirror

Instant AI Face Swap — One click to a brand new you!

[![GitHub release](https://img.shields.io/github/v/release/idootop/MagicMirror.svg)](https://github.com/idootop/MagicMirror/releases) [![Build APP](https://github.com/idootop/MagicMirror/actions/workflows/build-app.yaml/badge.svg)](https://github.com/idootop/MagicMirror/actions/workflows/build-app.yaml) [![Build Server](https://github.com/idootop/MagicMirror/actions/workflows/build-server.yaml/badge.svg)](https://github.com/idootop/MagicMirror/actions/workflows/build-server.yaml)

![](./docs/assets/banner.jpg)

## Features

- **Where AI Meets Your Beauty**: Effortlessly try on new faces, hairstyles, and outfits
- **Stupid Simple**: Drag and drop photos to instantly change faces - no complex settings required
- **Hardware Friendly**: Runs smoothly on standard computers, no dedicated GPU hardware required
- **Privacy First**: Completely offline processing - your images never leave your device
- **Ultra-Lightweight**: Tiny footprint with <10MB installer and <1GB model files
- **Portable Green Version**: A single zip you can unzip and run directly - no installation needed

## Tech Stack

This fork replaces the original Python inference server with a **pure Rust implementation**:

- **Desktop App**: [Tauri 2](https://tauri.app/) + React 18 + TypeScript + Vite
- **Inference Server**: [Axum](https://github.com/tokio-rs/axum) HTTP server + [ort](https://github.com/pykeio/ort) (ONNX Runtime / DirectML)
- **Models**: SCRFD (detection) + ArcFace (embedding) + inswapper (swap) + GFPGAN (enhancement)
- **Concurrency**: Worker pool with auto-scaled worker count, shared model memory
- **Process Model**: MagicMirror spawns/manages `server.exe` as a background child process (no console window), auto-reclaims orphaned servers, and cleans up on exit

See [Architecture Design](./docs/architecture.md) for details on the process model, worker system, REST API, and packaging.

## Get Started

> [👉 中文教程和下载地址请戳这里](./docs/cn/readme.md)

To get started with MagicMirror:

1. Follow the [Installation Guide](./docs/en/install.md)
2. Check out the [Usage Guide](./docs/en/usage.md)
3. See the [FAQ](./docs/en/faq.md) for common issues

If you have any questions, please [submit an issue](https://github.com/idootop/MagicMirror/issues).

> Note: MagicMirror only supports macOS 13 (Ventura) and Windows 10 and above.

## Motivation

![391785246-b3b52898-4d43-40db-8fbe-acbc00d78eec](https://github.com/user-attachments/assets/64ba0436-d7c2-4e81-bc4a-9ec00b5b7d7a)

Ever found yourself endlessly scrolling through hairstyles and outfits, wondering "How would this look on me?" As someone who loves exploring different styles but hates the hassle, I dreamed of an app that could instantly show me wearing any look I fancy.

While AI technology has advanced tremendously, most AI face applications either require complex setup, demand high-end GPU hardware, or rely on cloud processing that compromises privacy.

**The ideal solution should be as simple as taking a selfie** - just drag, drop, and transform. No technical expertise needed, no expensive hardware required, and no privacy concerns.

So, why not build one myself?

And that’s how MagicMirror came to life ✨

Enjoy! ;)

## Acknowledgments

MagicMirror builds upon several outstanding open-source projects:

- [TinyFace](https://github.com/idootop/TinyFace): The minimalist face swapping tool that just works. 
- [FaceFusion](https://github.com/facefusion/facefusion): Industry leading face manipulation platform
- [InsightFace](https://github.com/deepinsight/insightface): State-of-the-art 2D and 3D Face Analysis Project
- [Nuitka](https://github.com/Nuitka/Nuitka): Nuitka is a Python compiler written in Python.
- [Tauri](https://github.com/tauri-apps/tauri): Build smaller, faster, and more secure desktop and mobile applications with a web frontend.

## Disclaimer

MagicMirror is designed for personal entertainment and creative purposes only. Commercial use is prohibited. Please note:

- **Ethical Usage**: This software must not be used for activities including, but not limited to: a) impersonating others with malicious intent, b) spreading misinformation, c) violating personal privacy or dignity, d) creating explicit or inappropriate content.
- **Content Rights**: Users are responsible for: a) obtaining necessary permissions for source images, b) respecting copyrights and intellectual property, c) complying with local laws and regulations on AI-generated content.
- **Limitation of Liability**: The software and its developers are not liable for any user-generated content. Users assume full responsibility for the use of the generated content and any consequences that may arise from its use.

By using MagicMirror, you agree to these terms and commit to using the software responsibly.

## License

MIT License © 2024-PRESENT [Del Wang](https://del.wang)
