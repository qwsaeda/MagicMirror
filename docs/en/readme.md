# MagicMirror (English)

Instant AI Face Swap — One click to a brand new you!

> For the Chinese version, see [中文版](../cn/readme.md).

This repo is a deeply reworked fork: the inference layer is **fully rewritten in pure Rust**, with a concurrent worker pool, portable "green" packaging, and robust child-process lifecycle management.

## Developer Documentation

| Doc | Description |
|-----|-------------|
| [Architecture Design](../architecture.md) | System overview, process model, worker pool, REST API, packaging, OpenCV-alignment tips |
| [Rust Server Guide](../rust-server.md) | Why Rust, implementation details, **ort (ONNX Runtime) usage**, pitfalls & lessons learned |

## User Guides

- [Installation](./install.md)
- [Usage](./usage.md)
- [FAQ](./faq.md)

## Get Started

1. Download the latest release from the [Releases](https://github.com/idootop/MagicMirror/releases) page.
2. Unzip the portable package.
3. Double-click `start.bat` (Windows) or `MagicMirror.exe` — the app spawns `server.exe` automatically in the background (no console window), loads the models, and cleans up on exit.

If you have any questions, please [submit an issue](https://github.com/idootop/MagicMirror/issues).
