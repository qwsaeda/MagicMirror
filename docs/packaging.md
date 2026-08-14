# MagicMirror 打包方案

## 核心理念

**前端和 server 独立打包，用户解压到同目录即可使用。**

```
用户下载两个包，解压到同一目录：
  MagicMirror_Setup.exe  →  C:\Program Files\MagicMirror\MagicMirror.exe
  server.zip             →  C:\Program Files\MagicMirror\server.exe
                             C:\Program Files\MagicMirror\models/
```

## 产物清单

### 1. 桌面应用安装包

**文件名**: `MagicMirror_Setup.exe` (~50MB)

Tauri NSIS 安装器，安装到 `C:\Program Files\MagicMirror\`。

安装后目录结构：
```
C:\Program Files\MagicMirror\
├── MagicMirror.exe          # Tauri 桌面应用
├── start.bat                # 一键启动脚本
└── (用户需手动解压 server.zip 到此目录)
```

### 2. 服务器分发包

**文件名**: `server-windows-x86_64.zip` (full ~710MB / lite ~410MB)

解压后放入桌面应用同一目录即可：
```
server.zip 解压后:
├── server.exe               # Rust 推理服务器
└── models/
    ├── scrfd_2.5g.onnx          # 人脸检测 (3 MB)
    ├── arcface_w600k_r50.onnx   # 人脸嵌入 (166 MB)
    ├── inswapper_128_fp16.onnx  # 人脸交换 (265 MB)
    ├── inswapper_weight.bin     # 权重矩阵 (1 MB)
    └── gfpgan_1.4.onnx          # 质量增强 (324 MB, 仅 full 版)
```

### 3. 全量包（可选）

**文件名**: `MagicMirror-Full.zip` (full ~760MB / lite ~460MB)

包含所有内容，解压即用，无需安装器：
```
MagicMirror-Full.zip
├── MagicMirror.exe          # Tauri 桌面应用
├── server.exe               # Rust 推理服务器
├── models/
│   └── ...
├── start.bat                # 一键启动脚本
└── start.ps1                # PowerShell 启动脚本
```

## 使用方式

### 方式一：在线安装（推荐）

1. 下载 `MagicMirror_Setup.exe` 并安装
2. 首次启动应用，自动从 GitHub Releases 下载 `server-windows-x86_64-lite.zip`
3. 自动解压到安装目录，启动服务

### 方式二：离线安装

1. 下载 `MagicMirror_Setup.exe` 并安装
2. 下载 `server-windows-x86_64-full.zip`
3. 解压 server.zip 到安装目录（`C:\Program Files\MagicMirror\`）
4. 启动应用

### 方式三：全量包（便携版）

1. 下载 `MagicMirror-Full.zip`
2. 解压到任意目录
3. 运行 `start.bat` 或 `MagicMirror.exe`

## 构建流程

```powershell
# 1. 构建前端
pnpm build

# 2. 构建 Rust server
cd src-server
cargo build --release

# 3. 打包所有产物
.\scripts\build-all.ps1
```

## 产物输出

```
out/
├── MagicMirror_Setup.exe              # Tauri 安装包 (~50MB)
├── server-windows-x86_64-full.zip     # 服务器含 GFPGAN (~710MB)
├── server-windows-x86_64-lite.zip     # 服务器不含 GFPGAN (~410MB)
└── MagicMirror-Full.zip               # 全量包 (可选)
```

## 发布到 GitHub Releases

```
1. 构建所有产物
2. 上传到 GitHub Releases:
   - MagicMirror_Setup.exe
   - server-windows-x86_64-full.zip
   - server-windows-x86_64-lite.zip
   - MagicMirror-Full.zip (可选)
3. Tauri 前端通过 i18n 配置的 downloadURL 下载对应的 server.zip
```