# MagicMirror 架构设计

> 本文档描述 MagicMirror 的整体架构、核心模块、进程模型与构建流程，帮助开发者快速理解代码库。

## 1. 系统总览

MagicMirror 是一个**完全离线**的 AI 换脸桌面应用，采用 **Tauri (Rust + React)** + **独立 Rust HTTP 推理服务** 的双进程架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     MagicMirror (Tauri 桌面端)                │
│  ┌───────────────────┐        ┌──────────────────────────┐  │
│  │   React 前端       │  invoke│   Tauri Rust 后端         │  │
│  │  (Vite + UnoCSS)  │ ─────▶ │  commands.rs 命令层        │  │
│  └───────────────────┘        │  ├─ spawn_server          │  │
│        │ fetch                │  ├─ kill_server           │  │
│        ▼                      │  ├─ is_server_running     │  │
│  http://localhost:8023        │  └─ download_and_unzip    │  │
└───────────┬─────────────────────────────────────────────────┘
            │ 子进程管理 (spawn / kill / 自动接管)
            ▼
┌─────────────────────────────────────────────────────────────┐
│                magic-server.exe (Rust HTTP 服务)             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Axum Router (REST API)                               │  │
│  │  /status  /prepare  /task                             │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          ▼ mpsc::channel                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Worker Pool (默认 1 个，--workers 可配)                │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  TinyFace (Arc<Mutex>, 模型共享只加载一次)          │  │  │
│  │  │  ├─ Detector  (SCRFD)    detector.rs             │  │  │
│  │  │  ├─ Embedder (ArcFace)   embedder.rs             │  │  │
│  │  │  ├─ Swapper  (inswapper) swapper.rs              │  │  │
│  │  │  └─ Enhancer (GFPGAN)    enhancer.rs (可选)       │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 设计原则

- **离线优先**：所有模型推理在本地完成，图片不上传任何服务器。
- **无 GPU 依赖**：ONNX Runtime 使用 DirectML 后端，普通 CPU 即可运行。
- **单文件部署**：`MagicMirror.exe` 自动派生/接管 `server.exe`，用户无感。
- **模块解耦**：桌面端（Tauri）与推理服务（magic-server）通过 HTTP + 进程管理解耦，服务可独立被第三方调用。

## 2. 进程模型与生命周期

### 2.1 进程关系

```
start.bat / 双击 MagicMirror.exe
        │
        ▼
MagicMirror.exe (PID A)
        │
        ├── spawn_server() 检测端口 8023
        │     ├── 未占用 → 派生 server.exe (CREATE_NO_WINDOW)
        │     ├── 已被本进程子进程占用 → 复用 (返回 Ok(true))
        │     └── 被残留进程占用 → taskkill 清理后重新派生
        │
        ▼
server.exe (PID B, 父进程 = MagicMirror.exe)
```

### 2.2 子进程管理 (`src-tauri/src/commands.rs`)

| 命令 | 作用 |
|------|------|
| `spawn_server()` | 检查端口 → 清理残留 → 无窗口派生 server.exe |
| `kill_server()` | 终止本进程派生的子进程 + taskkill 兜底清理孤儿 |
| `is_server_running()` | TcpStream 探测 8023 端口 |

**关键实现点：**

```rust
// Windows 无窗口后台启动
const CREATE_NO_WINDOW: u32 = 0x08000000;
const DETACHED_PROCESS: u32 = 0x00000008;

Command::new(&server_path)
    .arg("--workers").arg("auto")
    .current_dir(&exe_dir)            // 让 server 通过 CWD 找到 models/
    .stdin(Stdio::null())
    .stdout(Stdio::from(out_log))     // 输出重定向到 srv_out.log
    .stderr(Stdio::from(err_log))     // 便于诊断启动失败
    .creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS)
    .spawn()
```

**关键设计决策：**

1. **前端始终调用 `spawn_server()`**，不做 `is_server_running` 短路 —— 由 Rust 统一处理"接管/清理残留/启动"，避免前端绕过接管逻辑。
2. **退出清理双保险**：
   - Rust 层 `lib.rs` 的 `RunEvent::Exit → kill_spawned_server()`（覆盖所有退出路径）
   - 前端 Mirror 页 Quit 按钮先 `kill_server()` 再 `exit(0)`
3. **不依赖 React 组件卸载钩子清理**：`useServer` 的 unmount cleanup 已被移除 —— 从启动页导航到主页会触发卸载，导致刚启动的 server 被误杀（这是早期"server 很快关闭"的根因）。
4. **残留接管**：若 8023 被非本进程派生的 server.exe 占用（旧版残留、手动启动），先 `taskkill /f /im server.exe` 同步清理，再派生受控的新实例。
5. **启动校验**：spawn 后轮询端口最多 8 秒，期间用 `child.try_wait()` 检测子进程异常退出（如模型缺失），失败则清理并向上报错。

### 2.3 模型查找顺序 (server 端 `get_models_dir`)

```
1. {CWD}/models          ← 优先（MagicMirror 派生时 CWD = exe 目录）
2. {exe_dir}/models
3. {HOME}/MagicMirror/models
```

## 3. 推理服务 (src-server/)

### 3.1 技术栈

- **框架**：axum 0.8 + tokio（异步 HTTP）
- **推理**：ort 2.0.0-rc.13（ONNX Runtime）+ `directml` 特性
- **图像**：image 0.25 + jpeg-encoder 0.7（4:2:0 色度采样，对齐 OpenCV）
- **张量**：ndarray 0.17（**必须**与 ort 内部版本一致，勿升级）
- **CORS**：`allow_origin(Any)`（Tauri 前端运行于 `tauri.localhost`）

### 3.2 推理管线

```
输入图 ──▶ SCRFD 检测人脸 (detector.rs)
              │
              ▼
目标图 ──▶ SCRFD 检测人脸
              │
              ▼
ArcFace 提取目标身份嵌入 (embedder.rs)
              │
              ▼
inswapper 交换人脸 (swapper.rs)
              │
              ▼
GFPGAN 质量增强 (enhancer.rs, 可选)
              │  blend: 0.25*swap + 0.75*enhanced
              ▼
paste_back 贴回原图 (正向仿射) ──▶ 输出 JPG
```

### 3.3 Worker 系统 (`worker.rs`)

**架构：**

```
HTTP /task ──▶ mpsc::channel(10) ──▶ Worker (持有 Arc<Mutex<TinyFace>>)
                                         │
                                    [推理: 检测→嵌入→交换→增强→贴回]
                                         │
                                    mpsc::Sender<TaskResult> ──▶ HTTP 响应
```

**Worker 数量计算：**

```
worker 数 = min(CPU 核心数, 可用内存 / 300MB)
            └─────────┬────────┘
              主要限制（计算密集任务不能超核数）

可用内存 = 总内存 - 共享模型(~774MB) - 安全余量(2GB)
```

**命令行参数：**

```bash
server.exe               # 默认 1 个 worker
server.exe --workers 4   # 指定 4 个
server.exe --workers auto  # 自动计算
```

**并发模型：**
- 当前实现为**单 worker 顺序执行**，`Arc<Mutex<TinyFace>>` 保证 `&mut Session` 安全。
- 设计意图是共享模型内存（只加载一次），未来可扩展为多 worker 并行消费任务队列。
- **注意**：`TinyFace` 的模型是共享的，但 ONNX `Session::run` 需要 `&mut`，因此多 worker 并行时需确保模型会话的互斥访问（当前用 Mutex 串行化）。

### 3.4 REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查，返回 "MagicMirror" |
| `/status` | GET | 运行状态（`starting`/`running`）+ worker 数 |
| `/prepare` | POST | 等待模型加载完成（幂等，180s 超时） |
| `/task` | POST | 提交换脸任务（同步等待结果，120s 超时） |
| `/task/{id}` | DELETE | 取消任务（当前返回 405，未实现） |

**请求：**
```json
POST /task
{
  "id": "/path/to/input.jpg",     // 同时作为输出路径锚点
  "inputImage": "/path/to/source.jpg",   // 待换脸图（被替换方）
  "targetFace": "/path/to/identity.jpg"  // 身份来源图
}
```

**响应：**
```json
{ "taskId": "...", "result": "/path/to/input_output.jpg" }
```

**输出路径规则：**
- 输出保存到 **`id` 文件所在目录**（即输入图同目录），文件名 `{basename}_output.jpg`
- 若文件已存在，追加毫秒时间戳 `{basename}_output_{timestamp}.jpg` 避免覆盖

### 3.5 日志控制

```rust
// 只记录 ERROR/WARN，抑制 ONNX 刷屏
tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();
std::env::set_var("ORT_LOGGING_LEVEL", "Error");
std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");
```

server 的 stdout/stderr 被 Tauri 重定向到 `srv_out.log` / `srv_err.log`，排查问题看这两个文件。

## 4. 桌面端 (src-tauri/)

### 4.1 前端技术栈

- React 18 + TypeScript + Vite 5
- UnoCSS（原子化 CSS）
- react-i18next（中英双语）
- react-router-dom（Launch 页 → Mirror 页）
- Tauri Plugin：`process`、`shell`、`os`

### 4.2 启动流程

```
LaunchPage 挂载
  └─ useDownload().download()    检查 server.exe 是否存在
      └─ useServer().launch()
          ├─ invoke("spawn_server")    始终调用，Rust 处理接管
          ├─ 轮询 is_server_running    最多 30s
          ├─ 等待 15s（模型加载）
          └─ POST /prepare (180s 超时)
              └─ navigate("/mirror")
```

### 4.3 换脸流程 (Mirror 页)

```
拖拽/选择图片 → useDragDrop
  └─ useSwapFace
      ├─ 上传 a.jpg (待换脸图) + b.png (身份图)
      └─ POST /task { id, inputImage, targetFace }
          └─ 获取 result 路径 → 预览
```

### 4.4 窗口配置

- 650×650 无边框透明窗口（`decorations: false, transparent: true`）
- CSP：`connect-src` 需包含 `http://localhost:8023` 以访问本地服务

## 5. 便携版（绿色版）打包

### 5.1 产物结构

```
MagicMirror-Portable/
├── MagicMirror.exe     # Tauri 桌面端（自动派生 server）
├── server.exe          # Rust 推理服务
├── models/             # ONNX 模型（共 5 个，约 774MB）
│   ├── scrfd_2.5g.onnx
│   ├── arcface_w600k_r50.onnx
│   ├── inswapper_128_fp16.onnx
│   ├── inswapper_weight.bin
│   └── gfpgan_1.4.onnx
├── scripts/batch-swap.bat   # 批量换脸脚本
└── start.bat           # 仅启动 MagicMirror.exe
```

### 5.2 模型清单

> 完整模型说明、输入输出规格、下载地址与存放方式见 [模型清单与下载指南](./models.md)。

| 模型 | 大小 | 用途 | 必需 |
|------|------|------|------|
| `scrfd_2.5g.onnx` | 3.3 MB | 人脸检测 | 是 |
| `arcface_w600k_r50.onnx` | 174 MB | 人脸嵌入 | 是 |
| `inswapper_128_fp16.onnx` | 277 MB | 人脸交换（Rust 用 fp16） | 是 |
| `inswapper_weight.bin` | 1 MB | 交换权重矩阵 | 是 |
| `gfpgan_1.4.onnx` | 340 MB | 质量增强 | 推荐 |

### 5.3 批量换脸 (`scripts/batch-swap.bat`)

```
自动模式:  batch-swap.bat                  # 查找 a.jpg 作为源脸，对其余图片换脸
手动模式:  batch-swap.bat source.jpg img1.jpg img2.jpg
```

脚本逻辑：netstat 检测 8023 → 未运行则启动 server → 轮询 /status → curl 调用 /prepare 与 /task。

### 5.4 构建命令

```bash
# 前端
pnpm build                    # Vite 构建到 dist/

# Tauri 桌面端 (targets 已设为 []，仅产出 exe 不打包安装器)
pnpm tauri build

# Rust server
cd src-server && cargo build --release

# 手动打便携版 zip（参考 scripts/build-all.ps1 或人工复制上述产物）
```

## 6. 关键代码约定

### 6.1 Rust 服务端规范 (src-server/)

- 错误处理统一使用 `Result<T, InferenceError>`，禁止 `.unwrap()`
- 所有模型组件实现 `OnnxModel` trait（`load()` / `prepare()` / `session()`）
- 图像转换工具集中在 `image_utils.rs`，避免重复定义 `ImageData`
- `ndarray` 锁定 `0.17`，与 ort 内部版本一致

### 6.2 图像处理对齐 OpenCV 的关键点（易踩坑）

1. **浮点转整型用 `.round() as u8`**：cv2 用 `round()`，Rust `as u8` 是截断，差 0.5 会系统性偏移。
2. **JPEG 用 `jpeg_encoder` 且 `SamplingFactor::R_4_2_0`**：默认 `image` crate 是 4:4:4，文件大 21% 且与 cv2 输出不一致。
3. **仿射变换用正向**：`paste_back` 用 `transform_point(affine, x, y)` 而非逆矩阵，否则 mask 为空、输出等于原图。
4. **GFPGAN 增强的是换脸结果**，不是原图，否则增强会把结果拉回原图。
5. **SCRFD anchor 的 Y 轴从上到下**（`cy = i * stride`），颠倒会导致 landmarks 翻转。

### 6.3 前端约定

- 子进程清理**只在 Rust 层**（`RunEvent::Exit`），前端组件卸载钩子不得调用 `kill_server`（会误杀）。
- server 输出路径由 `id` 字段决定（等于输入图路径），前端传完整路径即可。

## 7. 目录结构

```
MagicMirror/
├── src-tauri/               # Tauri 桌面端
│   ├── src/
│   │   ├── main.rs          # 入口
│   │   ├── lib.rs           # Builder + RunEvent::Exit 清理
│   │   ├── commands.rs      # spawn/kill server 等 Tauri 命令
│   │   └── utils.rs         # 下载/解压工具
│   ├── capabilities/default.json
│   └── tauri.conf.json
├── src-server/              # Rust 推理服务
│   └── src/
│       ├── main.rs          # Axum HTTP + 参数解析 + CORS
│       ├── worker.rs        # Worker 池 + 自动计算数量
│       ├── lib.rs
│       └── inference/
│           ├── mod.rs       # TinyFace 编排
│           ├── detector.rs  # SCRFD
│           ├── embedder.rs  # ArcFace
│           ├── swapper.rs   # inswapper
│           ├── enhancer.rs  # GFPGAN
│           ├── warp.rs      # 仿射变换
│           └── image_utils.rs
├── src-python/              # Python 参考实现（原始版本，仅作基线对比）
├── scripts/                 # 构建与批量脚本
├── tests/                   # 测试脚本与基准图
│   └── fixtures/            # a.jpg / b.png / py_tinyface_baseline.jpg
├── docs/                    # 文档
└── out/                     # 构建产物（gitignore）
```

## 8. 测试与验证基线

### 8.1 测试图片

- 源图（待换脸）：`tests/fixtures/a.jpg` (1035×690)
- 身份图：`tests/fixtures/b.png` (250×188)
- Python 基线：`tests/fixtures/py_tinyface_baseline.jpg`

### 8.2 相似度基线

Rust 输出 vs Python 基线：**>87% 像素差值 ≤5，<1% 像素差值 >20，平均差 ~2.3**。

### 8.3 端到端验证清单

- [ ] 干净环境启动：`server.exe` 是 `MagicMirror.exe` 子进程，无窗口，监听 8023
- [ ] 残留接管：占用 8023 的残留 server 被清理，启动受控新实例
- [ ] 正常关闭（WM_CLOSE）：server 跟随退出，端口释放
- [ ] `/status` 返回 `{"status":"running","workers":N}`
- [ ] 换脸：POST /task 返回可访问的输出路径
