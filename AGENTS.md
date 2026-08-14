# MagicMirror Agent Instructions

## Project Structure

```
MagicMirror/
├── src-tauri/          # Tauri desktop app (Rust + TypeScript/React frontend)
├── src-server/         # Rust HTTP server for ONNX inference
├── src-python/         # Python inference scripts and ONNX models
├── scripts/            # Build and utility scripts
│   ├── build-all.ps1       # Unified packaging script
│   ├── build-server-rust.ps1  # Server-only packaging
│   ├── build-server-rust.sh
│   ├── build-server.sh
│   ├── dist.js
│   ├── face-swap.ps1       # User-facing swap script
│   └── face-swap.bat
├── tests/              # Test scripts and data
│   ├── fixtures/           # Test images and data files
│   │   ├── a.jpg           # Source face (1035x690)
│   │   ├── b.png           # Target identity (250x188)
│   │   ├── c_a_to_b.jpg    # Rust server output
│   │   ├── output1.jpg     # Python reference output
│   │   ├── py_tinyface_baseline.jpg  # Python baseline
│   │   ├── py_affine.npy
│   │   └── py_meta.txt
│   ├── compare_diff.py
│   ├── compare_outputs.py
│   ├── ... (debug scripts)
├── docs/               # Documentation
│   └── packaging.md
├── out/                # Build artifacts (generated)
├── dist/               # Built frontend assets (generated)
└── start.bat           # Dev one-click start

## Key Commands

| Command | Location | Purpose |
|---------|----------|---------|
| `pnpm dev` | root | Frontend dev server (Vite) |
| `pnpm build` | root | Build frontend |
| `pnpm tauri dev` | root | Full Tauri app development |
| `cargo check` | src-server/ | Check Rust compilation |
| `cargo clippy` | src-server/ | Lint Rust code |
| `cargo build --release` | src-server/ | Build Rust server |
| `bash scripts/build-server-rust.sh` | root | Full server distribution package |
| `server.exe --workers auto` | - | Start with auto-calculated workers |
| `server.exe --workers 4` | - | Start with 4 workers |

## Rust Server Architecture (src-server/)

### Inference Pipeline

```
TinyFace (orchestrator)
├── Detector (SCRFD)     -> src/inference/detector.rs
├── Embedder (ArcFace)   -> src/inference/embedder.rs
├── Swapper (inswapper)  -> src/inference/swapper.rs
└── Enhancer (GFPGAN)    -> src/inference/enhancer.rs (optional)
```

### Key Types and Traits

```rust
// Result type alias for inference operations
type InferenceResult<T> = Result<T, InferenceError>;

// Common trait for all ONNX model components
trait OnnxModel {
    fn load(&mut self, model_path: impl AsRef<Path>) -> InferenceResult<()>;
    fn prepare(&self) -> InferenceResult<()>;
    fn session(&self) -> Option<&Session>;
}

// Data structures
struct FaceBox { x1, y1, x2, y2, score, landmarks: [[f32; 5]; 2] }
struct FaceCrop { data: Vec<u8>, width, height, channels }
struct TinyFace { detector, embedder, swapper, enhancer: Option }
```

### ort API Usage (v2.0.0-rc.13)

```rust
// Session requires &mut reference for run()
let session = self.session.as_mut().ok_or(InferenceError::NotLoaded)?;

// Input tensor creation
let input_tensor = preprocess_image(img);
let outputs = session.run(inputs!["input" => TensorRef::from_array_view(&input_tensor)?])?;

// Output extraction
let output = outputs.into_iter().next()
    .ok_or_else(|| InferenceError::Onnx(ort::Error::new("No outputs")))?;
let array: ndarray::ArrayViewD<'_, f32> = output.1.try_extract_array::<f32>()?;
```

### ndarray Version Constraint

`Cargo.toml` specifies `ndarray = "0.17"` to match ort's internal version. Do not upgrade without checking ort compatibility.

### Models Location

Models are stored at: `C:\Users\Administrator\MagicMirror\models\`

- `scrfd_2.5g.onnx` - Face detection (3.3 MB)
- `arcface_w600k_r50.onnx` - Face embedding (174 MB)
- `inswapper_128_fp16.onnx` - Face swapping (277 MB, **used by Rust**)
- `inswapper_128.onnx` - Face swapping (554 MB, fp32, **used by Python only**)
- `inswapper_weight.bin` - Weight matrix (1 MB, **used by Rust**)
- `inswapper_weight.npy` - Weight matrix numpy (1 MB, **used by Python only**)
- `gfpgan_1.4.onnx` - Quality enhancement (340 MB, optional)

### Models Expected

- `scrfd_2.5g.onnx` - Face detection (required)
- `arcface_w600k_r50.onnx` - Face embedding (required)
- `inswapper_128_fp16.onnx` - Face swapping (required)
- `inswapper_weight.bin` - Weight matrix (required)
- `gfpgan_1.4.onnx` - Quality enhancement (optional, recommended)

## Rust 代码规范 (src-server/)

### 代码风格

1. **错误处理统一**
   - 使用 `InferenceResult<T>` 类型别名
   - 所有错误返回 `Result<T, InferenceError>`
   - 禁止使用 `.unwrap()`，改用 `?` 或 `.ok_or()`

2. **避免重复代码**
   - 提取公共逻辑到辅助函数
   - 图像转换函数统一放在 `image_utils.rs`
   - 不要在不同文件中定义相同的 `ImageData` 结构体

3. **使用类型别名**
   ```rust
   type InferenceResult<T> = Result<T, InferenceError>;
   type Embedding = Vec<f32>;
   type ImageBytes = Vec<u8>;
   ```

4. **使用宏简化重复代码**
   ```rust
   // 为 ONNX 模型实现统一接口
   macro_rules! impl_onnx_model {
       ($struct_name:ident) => {
           impl OnnxModel for $struct_name {
               fn load(&mut self, model_path: impl AsRef<Path>) -> InferenceResult<()> { ... }
               fn prepare(&self) -> InferenceResult<()> { ... }
               fn session(&self) -> Option<&Session> { ... }
           }
       };
   }
   ```

5. **Trait 设计**
   - 所有模型组件实现 `OnnxModel` trait
   - 统一 `load()` / `prepare()` / `session()` 接口

### API 设计原则

1. **统一错误响应格式**
   ```rust
   #[derive(serde::Serialize)]
   pub struct ApiError {
       pub code: u16,
       pub message: String,
   }
   ```

2. **接口幂等性**
   - `/status` GET - 无副作用
   - `/prepare` POST - 可重复调用
   - `/task` POST - 幂等（相同 id 返回相同结果）

3. **输出路径唯一性**
   - 使用 `payload.id` 命名输出文件，避免并发冲突
   - 格式：`{basename}_{id}_output.jpg`

### 并发安全规范

1. **使用 `Arc<RwLock<T>>` 共享状态**
   ```rust
   struct AppState {
       tinyface: Arc<RwLock<TinyFace>>,
       prepared: Arc<RwLock<bool>>,
   }
   ```

2. **写锁保护模型加载**
   - `prepare` handler 获取写锁加载模型
   - 其他请求获取读锁等待

3. **任务队列（未来优化）**
   - 当前：同步处理，阻塞其他请求
   - 建议：使用 `tokio::sync::mpsc` 实现异步任务队列

## Frontend

- TypeScript/React with Vite
- UnoCSS for styling
- i18next for internationalization
- Windows: 650x650 transparent frameless window

## Build Artifacts

- Frontend: `dist/`
- Rust server: `src-server/target/release/magic-server.exe`
- Server distribution: `out/server.zip` (created by `build-server-rust.sh`)

## Test Images (VERIFY BEFORE RUNNING)

Test image filenames are easy to get wrong and cause hours of confusion. Always confirm with `Get-ChildItem F:\MagicMirror\tests\fixtures` first.

- Source image (face to be replaced): **`F:\MagicMirror\tests\fixtures\a.jpg`** (1035x690)
- Target image (identity source): **`F:\MagicMirror\tests\fixtures\b.png`** (250x188)

DO NOT assume `a.png` / `b.jpg`. If the server returns "Read input failed: 系统找不到指定的文件" it means the filename is wrong — the swap "looks like nothing happened" because the file was never read.

## Common Issues

1. **onnxruntime.dll not found on Windows**: Build script copies it from ort crate; ensure release build completes fully
2. **Session borrow errors**: ort Session requires `&mut self` for `.run()` - use `as_mut()` not `as_ref()`
3. **ndarray version mismatch**: ort v2.0.0-rc.13 expects ndarray 0.17; check Cargo.lock if upgrading
4. **Error type conversion**: Use `InferenceError::Onnx(ort::Error::new("msg"))` not `.to_string()`
5. **"No face swap" is almost always wrong test filename**: The swap pipeline works (Rust vs Python baseline: 99% pixel similarity). If the output looks unswapped, the input image paths in the task request are wrong. Server logs the read error in srv_out.log.
6. **ort session load is slow and logs profusely**: Model loading (esp. GFPGAN 340MB) takes ~15s and floods logs with "Reserving memory in BFCArena". This is normal, not a hang. Check `srv_out.log` / `srv_err.log` for the real error near the `Reading input image:` line.
7. **Server output path follows CWD**: `a_output.jpg` is written to the server's current working directory, NOT next to the input image. When started from `src-server\`, output lands in `src-server\a_output.jpg`.
8. **"No face swap" output = original image**: Check `paste_back` affine direction (forward, not inverse). Check SCRFD anchor Y axis (top-to-bottom, not bottom-to-top). Check `estimate_similarity_transform` rotation (polar decomposition or linear solver, not buggy SVD).
9. **JPEG file too large**: Default `image` crate encoder uses 4:4:4 chroma. Use `jpeg-encoder` crate with `SamplingFactor::R_4_2_0` for 4:2:0 (matches cv2).
10. **Systematic pixel offset ~0.5**: cv2 uses `round()` for float-to-int conversion; Rust `as u8` truncates. Use `.round() as u8` everywhere.

## Validation Baseline

Python reference output (correct, verified): `F:\MagicMirror\tests\fixtures\py_tinyface_baseline.jpg` (generated by `run_tinyface3.py` using real a.jpg/b.png).

Rust vs Python expected similarity: **>87% pixels within 5, <1% pixels differing by >20**, mean diff ~2.3.

## Lessons Learned (Debugging Face Swap Pipeline)

### Root Cause of "No Face Swap" (output ≈ original image)

The number one symptom — swapped output looks identical to input — was caused by **mask being empty** in `paste_back`. The mask is empty when the affine transform maps original pixels outside the 128x128 template. Debug signature: `Result mean diff from input: 0.00` in test_swap output.

### All Fixed Bugs

| # | File | Bug | Symptom | Fix |
|---|------|-----|---------|-----|
| 1 | `swapper.rs` `paste_back` | Used **inverse** affine (`transform_point(&inv, x, y)`) instead of forward affine | mask empty → output = original | Use `transform_point(affine, x, y)` |
| 2 | `detector.rs` `generate_anchors` | `cy = (feat_h - 1 - i) * stride` (bottom-to-top from `[::-1]`) | landmarks Y flipped, face aligned wrong | `cy = i * stride` (top-to-bottom) |
| 3 | `warp.rs` `estimate_similarity_transform` | Hand-written 2x2 SVD returned zero rotation (r01=r10=0) | face not rotated to template | Replaced with linear least-squares solver matching OpenCV `estimateAffinePartial2D` |
| 4 | `warp.rs` `ARCFACE_WARP_TEMPLATE` | Points 4,5 Y = 0.575/0.573 (should be 0.824/0.823) | wrong embedding alignment | Corrected values |
| 5 | `swapper.rs` `swap` | Model outputs RGB, code converted to BGR, then `paste_back` treated as RGB | R/B channels swapped | Output RGB directly |
| 6 | `warp.rs` solver | `atb[1]` sign wrong: `sy*dx - sx*dy` | affine dramatically wrong (b = 10.46 vs -0.06) | `sx*dy - sy*dx` |
| 7 | `mod.rs` GFPGAN | Warped the **original** image for GFPGAN input, not the swapped result | enhancer dragged result toward original | Warp from `result` (swapped image) |
| 8 | `swapper.rs`/`enhancer.rs` | `as u8` (truncate) instead of `.round()` | systematic ~0.5 offset vs cv2 (cv2 rounds) | `.round() as u8` in warp, paste_back, model decode |
| 9 | `main.rs` | `rgb_image.save()` default JPEG quality + 4:4:4 chroma | file 87KB vs Python 197KB; 21% larger from 4:4:4 | `jpeg_encoder::Encoder::new(&mut file, 95)` + `set_sampling_factor(SamplingFactor::R_4_2_0)` |

### Key Debugging Techniques

1. **print affine matrix, not just final image**: Compare Rust affine vs Python `cv2.estimateAffinePartial2D` — the solver must match exactly (Rust linear solver == RANSAC for 5 inlier points).
2. **Decompose the difference**: Compare Rust vs Python at each stage (affine → warp crop → embedding cos → swapper output → paste). Use `cos` similarity for embeddings, `mean abs diff` for images.
3. **Simulate Rust in Python**: Replicate the exact Rust pipeline in Python (same affine formula, bilinear warp, box mask) to isolate whether Rust deviates from its own intended logic vs from Python.
4. **Check JPEG SOF sampling factors**: Rust `image` crate encodes 4:4:4 by default; cv2 uses 4:2:0. This makes files ~21% larger with no quality gain. Use `jpeg-encoder` crate with `SamplingFactor::R_4_2_0`.
5. **cv2 rounds, Rust truncates**: Every `as u8` on a float near x.5 differs by 1 from cv2. Use `.round()`.
6. **GFPGAN must enhance the SWAPPED face**: `enhance_face(temp_vision_frame=swapped_result, ...)` — warp the output, not the input.

### 并发 Worker 系统设计 (2026-08-13)

**设计理念：**
- 共享模型内存（只加载一次，所有 worker 共享）
- 每个 worker 独立推理状态（锁保护）
- 通过命令行参数控制 worker 数量

**命令行参数：**
```bash
# 自动计算 worker 数量（基于内存和 CPU）
server.exe --workers auto

# 指定 worker 数量
server.exe --workers 4

# 默认单 worker
server.exe
```

**内存计算：**
```
Worker数量 = min(CPU核心数, 可用内存 / 每Worker峰值内存)

注意：Worker数量不能超过CPU核心数，因为是密集计算任务。

示例（8GB 内存，4核 CPU）：
- CPU 限制: 4 个 (主要限制)
- 可用内存: ~6GB (扣除安全余量)
- 每 Worker 峰值: ~300MB
- 内存上限: 20 个
- 实际使用: 4 个 (取 min)
```

**架构：**
```
请求 → Task Channel → Worker (Mutex<TinyFace>)
                         ↓
                    [推理]
                         ↓
                    返回结果
```

## Frontend

| 优先级 | 问题 | 文件 | 修复 |
|--------|------|------|------|
| P0 | 并发输出文件名冲突 | `main.rs` | 使用 `payload.id` 生成唯一路径 |
| P0 | `cancel_task` 虚假成功 | `main.rs` | 返回 405 Method Not Allowed |
| P1 | 死代码: `image_utils.rs` 重复 `ImageData` | `image_utils.rs` | 删除死代码，保留工具函数 |
| P1 | `swap_face_simple` 未使用 | `mod.rs` | 添加 `#[allow(dead_code)]` |
| P1 | `get_one_face` 未使用 | `mod.rs` | 添加 `#[allow(dead_code)]` |
| P1 | `crop_face` 冗余 min 调用 | `detector.rs` | 移除重复的 `.min(img.width)` |
| P2 | `has_weight` 每次 infer 重新计算 | `enhancer.rs` | 缓存到 struct 字段 |
| P2 | 矩阵乘法性能 | `swapper.rs` | 使用 `ndarray` 优化 |
| P3 | `l2_normalize` 重复定义 | `image_utils.rs` | 统一到一个文件 |

## Frontend

- Enhancer blend: Python uses `0.4*swap + 0.6*enhanced` (`face_enhancer_blend=60`). Raising Rust to `0.25*swap + 0.75*enhanced` improves face sharpness +54% (159.6 vs 103.9 Laplacian variance) with negligible color loss (-1.6 saturation).
- Face sharpness indicator: `cv2.Laplacian(face, CV_64F).var()`.
- Color saturation indicator: `cv2.cvtColor(face, BGR2HSV)[:,:,1].mean()`.

## Portable Mode (Green Version) Issues & Fixes

### Problem: Frontend stuck on "启动中..."

**Root Cause**: Multiple issues accumulated:
1. **CORS blocked** - Tauri frontend runs on `http://tauri.localhost`, but server had no CORS headers
2. **ONNX Runtime excessive logging** - Model loading flooded console with INFO logs, masking real errors
3. **Prepare timeout too short** - GFPGAN 340MB needs ~15s to load, default 60s timeout not enough
4. **Output path was relative** - Server returned `"a_output.jpg"` but Tauri needed absolute path

**Fixes Applied**:

| Issue | Fix |
|-------|-----|
| CORS blocked | Added `CorsLayer::allow_origin(Any)` to server router |
| ONNX excessive logs | Set `ORT_LOGGING_LEVEL=Error` env var + tracing WARN level |
| Prepare timeout | Increased to 180s (`AbortSignal.timeout(180000)`) |
| Output path | Use `cwd.join("{basename}_output.jpg")` for absolute path |
| Model load time | Add 15s delay before calling `/prepare` after server ready |

**Key Code Changes**:

```rust
// src-server/src/main.rs - Add CORS middleware
let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods([axum::http::Method::GET, axum::http::Method::POST, axum::http::Method::DELETE])
    .allow_headers(Any);
// ... apply .layer(cors) to router

// Disable ONNX Runtime verbose logging
std::env::set_var("ORT_LOGGING_LEVEL", "Error");
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::WARN)
    .init();
```

```typescript
// src/services/server.ts
async prepare(): Promise<boolean> {
  const res = await fetch(`${this._baseURL}/prepare`, {
    method: "post",
    signal: AbortSignal.timeout(180000), // 3 minutes
  });
  // ...
}

// Wait for server to be fully ready (models loading takes time)
await new Promise((r) => setTimeout(r, 15000));
const prepared = await this.prepare();
```

### Portable Startup Flow

```
start.bat
  ↓ start "" server.exe (background)
  ↓ timeout 15s (wait for models to load)
  ↓ start "" MagicMirror.exe
  ↓
LaunchPage
  ↓ download() → check server.exe exists
  ↓ launch() → check /status → if running, call /prepare
  ↓ navigate("/mirror") when prepared=true
```

### Model Loading Time

| Model | Size | Load Time |
|-------|------|-----------|
| scrfd_2.5g.onnx | 3 MB | ~1s |
| arcface_w600k_r50.onnx | 166 MB | ~5s |
| inswapper_128_fp16.onnx | 265 MB | ~5s |
| gfpgan_1.4.onnx | 340 MB | ~15s |
| **Total** | **~774 MB** | **~30s** |

### Debugging Checklist

If frontend is stuck on "启动中...":
1. Open F12 → Console
2. Look for CORS errors: `blocked by CORS policy`
3. Look for prepare timeout: `TimeoutError: signal timed out`
4. Check server logs in separate console window
5. Verify models exist in `models/` directory

### API 接口设计

Server 提供 RESTful API 供第三方调用：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/status` | GET | 返回运行状态 |
| `/prepare` | POST | 加载模型（幂等） |
| `/task` | POST | 执行换脸 |
| `/task/{id}` | DELETE | 取消任务（当前未实现） |

**请求格式：**
```json
POST /task
{
  "id": "unique_task_id",
  "inputImage": "/path/to/source.jpg",
  "targetFace": "/path/to/identity.jpg"
}
```

**响应格式：**
```json
{
  "result": "/path/to/output.jpg"
}
```

**错误响应：**
```json
{
  "code": 400,
  "message": "Error description"
}
```

### Shell Scope Configuration

For portable mode, add these shell commands in `capabilities/default.json`:

```json
{
  "name": "server-windows-local",
  "cmd": "./server.exe",
  "args": true,
  "sidecar": false
}
```

And in `tauri.conf.json`:
```json
"resources": {
  "vendor/server.exe": "server.exe",
  "vendor/models/*": "models/"
}
```

## Server 子进程生命周期管理 (2026-08-14)

### 目标

MagicMirror 启动时自动派生 `server.exe` 子进程（无控制台窗口、后台运行），关闭时随主进程一起退出；保证主进程始终拥有受控的 server，不留孤儿。

### 架构

```
前端 launch() → invoke("spawn_server")  (始终调用，不做 is_server_running 短路)
                     │
                     ▼
Rust spawn_server()
  ├─ server.exe 不存在 → Err
  ├─ 端口 8023 未被占用 → 派生子进程 (CREATE_NO_WINDOW | DETACHED_PROCESS)
  ├─ 已被本进程子进程占用（且存活）→ Ok(true) 复用
  ├─ 被残留进程占用 → taskkill /f /im server.exe 同步清理 → 等待释放 → 派生新实例
  └─ spawn 后轮询端口 ≤8s，期间 try_wait() 检测子进程异常退出 → 失败则清理并报错

退出清理（双保险）：
  1. lib.rs RunEvent::Exit → kill_spawned_server()（覆盖所有退出路径，含 WM_CLOSE）
  2. 前端 Mirror 页 Quit 按钮：先 kill_server() 再 exit(0)
```

### 关键代码约定

1. **前端 `launch()` 必须始终调用 `spawn_server()`**，不要用 `is_server_running` 短路：
   否则端口被残留 server 占用时前端会走 prepare() 分支，绕过 Rust 的"清理残留→接管"逻辑，
   导致弹控制台的孤儿进程永久存在（早期 bug 根因）。
2. **禁止在 React 组件卸载钩子中调用 `Server.kill()`**：从 Launch 页 navigate 到 Mirror 页会触发
   Launch 组件卸载，导致刚启动的 server 被误杀（"server 很快关闭"的根因，已修复移除）。
   退出清理只由 Rust `RunEvent::Exit` 负责。
3. **无窗口后台启动**：`creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS)`，否则弹黑框。
4. **输出重定向到日志**：stdout/stderr 写入 `srv_out.log` / `srv_err.log`，不要 `Stdio::null()`，
   否则 server 启动失败（如模型缺失）无从诊断。
5. **current_dir 设为 exe 目录**：server 通过 CWD 找 `models/`（`get_models_dir` 优先查 CWD）。
6. **Mutex 锁中毒处理**：用 `lock().unwrap_or_else(|e| e.into_inner())`，勿静默吞掉 `if let Ok`。
7. **taskkill 兜底必须同步**：`.output()` 而非 `.spawn()`，否则残留清理与新派生存在竞态。
8. **spawn_server 返回语义**：`Ok(true)`=就绪（新派生或复用），`Ok(false)`=端口被其他进程占用且
   无法清理，`Err`=server.exe 缺失/派生失败。前端 `if (!spawned) return false`。

### 验证清单

启动 MagicMirror 后执行 `Get-CimInstance Win32_Process -Filter "Name='server.exe'"`：
- server.exe 的 ParentProcessId == MagicMirror.exe PID（确认子进程关系）
- server.exe MainWindowTitle 为空（无控制台窗口）
- netstat 8023 处于 LISTENING
- WM_CLOSE 关闭 MagicMirror 后 server.exe 同步退出、端口释放

详细设计见 `docs/architecture.md` 第 2 节。
