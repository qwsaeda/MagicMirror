# MagicMirror 批量换脸脚本（放在图片目录运行）
#
# 用法:
#   1. 自动模式: 双击 batch-swap.bat，源脸图片命名为 a.jpg 放同一目录
#   2. 手动模式: batch-swap.bat <源脸图片> <目标图片1> <目标图片2> ...
#
# 说明:
#   - 会自动查找 MagicMirror 安装目录（可用环境变量 MM_HOME 指定）
#   - server 在后台启动，日志写到 <安装目录>\srv_out.log / srv_err.log
#   - 输出为 <被换图片名>_output.jpg，保存在被换图片同一目录
#   - 已存在的输出文件（*_output*）不会被再次处理

$ErrorActionPreference = 'Stop'

# 图片目录 = 脚本所在目录
$imgDir = $PSScriptRoot

function Pause-Exit {
    param([int]$Code = 1)
    Read-Host '按回车键退出' | Out-Null
    exit $Code
}

# ---------- 定位应用目录 ----------
function Find-AppDir {
    # 1) 环境变量
    if ($env:MM_HOME -and (Test-Path -LiteralPath (Join-Path $env:MM_HOME 'server.exe'))) {
        return $env:MM_HOME
    }
    # 2) 注册表卸载信息
    $regKeys = @(
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MagicMirror',
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MagicMirror',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\MagicMirror'
    )
    foreach ($k in $regKeys) {
        if (Test-Path -LiteralPath $k) {
            $loc = (Get-ItemProperty -LiteralPath $k -ErrorAction SilentlyContinue).InstallLocation
            if ($loc -and (Test-Path -LiteralPath (Join-Path $loc 'server.exe'))) { return $loc }
        }
    }
    # 3) 常见安装位置
    $candidates = @(
        "$env:LOCALAPPDATA\MagicMirror",
        "$env:LOCALAPPDATA\Programs\MagicMirror",
        "$env:ProgramFiles\MagicMirror",
        "${env:ProgramFiles(x86)}\MagicMirror"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path -LiteralPath (Join-Path $c 'server.exe'))) { return $c }
    }
    return $null
}

$appDir = Find-AppDir
if (-not $appDir) {
    Write-Host '[INFO] 未自动找到 MagicMirror 安装目录'
    $typed = (Read-Host '请输入安装目录（含 server.exe 的文件夹）').Trim().Trim('"')
    if ($typed -and (Test-Path -LiteralPath (Join-Path $typed 'server.exe'))) { $appDir = $typed }
}
if (-not $appDir -or -not (Test-Path -LiteralPath (Join-Path $appDir 'server.exe'))) {
    Write-Host '[错误] 找不到 server.exe。可先用环境变量 MM_HOME 指定安装目录再运行'
    Pause-Exit
}
Write-Host "[INFO] 应用目录: $appDir"

# ---------- 检测/启动 server ----------
function Test-ServerPort {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync('127.0.0.1', 8023)
        if ($task.Wait(1000) -and $client.Connected) { return $true }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

if (Test-ServerPort) {
    Write-Host '[INFO] Server 已在运行，直接复用'
} else {
    Write-Host "[INFO] 启动 MagicMirror server..."
    $serverExe = Join-Path $appDir 'server.exe'
    try {
        Start-Process -FilePath $serverExe -ArgumentList '--workers', 'auto' `
            -WorkingDirectory $appDir -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $appDir 'srv_out.log') `
            -RedirectStandardError (Join-Path $appDir 'srv_err.log')
    } catch {
        # 个别环境下隐藏窗口与日志重定向不兼容，退化为仅隐藏启动
        Start-Process -FilePath $serverExe -ArgumentList '--workers', 'auto' `
            -WorkingDirectory $appDir -WindowStyle Hidden
    }
}

# 等待端口监听（最多 60 秒）
$waited = 0
while (-not (Test-ServerPort)) {
    $waited += 2
    if ($waited -ge 60) {
        Write-Host "[错误] Server 启动超时，请查看 $appDir\srv_out.log"
        Pause-Exit
    }
    Start-Sleep -Seconds 2
}

# ---------- 等待模型加载 ----------
Write-Host '[INFO] 等待模型加载（首次启动约需 30 秒）...'
try {
    $prep = Invoke-RestMethod -Uri 'http://127.0.0.1:8023/prepare' -Method Post `
        -Body '{}' -ContentType 'application/json' -TimeoutSec 200
} catch {
    Write-Host "[错误] 无法访问 server: $($_.Exception.Message)"
    Write-Host "请查看 $appDir\srv_out.log"
    Pause-Exit
}
if (-not $prep.success) {
    Write-Host '[错误] 模型加载失败或超时，请查看 srv_out.log'
    Pause-Exit
}
Write-Host '[INFO] Server 已就绪'
Write-Host ''

# ---------- 调用换脸接口 ----------
# server 语义: inputImage = 被换脸的图（输出写它旁边）; targetFace = 身份源
function Invoke-SwapTask {
    param([string]$SrcPath, [string]$InputPath)
    $payload = @{
        id         = 'swap_' + [guid]::NewGuid().ToString('N')
        inputImage = $InputPath
        targetFace = $SrcPath
    } | ConvertTo-Json -Compress
    # 显式按 UTF-8 发送，避免中文路径被错误编码
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $resp = Invoke-RestMethod -Uri 'http://127.0.0.1:8023/task' -Method Post `
        -Body $bodyBytes -ContentType 'application/json; charset=utf-8' -TimeoutSec 180
    return $resp.result
}

function Resolve-ImgPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $imgDir $Path
}

# ---------- 收集任务 ----------
$srcPath = $null
$targets = @()

if ($args.Count -gt 0) {
    # 手动模式: 第 1 个参数为源脸，其余为待处理图片
    $srcPath = Resolve-ImgPath $args[0]
    if (-not (Test-Path -LiteralPath $srcPath)) {
        Write-Host "[错误] 找不到源脸图片: $srcPath"
        Pause-Exit
    }
    if ($args.Count -lt 2) {
        Write-Host '[错误] 手动模式至少需要一张待处理图片'
        Write-Host '用法: batch-swap.bat <源脸图片> <目标图片1> <目标图片2> ...'
        Pause-Exit
    }
    for ($i = 1; $i -lt $args.Count; $i++) {
        $targets += Resolve-ImgPath $args[$i]
    }
} else {
    # 自动模式: 源脸为 a.jpg，处理目录下所有 jpg/jpeg/png
    $srcPath = Join-Path $imgDir 'a.jpg'
    if (-not (Test-Path -LiteralPath $srcPath)) {
        Write-Host '[错误] 找不到源脸图片 a.jpg'
        Write-Host '请把要换上的源脸命名为 a.jpg，和待处理图片放在同一目录'
        Pause-Exit
    }
    $targets = Get-ChildItem -Path (Join-Path $imgDir '*') -File -Include *.jpg, *.jpeg, *.png |
        Where-Object { $_.Name -ine 'a.jpg' -and $_.Name -notmatch '_output' } |
        ForEach-Object { $_.FullName }
}

if ($targets.Count -eq 0) {
    Write-Host '[错误] 没有找到可处理的图片（只支持 jpg/jpeg/png）'
    Pause-Exit
}

Write-Host "[INFO] 源脸: $srcPath"
Write-Host "[INFO] 共 $($targets.Count) 张图片待处理"
Write-Host ''

# ---------- 批量处理 ----------
$ok = 0
$fail = 0
foreach ($t in $targets) {
    Write-Host "[INFO] 正在处理: $t"
    try {
        $out = Invoke-SwapTask -SrcPath $srcPath -InputPath $t
        Write-Host "[OK] 已输出: $out"
        $ok++
    } catch {
        Write-Host "[错误] 处理失败: $t ($($_.Exception.Message))"
        $fail++
    }
    Write-Host ''
}

Write-Host "[完成] 成功 $ok 张，失败 $fail 张"
Pause-Exit 0
