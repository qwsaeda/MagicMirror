@echo off
rem MagicMirror 批量换脸启动器（逻辑在 batch-swap.ps1，两个文件需放同一目录）
rem 使用方法：把这两个文件放到图片目录，双击本文件运行
setlocal
if not exist "%~dp0batch-swap.ps1" (
    echo [错误] 找不到 batch-swap.ps1
    echo 请把 batch-swap.bat 和 batch-swap.ps1 放在同一目录
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0batch-swap.ps1" %*
if errorlevel 1 pause
