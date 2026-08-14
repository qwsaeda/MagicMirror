@echo off
cd /d "%~dp0"

echo ========================================
echo   MagicMirror 开发环境一键启动
echo ========================================

:: 检查 Rust server 是否存在
if not exist "src-server\target\release\magic-server.exe" (
    echo [INFO] 构建 Rust server...
    cd src-server
    call cargo build --release
    cd ..
    if errorlevel 1 (
        echo [ERROR] Rust server 构建失败
        pause
        exit /b 1
    )
    echo [OK] Rust server 构建完成
)

:: 检查前端依赖
if not exist "node_modules" (
    echo [INFO] 安装前端依赖...
    call pnpm install
    if errorlevel 1 (
        echo [ERROR] 前端依赖安装失败
        pause
        exit /b 1
    )
    echo [OK] 前端依赖安装完成
)

:: 启动 Rust server (后台)
echo [INFO] 启动 Rust server (localhost:8023)...
start "MagicMirror Server" /B "src-server\target\release\magic-server.exe"

:: 等待服务器启动
echo [INFO] 等待服务器就绪...
timeout /t 3 /nobreak >nul

:: 启动前端 dev server
echo [INFO] 启动前端开发服务器 (localhost:1420)...
start "MagicMirror Frontend" cmd /c "pnpm dev & pause"

echo.
echo ========================================
echo   服务已启动!
echo   Frontend: http://localhost:1420
echo   Server:   http://localhost:8023
echo ========================================
echo.
echo 按任意键停止所有服务...
pause >nul

:: 停止服务
echo [INFO] 停止服务...
taskkill /f /im "magic-server.exe" >nul 2>&1
taskkill /f /im "node.exe" /fi "WINDOWTITLE eq MagicMirror Frontend*" >nul 2>&1
echo [OK] 服务已停止