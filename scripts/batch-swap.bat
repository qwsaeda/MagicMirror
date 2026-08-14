@echo off
setlocal enabledelayedexpansion

:: MagicMirror 批量换脸脚本
:: 用法: 
::   1. 自动模式: 将脚本放在包含 a.jpg (源脸) 和图片的目录下运行
::   2. 手动模式: batch-swap.bat [源脸图片] [目标图片1] [目标图片2] ...

cd /d "%~dp0"

:: 检查 server.exe 是否存在
if not exist "server.exe" (
    echo [错误] 找不到 server.exe
    echo 请确保 server.exe 和 models 目录在同一目录下
    pause
    exit /b 1
)

:: 检查 Models 目录是否存在
if not exist "models" (
    echo [错误] 找不到 models 目录
    pause
    exit /b 1
)

:: 检查端口是否已被占用（server 是否已在运行）
netstat -ano | findstr ":8023" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Server 已在运行 (端口 8023 已占用)
) else (
    echo [INFO] 启动 MagicMirror server...
    start "" /B server.exe --workers auto
    
    :: 等待 server 启动 (最多 60 秒)
    echo [INFO] 等待 server 启动...
    set /a waited=0
    :wait_loop
    timeout /t 2 /nobreak >nul
    set /a waited+=2
    
    :: 检查端口是否监听
    netstat -ano | findstr ":8023" >nul 2>&1
    if %errorlevel% neq 0 (
        if !waited! GEQ 60 (
            echo [错误] Server 启动超时
            pause
            exit /b 1
        )
        echo [INFO] 已等待 !waited! 秒...
        goto :wait_loop
    )
    
    echo [INFO] Server 已启动
)

:: 等待模型加载完成
echo [INFO] 等待模型加载...
set /a waited=0
:prepare_loop
timeout /t 2 /nobreak >nul
set /a waited+=2

:: 尝试调用 prepare 接口
curl -s -X POST http://localhost:8023/prepare -H "Content-Type: application/json" -d "{}" >nul 2>&1
if %errorlevel% neq 0 (
    if !waited! GEQ 90 (
        echo [错误] Server 响应超时
        pause
        exit /b 1
    )
    echo [INFO] 已等待 !waited! 秒...
    goto :prepare_loop
)

echo [INFO] Server 已就绪
echo.

:: 批量换脸模式
if "%~1"=="" (
    :: 自动模式：查找 a.jpg 作为源脸
    if not exist "a.jpg" (
        echo [错误] 找不到源脸图片 a.jpg
        echo 请将源脸图片命名为 a.jpg 放到当前目录
        pause
        exit /b 1
    )
    
    echo [INFO] 使用 a.jpg 作为源脸图片
    set "SOURCE=a.jpg"
    
    :: 查找所有需要换脸的图片 (排除 a.jpg 和已有的 _output.jpg)
    set COUNT=0
    for %%f in (*.jpg *.jpeg *.png) do (
        if /i not "%%~nxf"=="a.jpg" (
            if /i not "%%~nxf"=="a_output.jpg" (
                if /i not "%%~nxf"=="batch-swap.bat" (
                    set /a COUNT+=1
                )
            )
        )
    )
    
    if !COUNT! EQU 0 (
        echo [错误] 没有可处理的图片
        pause
        exit /b 1
    )
    
    echo [INFO] 找到 !COUNT! 张图片需要处理
    echo.
    
    :: 处理每张图片
    for %%f in (*.jpg *.jpeg *.png) do (
        if /i not "%%~nxf"=="a.jpg" (
            if /i not "%%~nxf"=="a_output.jpg" (
                if /i not "%%~nxf"=="batch-swap.bat" (
                    echo [INFO] 正在处理: %%f
                    call :process_image "a.jpg" "%%f"
                )
            )
        )
    )
    
    echo.
    echo [完成] 批量换脸完成！
    pause
    exit /b 0
) else (
    :: 手动指定模式
    set "SOURCE=%~1"
    shift
    :arg_loop
    if "%~1"=="" goto :done_args
    echo [INFO] 正在处理: %~1
    call :process_image "%SOURCE%" "%~1"
    shift
    goto :arg_loop
    :done_args
)

echo.
echo [完成] 处理完成！
pause
exit /b 0

:: 处理单个图片
:process_image
set "SRC=%~1"
set "INPUT=%~2"
echo [INFO] 源: %SRC%, 目标: %INPUT%

:: 调用 server API 进行换脸
curl -s -X POST http://localhost:8023/task ^
    -H "Content-Type: application/json" ^
    -d "{\"id\":\"%INPUT%\",\"inputImage\":\"%SRC%\",\"targetFace\":\"%INPUT%\"}" 

echo.
goto :eof
