@echo off
REM MagicMirror Face Swap Script (Batch version)
REM Usage: face-swap.bat
REM Requires: Server running on http://localhost:8023

setlocal enabledelayedexpansion

set "SERVER_URL=http://localhost:8023"
set "OUTPUT_DIR=%~dp0"

echo === MagicMirror Face Swap ===
echo.

REM Check if server is running
echo Checking server...
curl -s --connect-timeout 5 %SERVER_URL%/status >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Server not running at %SERVER_URL%
    echo Please start the server first:
    echo   cd F:\MagicMirror\src-server
    echo   .\target\release\magic-server.exe
    pause
    exit /b 1
)
echo   Server is running
echo.

REM Find a* and b* images
echo Looking for images in: %OUTPUT_DIR%
set "A_COUNT=0"
set "B_COUNT=0"

for %%f in (%OUTPUT_DIR%a*.jpg %OUTPUT_DIR%a*.png) do (
    if exist "%%f" set /a A_COUNT+=1
)
for %%f in (%OUTPUT_DIR%b*.jpg %OUTPUT_DIR%b*.png) do (
    if exist "%%f" set /a B_COUNT+=1
)

if %A_COUNT% equ 0 (
    echo ERROR: No images starting with 'a' found
    echo   Example: a_face.jpg
    pause
    exit /b 1
)
if %B_COUNT% equ 0 (
    echo ERROR: No images starting with 'b' found
    echo   Example: b_face.jpg
    pause
    exit /b 1
)

echo Found !A_COUNT! source image(s) (a*)
echo Found !B_COUNT! target image(s) (b*)
echo.

REM Prepare models
echo Preparing models...
curl -s -X POST %SERVER_URL%/prepare --connect-timeout 60 >nul
if %errorlevel% neq 0 (
    echo ERROR: Failed to prepare models
    pause
    exit /b 1
)
echo   Models loaded successfully
echo.

REM Process each combination
set "PROCESSED=0"
for %%a in (%OUTPUT_DIR%a*.jpg %OUTPUT_DIR%a*.png) do (
    if exist "%%a" (
        for %%b in (%OUTPUT_DIR%b*.jpg %OUTPUT_DIR%b*.png) do (
            if exist "%%b" (
                echo Processing: %%~nxa -> %%~nxb
                set "OUTPUT_NAME=c_%%~na_to_%%~nb.jpg"
                set "OUTPUT_PATH=%OUTPUT_DIR%%%OUTPUT_NAME"
                
                REM Create JSON body
                set "BODY={""id"":""swap-%PROCESSED%"",""input_image"":""%%~fa"",""target_face"":""%%~fb""}"
                
                curl -s -X POST %SERVER_URL%/task -H "Content-Type: application/json" -d "!BODY!" --connect-timeout 120 > "%TEMP%\swap_result.json" 2>&1
                
                if %errorlevel% equ 0 (
                    for /f "tokens=2 delims=:," %%r in ('type "%TEMP%\swap_result.json" ^| findstr "result"') do (
                        set "RESULT=%%~r"
                        set "RESULT=!RESULT:"=!"
                        if exist "!RESULT!" (
                            copy /Y "!RESULT!" "!OUTPUT_PATH!" >nul
                            echo   Success: !OUTPUT_NAME!
                            set /a PROCESSED+=1
                        )
                    )
                ) else (
                    echo   Error processing
                )
            )
        )
    )
)

echo.
echo === Complete ===
echo Processed: !PROCESSED! swap(s)
echo Output directory: %OUTPUT_DIR%
echo.
echo Output files start with 'c_' prefix
pause
