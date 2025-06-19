@echo off
echo Lyncoin Flex CUDA Miner - Quick Start
echo ====================================
echo.

REM Check if the executable exists
if not exist "build\bin\Release\flex-cuda-miner.exe" (
    echo Miner executable not found. Building first...
    call build.bat
    if %ERRORLEVEL% NEQ 0 (
        echo Build failed. Please check the error messages above.
        pause
        exit /b 1
    )
    echo.
)

REM Check for CUDA devices
echo Checking for CUDA devices...
nvidia-smi >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed.
    echo.
)

echo Starting Lyncoin Flex CUDA Miner...
echo Press Ctrl+C to stop mining.
echo.

REM Change to executable directory and run
cd build\bin\Release
flex-cuda-miner.exe

cd ..\..\..
pause
