@echo off
REM Lyncoin Flex CUDA Miner - Production Deployment Script for Windows
REM =================================================================

echo.
echo ========================================
echo Lyncoin Flex CUDA Miner Setup
echo ========================================
echo.

REM Check for CUDA toolkit
echo [1/5] Checking CUDA installation...
nvcc --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: CUDA toolkit not found. Please install CUDA Toolkit 11.0 or higher.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)
echo ✓ CUDA toolkit found

REM Check for CMake
echo [2/5] Checking CMake installation...
cmake --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake not found. Please install CMake 3.18 or higher.
    echo Download from: https://cmake.org/download/
    pause
    exit /b 1
)
echo ✓ CMake found

REM Check for Visual Studio
echo [3/5] Checking Visual Studio installation...
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Visual Studio compiler not found.
    echo Please install Visual Studio 2019 or 2022 with C++ development tools.
    echo Make sure to run this script from "Developer Command Prompt for VS"
    pause
    exit /b 1
)
echo ✓ Visual Studio compiler found

REM Create build directory
echo [4/5] Setting up build environment...
if not exist "build" mkdir build
cd build

REM Configure project
echo Configuring project with CMake...
cmake -G "Visual Studio 16 2019" -A x64 ..
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed!
    pause
    cd ..
    exit /b 1
)

REM Build project
echo [5/5] Building miner (this may take several minutes)...
cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed!
    pause
    cd ..
    exit /b 1
)

cd ..

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executables created:
echo  - build\bin\Release\flex-cuda-miner.exe   (Main miner)
echo  - build\bin\Release\flex-miner-test.exe   (Test suite)
echo.
echo Next steps:
echo 1. Edit config.ini with your pool settings
echo 2. Run tests: build\bin\Release\flex-miner-test.exe
echo 3. Start mining: build\bin\Release\flex-cuda-miner.exe
echo.
echo For pool mining, use:
echo build\bin\Release\flex-cuda-miner.exe --pool stratum+tcp://pool.example.com:4444 --user your_address
echo.
echo See README.md and PRODUCTION_GUIDE.md for detailed instructions.
echo.
pause
