@echo off
echo Building Lyncoin Flex CUDA Miner - Production Version
echo ====================================================
echo.

REM Check if CUDA is installed
where nvcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: CUDA toolkit not found. Please install CUDA toolkit first.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

REM Check if CMake is installed
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: CMake not found. Please install CMake first.
    echo Download from: https://cmake.org/download/
    pause
    exit /b 1
)

echo Checking CUDA installation...
nvcc --version
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure project
echo Configuring project with production features...
echo - Pool mining support
echo - Performance optimization
echo - Comprehensive testing
echo - Algorithm implementations: Keccak, Blake, BMW, Groestl
echo.

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo Error: CMake configuration failed.
    cd ..
    pause
    exit /b 1
)

REM Build project
echo Building project...
cmake --build . --config Release --parallel
if %ERRORLEVEL% NEQ 0 (
    echo Error: Build failed.
    cd ..
    pause
    exit /b 1
)

echo.
echo =====================================
echo Build completed successfully!
echo =====================================
echo.
echo Available executables:
echo 1. Main miner: build\bin\Release\flex-cuda-miner.exe
echo 2. Test suite: build\bin\Release\flex-miner-test.exe
echo.
echo Production features included:
echo ✓ Multi-algorithm support (Keccak, Blake, BMW, Groestl)
echo ✓ Mining pool integration (Stratum protocol)
echo ✓ Performance profiling and optimization
echo ✓ Comprehensive testing framework
echo ✓ Real-time statistics and monitoring
echo ✓ Cross-platform compatibility
echo.
echo Next steps:
echo 1. Run tests: build\bin\Release\flex-miner-test.exe
echo 2. Run miner: build\bin\Release\flex-cuda-miner.exe [options]
echo 3. Configure pool: Use --pool option for pool mining
echo.

cd ..
pause
