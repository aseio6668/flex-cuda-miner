#!/bin/bash
# Lyncoin Flex CUDA Miner - Production Deployment Script for Linux
# =================================================================

echo ""
echo "========================================"
echo "Lyncoin Flex CUDA Miner Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $2 found"
        return 0
    else
        echo -e "${RED}✗${NC} $2 not found"
        return 1
    fi
}

# Check for required tools
echo "[1/6] Checking dependencies..."

if ! check_command "nvcc" "CUDA toolkit"; then
    echo "ERROR: CUDA toolkit not found. Please install CUDA Toolkit 11.0 or higher."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

if ! check_command "cmake" "CMake"; then
    echo "ERROR: CMake not found. Please install CMake 3.18 or higher."
    echo "Ubuntu/Debian: sudo apt install cmake"
    echo "CentOS/RHEL: sudo yum install cmake"
    exit 1
fi

if ! check_command "gcc" "GCC compiler"; then
    echo "ERROR: GCC not found. Please install GCC."
    echo "Ubuntu/Debian: sudo apt install build-essential"
    echo "CentOS/RHEL: sudo yum groupinstall 'Development Tools'"
    exit 1
fi

# Check for optional dependencies
echo "[2/6] Checking optional dependencies..."
check_command "pkg-config" "pkg-config"
check_command "curl" "curl (for pool communication)"

# Install jsoncpp if available
echo "[3/6] Checking for JSON library..."
if pkg-config --exists jsoncpp; then
    echo -e "${GREEN}✓${NC} jsoncpp found via pkg-config"
else
    echo -e "${YELLOW}!${NC} jsoncpp not found - will use built-in simple parser"
fi

# Check GPU
echo "[4/6] Checking NVIDIA GPU..."
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
else
    echo -e "${YELLOW}!${NC} nvidia-smi not found - make sure NVIDIA drivers are installed"
fi

# Create build directory
echo "[5/6] Setting up build environment..."
mkdir -p build
cd build

# Configure project
echo "Configuring project with CMake..."
if cmake ..; then
    echo -e "${GREEN}✓${NC} CMake configuration successful"
else
    echo -e "${RED}✗${NC} CMake configuration failed!"
    exit 1
fi

# Build project
echo "[6/6] Building miner (this may take several minutes)..."
if make -j$(nproc); then
    echo -e "${GREEN}✓${NC} Build completed successfully"
else
    echo -e "${RED}✗${NC} Build failed!"
    exit 1
fi

cd ..

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Executables created:"
echo "  - build/bin/flex-cuda-miner   (Main miner)"
echo "  - build/bin/flex-miner-test   (Test suite)"
echo ""
echo "Next steps:"
echo "1. Edit config.ini with your pool settings"
echo "2. Run tests: ./build/bin/flex-miner-test"
echo "3. Start mining: ./build/bin/flex-cuda-miner"
echo ""
echo "For pool mining, use:"
echo "./build/bin/flex-cuda-miner --pool stratum+tcp://pool.example.com:4444 --user your_address"
echo ""
echo "See README.md and PRODUCTION_GUIDE.md for detailed instructions."
echo ""
