#!/bin/bash
# Build script for Linux systems

echo "Building Lyncoin Flex CUDA Miner..."
echo "===================================="

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA toolkit first."
    echo "Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "CentOS/RHEL: sudo yum install cuda-toolkit"
    exit 1
fi

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake first."
    echo "Ubuntu/Debian: sudo apt install cmake"
    echo "CentOS/RHEL: sudo yum install cmake"
    exit 1
fi

# Check if GCC is installed
if ! command -v gcc &> /dev/null; then
    echo "Error: GCC not found. Please install build tools first."
    echo "Ubuntu/Debian: sudo apt install build-essential"
    echo "CentOS/RHEL: sudo yum groupinstall 'Development Tools'"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure project
echo "Configuring project..."
cmake .. -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    cd ..
    exit 1
fi

# Build project
echo "Building project..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Error: Build failed."
    cd ..
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo "Executable location: build/bin/flex-cuda-miner"
echo ""
echo "To run the miner:"
echo "cd build/bin && ./flex-cuda-miner"

cd ..
