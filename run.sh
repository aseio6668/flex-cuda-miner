#!/bin/bash
# Quick start script for Linux

echo "Lyncoin Flex CUDA Miner - Quick Start"
echo "====================================="
echo ""

# Check if the executable exists
if [ ! -f "build/bin/flex-cuda-miner" ]; then
    echo "Miner executable not found. Building first..."
    chmod +x build.sh
    ./build.sh
    if [ $? -ne 0 ]; then
        echo "Build failed. Please check the error messages above."
        exit 1
    fi
    echo ""
fi

# Check for CUDA devices
echo "Checking for CUDA devices..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -L
else
    echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
fi
echo ""

echo "Starting Lyncoin Flex CUDA Miner..."
echo "Press Ctrl+C to stop mining."
echo ""

# Change to executable directory and run
cd build/bin
./flex-cuda-miner

cd ../..
