# Lyncoin Flex CUDA Miner - Project Structure

## Directory Structure
```
flex-cuda-miner/
├── src/                          # Source code
│   ├── main.cpp                  # Main application
│   ├── test.cpp                  # Test program
│   └── cuda/                     # CUDA kernels
│       ├── flex_cuda.cu          # Main Flex algorithm implementation
│       ├── cuda_keccak512.cu     # Keccak hash implementation
│       ├── cuda_blake512.cu      # Blake512 hash implementation
│       ├── cuda_hash_stubs.cu    # Placeholder implementations
│       └── *.h                   # Header files
├── build/                        # Build output directory
│   ├── bin/                      # Compiled executables
│   └── ...                       # CMake build files
├── CMakeLists.txt               # CMake build configuration
├── build.bat                    # Windows build script
├── build.sh                     # Linux build script
├── run.bat                      # Windows quick start script
├── run.sh                       # Linux quick start script
├── config.ini                   # Configuration template
├── README.md                    # Documentation
└── PROJECT_STRUCTURE.md         # This file
```

## Key Files Description

### Source Files
- **`main.cpp`**: Main miner application with GPU detection, mining loop, and statistics
- **`test.cpp`**: Test program to verify algorithm implementations
- **`flex_cuda.cu`**: Core CUDA implementation of the Flex algorithm
- **`cuda_keccak512.cu`**: Complete Keccak/SHA-3 implementation for CUDA
- **`cuda_blake512.cu`**: Complete Blake512 implementation for CUDA
- **`cuda_hash_stubs.cu`**: Placeholder implementations for remaining hash algorithms

### Build Files
- **`CMakeLists.txt`**: Cross-platform build configuration
- **`build.bat`**: Windows build script (uses Visual Studio)
- **`build.sh`**: Linux build script (uses GCC/Make)

### Runtime Files
- **`run.bat`**: Windows quick start script
- **`run.sh`**: Linux quick start script
- **`config.ini`**: Configuration template (not yet implemented)

## Implementation Status

### ✅ Completed
- Project structure and build system
- CUDA kernel framework
- Keccak512/256 hash implementations
- Blake512 hash implementation
- Main miner application
- GPU detection and configuration
- Basic mining loop
- Statistics and performance monitoring
- Cross-platform build support
- Documentation and usage instructions

### ⚠️ Partial
- Hash algorithm implementations (2/14 complete)
- Algorithm selection logic (framework done)
- Performance optimizations (basic level)

### ❌ Not Implemented
- Remaining 12 core hash algorithms
- CryptoNote algorithm variants
- Mining pool integration
- Configuration file support
- Multi-GPU support
- Advanced error handling
- Comprehensive testing suite

## Development Priorities

### Phase 1: Core Functionality
1. Complete remaining hash algorithm implementations
2. Verify algorithm selection logic
3. Performance testing and optimization
4. Cross-platform compatibility testing

### Phase 2: Mining Features
1. Mining pool integration (stratum protocol)
2. Configuration file support
3. Advanced logging and monitoring
4. Error handling and recovery

### Phase 3: Advanced Features
1. Multi-GPU support
2. Auto-tuning for different GPU architectures
3. Web-based monitoring interface
4. Mining profitability calculations

## Getting Started

### Quick Start
1. Install prerequisites (CUDA, CMake, compiler)
2. Run build script for your platform
3. Run the miner with default settings

### Development Setup
1. Clone the repository
2. Install development tools
3. Run test program to verify functionality
4. Make modifications and rebuild

## Contributing

Areas where contributions are most needed:
1. **Hash Algorithm Implementations**: Complete the remaining 12 algorithms
2. **Performance Optimization**: GPU-specific optimizations
3. **Pool Integration**: Stratum protocol implementation
4. **Testing**: Comprehensive test suite
5. **Documentation**: Code comments and usage examples

See README.md for detailed contribution guidelines.
