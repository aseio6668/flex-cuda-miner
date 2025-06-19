cmake_minimum_required(VERSION 3.18)
project(LyncoinFlexCudaMiner LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(JSONCPP jsoncpp)
endif()

# If system jsoncpp not found, we'll create a simple version
if(NOT JSONCPP_FOUND)
    message(STATUS "System jsoncpp not found. Using simplified JSON parsing.")
    add_definitions(-DSIMPLE_JSON_PARSER)
endif()

enable_language(CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/flex_cuda.cu
    src/cuda/cuda_keccak512.cu
    src/cuda/cuda_blake512.cu
    src/cuda/cuda_bmw512_impl.cu
    src/cuda/cuda_groestl512_impl.cu
    src/cuda/cuda_skein512_impl.cu
    src/cuda/cuda_luffa512_impl.cu
    src/cuda/cuda_cubehash512_impl.cu
    src/cuda/cuda_shavite512_impl.cu
    src/cuda/cuda_simd512_impl.cu
    src/cuda/cuda_echo512_impl.cu
    src/cuda/cuda_shabal512_impl.cu
    src/cuda/cuda_hamsi512_impl.cu
    src/cuda/cuda_fugue512_impl.cu
    src/cuda/cuda_whirlpool512_impl.cu
    src/cuda/cuda_hash_stubs.cu
    src/cuda/cuda_ghostrider.cu
    src/cuda/cuda_jh512.cu
    src/cuda/cuda_sha512.cu
)

# C++ source files
set(CPP_SOURCES
    src/main.cpp
    src/mining_pool.cpp
    src/config.cpp
    src/flex_cpu.cpp
    src/performance_test.cpp
    src/test_suite.cpp
    src/simple_json.cpp
    src/rpc_client.cpp
)

# Create main miner executable
add_executable(flex-cuda-miner 
    src/main_working.cpp
    src/mining_pool.cpp
    src/rpc_client.cpp
    src/config.cpp
    src/flex_cpu.cpp
    src/performance_test.cpp
    src/simple_json.cpp
    ${CUDA_SOURCES}
)

# Create test suite executable
add_executable(flex-miner-test
    src/test_suite.cpp
    src/config.cpp
    src/flex_cpu.cpp
    src/performance_test.cpp
    src/mining_pool.cpp
    src/rpc_client.cpp
    src/simple_json.cpp
    ${CUDA_SOURCES}
)

# Set CUDA properties for both executables
set_property(TARGET flex-cuda-miner PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET flex-miner-test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(flex-cuda-miner ${CUDA_LIBRARIES})
target_link_libraries(flex-miner-test ${CUDA_LIBRARIES})

# Link JSON library if available
if(JSONCPP_FOUND)
    target_link_libraries(flex-cuda-miner ${JSONCPP_LIBRARIES})
    target_include_directories(flex-cuda-miner PRIVATE ${JSONCPP_INCLUDE_DIRS})
endif()

# Add threading support
find_package(Threads REQUIRED)
target_link_libraries(flex-cuda-miner Threads::Threads)
target_link_libraries(flex-miner-test Threads::Threads)

# Windows-specific libraries
if(WIN32)
    target_link_libraries(flex-cuda-miner ws2_32 winhttp)
    target_link_libraries(flex-miner-test ws2_32 winhttp)
    target_link_libraries(flex-miner-test ws2_32)
endif()

# Compiler-specific options
if(MSVC)
    # Visual Studio specific settings
    target_compile_options(flex-cuda-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
else()
    # GCC/Clang settings
    target_compile_options(flex-cuda-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

# Set output directory
set_target_properties(flex-cuda-miner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Print configuration info
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Optional: Add optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(flex-cuda-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        target_compile_options(flex-cuda-miner PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    else()
        target_compile_options(flex-cuda-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        target_compile_options(flex-cuda-miner PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    endif()
endif()
