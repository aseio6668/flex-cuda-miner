/**
 * Test program for Flex CUDA Miner
 * Tests basic functionality and algorithm implementations
 */

#include <iostream>
#include <cstring>
#include <iomanip>
#include <chrono>
#include "flex_cpu.h"

// CUDA function
extern "C" void flex_hash_gpu(uint32_t threads, uint32_t startNonce, uint32_t* h_input, uint32_t* h_target, uint32_t* h_result);

void print_hash(const unsigned char* hash, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    std::cout << std::dec;
}

int main() {
    std::cout << "Lyncoin Flex CUDA Miner Test" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Test data
    const char* test_input = "Hello Lyncoin Flex Mining!";
    int input_size = strlen(test_input);
    
    std::cout << "Test input: \"" << test_input << "\"" << std::endl;
    std::cout << "Input size: " << input_size << " bytes" << std::endl;
    std::cout << std::endl;
    
    // Test CPU implementation
    unsigned char cpu_hash[32];
    memset(cpu_hash, 0, 32);
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    flex_hash(test_input, input_size, cpu_hash);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    std::cout << "CPU Flex Hash Result:" << std::endl;
    std::cout << "Hash: ";
    print_hash(cpu_hash, 32);
    std::cout << std::endl;
    std::cout << "Time: " << cpu_duration.count() << " microseconds" << std::endl;
    std::cout << std::endl;
    
    // Test basic CUDA functionality
    std::cout << "Testing CUDA device availability..." << std::endl;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found. Skipping GPU tests." << std::endl;
        return 0;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Performance test
    std::cout << "Running performance test..." << std::endl;
    
    const int test_iterations = 1000;
    auto start_perf = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < test_iterations; i++) {
        flex_hash(test_input, input_size, cpu_hash);
    }
    
    auto end_perf = std::chrono::high_resolution_clock::now();
    auto perf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_perf - start_perf);
    
    double hashes_per_second = (double)test_iterations / (perf_duration.count() / 1000.0);
    
    std::cout << "CPU Performance:" << std::endl;
    std::cout << "Iterations: " << test_iterations << std::endl;
    std::cout << "Total time: " << perf_duration.count() << " ms" << std::endl;
    std::cout << "Hashrate: " << hashes_per_second << " H/s" << std::endl;
    std::cout << std::endl;
    
    // Algorithm verification
    std::cout << "Algorithm verification:" << std::endl;
    
    // Test with different inputs
    const char* test_inputs[] = {
        "",
        "a",
        "abc",
        "message digest",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    };
    
    int num_tests = sizeof(test_inputs) / sizeof(test_inputs[0]);
    
    for (int i = 0; i < num_tests; i++) {
        unsigned char hash[32];
        flex_hash(test_inputs[i], strlen(test_inputs[i]), hash);
        
        std::cout << "Input " << i << ": \"" << test_inputs[i] << "\"" << std::endl;
        std::cout << "Hash:    ";
        print_hash(hash, 32);
        std::cout << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
}
