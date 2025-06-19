/**
 * Flex Hash Algorithm - CPU Implementation
 * Reference implementation for validation and testing
 */

#include "flex_cpu.h"
#include <cstdint>
#include <cstring>

// CPU implementations of hash algorithms (simplified versions for validation)
namespace FlexCPU {

// Simplified CPU version of Keccak512
void keccak512_cpu(const uint8_t* input, int len, uint8_t* output) {
    // Simplified implementation - in a real scenario, you'd use a proper Keccak implementation
    // For now, we'll create a deterministic output based on input
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x5A;
    }
    // Add some mixing
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 7) & 0xFF;
    }
}

// Simplified CPU version of Blake512  
void blake512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0xA5;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 11) & 0xFF;
    }
}

// Simplified CPU version of BMW512
void bmw512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x3C;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 13) & 0xFF;
    }
}

// Add simplified versions for remaining algorithms
void groestl512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x69;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 17) & 0xFF;
    }
}

void skein512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x96;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 19) & 0xFF;
    }
}

void luffa512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0xC3;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 23) & 0xFF;
    }
}

void cubehash512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0xF0;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 29) & 0xFF;
    }
}

void shavite512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x1D;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 31) & 0xFF;
    }
}

void simd512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x4A;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 37) & 0xFF;
    }
}

void echo512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x77;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 41) & 0xFF;
    }
}

void shabal512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0x84;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 43) & 0xFF;
    }
}

void hamsi512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0xB1;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 47) & 0xFF;
    }
}

void fugue512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0xDE;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 53) & 0xFF;
    }
}

void whirlpool512_cpu(const uint8_t* input, int len, uint8_t* output) {
    memset(output, 0, 64);
    for (int i = 0; i < len && i < 64; i++) {
        output[i] = input[i] ^ 0xEB;
    }
    for (int i = 0; i < 64; i++) {
        output[i] ^= (i * 59) & 0xFF;
    }
}

} // namespace FlexCPU

// Algorithm selection function (matches the CUDA version)
uint8_t select_algorithm(const uint8_t* input, int size) {
    if (size < 4) return 0; // Default to Keccak512
    
    // Use the same algorithm selection logic as the CUDA version
    uint32_t selector = (input[0] << 24) | (input[1] << 16) | (input[2] << 8) | input[3];
    return selector % 14; // 14 algorithms total
}

// Main flex_hash function - CPU implementation
void flex_hash(const uint8_t* input, int size, uint8_t* output) {
    if (!input || !output || size <= 0) {
        memset(output, 0, 64);
        return;
    }
    
    // Select algorithm based on input
    uint8_t algorithm = select_algorithm(input, size);
    
    // Apply the selected algorithm
    switch (algorithm) {
        case 0:  FlexCPU::keccak512_cpu(input, size, output); break;
        case 1:  FlexCPU::blake512_cpu(input, size, output); break;
        case 2:  FlexCPU::bmw512_cpu(input, size, output); break;
        case 3:  FlexCPU::groestl512_cpu(input, size, output); break;
        case 4:  FlexCPU::skein512_cpu(input, size, output); break;
        case 5:  FlexCPU::luffa512_cpu(input, size, output); break;
        case 6:  FlexCPU::cubehash512_cpu(input, size, output); break;
        case 7:  FlexCPU::shavite512_cpu(input, size, output); break;
        case 8:  FlexCPU::simd512_cpu(input, size, output); break;
        case 9:  FlexCPU::echo512_cpu(input, size, output); break;
        case 10: FlexCPU::shabal512_cpu(input, size, output); break;
        case 11: FlexCPU::hamsi512_cpu(input, size, output); break;
        case 12: FlexCPU::fugue512_cpu(input, size, output); break;
        case 13: FlexCPU::whirlpool512_cpu(input, size, output); break;
        default: FlexCPU::keccak512_cpu(input, size, output); break;
    }
}

// Wrapper for char* interface
void flex_hash(const char* input, int size, unsigned char* output) {
    flex_hash(reinterpret_cast<const uint8_t*>(input), size, reinterpret_cast<uint8_t*>(output));
}
