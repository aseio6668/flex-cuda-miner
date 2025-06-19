/**
 * CUDA implementation of CubeHash512
 * Based on the CubeHash specification by Daniel J. Bernstein
 */

#include "cuda_cubehash512.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define CUBEHASH_ROUNDS 16
#define CUBEHASH_BLOCK_SIZE 32

// CubeHash rotation
__device__ __forceinline__ uint32_t rotate_left(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// CubeHash round function
__device__ void cubehash_round(uint32_t state[32]) {
    uint32_t temp[16];
    
    // Add x_0jklm into x_1jklm modulo 2^32
    for (int i = 0; i < 16; i++) {
        state[i + 16] += state[i];
    }
    
    // Rotate x_0jklm upwards by 7 bits
    for (int i = 0; i < 16; i++) {
        state[i] = rotate_left(state[i], 7);
    }
    
    // Swap x_00klm with x_01klm
    for (int i = 0; i < 8; i++) {
        uint32_t tmp = state[i];
        state[i] = state[i + 8];
        state[i + 8] = tmp;
    }
    
    // XOR x_1jklm into x_0jklm
    for (int i = 0; i < 16; i++) {
        state[i] ^= state[i + 16];
    }
    
    // Swap x_1jk0m with x_1jk1m
    for (int i = 0; i < 16; i += 2) {
        uint32_t tmp = state[i + 16];
        state[i + 16] = state[i + 17];
        state[i + 17] = tmp;
    }
    
    // Add x_0jklm into x_1jklm modulo 2^32
    for (int i = 0; i < 16; i++) {
        state[i + 16] += state[i];
    }
    
    // Rotate x_0jklm upwards by 11 bits
    for (int i = 0; i < 16; i++) {
        state[i] = rotate_left(state[i], 11);
    }
    
    // Swap x_0j0lm with x_0j1lm
    for (int i = 0; i < 16; i += 4) {
        for (int j = 0; j < 2; j++) {
            uint32_t tmp = state[i + j];
            state[i + j] = state[i + j + 2];
            state[i + j + 2] = tmp;
        }
    }
    
    // XOR x_1jklm into x_0jklm
    for (int i = 0; i < 16; i++) {
        state[i] ^= state[i + 16];
    }
    
    // Swap x_1jkl0 with x_1jkl1
    for (int i = 16; i < 32; i += 2) {
        uint32_t tmp = state[i];
        state[i] = state[i + 1];
        state[i + 1] = tmp;
    }
}

__device__ void cuda_cubehash512_impl(const uint8_t* input, uint32_t inputLen, uint8_t* output) {
    uint32_t state[32];
    
    // Initialize state
    // Initial value for CubeHash512
    uint32_t iv[32] = {
        0xEA2BD4B4, 0xCCD6F29F, 0x63117E71, 0x35481EAE,
        0x22512D5B, 0xE5D94E63, 0x7E624131, 0xF4CC12BE,
        0xC2D0B696, 0x42AF2070, 0xD0720C35, 0x3361DA8C,
        0x28CCECA4, 0x8EF8AD83, 0x4680AC00, 0x40E5FBAB,
        0xD89041C3, 0x6107FBD5, 0x6C859D41, 0xF0B26679,
        0x09392549, 0x5FA25603, 0x65C892FD, 0x93CB6285,
        0x2AF2B5AE, 0x9E4B4E60, 0x774ABFDD, 0x85254725,
        0x15815AEB, 0x4AB6AAD6, 0x9CDAF8AF, 0xD6032C0A
    };
    
    for (int i = 0; i < 32; i++) {
        state[i] = iv[i];
    }
    
    // Process input in 32-byte blocks
    uint32_t pos = 0;
    while (pos + 32 <= inputLen) {
        // XOR block into state
        for (int i = 0; i < 8; i++) {
            uint32_t word = 0;
            for (int j = 0; j < 4; j++) {
                word |= ((uint32_t)input[pos + i * 4 + j]) << (j * 8);
            }
            state[i] ^= word;
        }
        
        // Apply rounds
        for (int round = 0; round < CUBEHASH_ROUNDS; round++) {
            cubehash_round(state);
        }
        
        pos += 32;
    }
    
    // Process remaining bytes with padding
    uint8_t padded_block[32];
    uint32_t remaining = inputLen - pos;
    
    // Copy remaining bytes
    for (uint32_t i = 0; i < remaining; i++) {
        padded_block[i] = input[pos + i];
    }
    
    // Add padding bit
    padded_block[remaining] = 0x80;
    
    // Zero fill the rest
    for (uint32_t i = remaining + 1; i < 32; i++) {
        padded_block[i] = 0x00;
    }
    
    // Process final block
    for (int i = 0; i < 8; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 4; j++) {
            word |= ((uint32_t)padded_block[i * 4 + j]) << (j * 8);
        }
        state[i] ^= word;
    }
    
    // Apply rounds
    for (int round = 0; round < CUBEHASH_ROUNDS; round++) {
        cubehash_round(state);
    }
    
    // Finalization: XOR 0x01 into state[31]
    state[31] ^= 0x01;
    
    // Apply final rounds
    for (int round = 0; round < CUBEHASH_ROUNDS; round++) {
        cubehash_round(state);
    }
    
    // Extract 512-bit output (64 bytes)
    for (int i = 0; i < 16; i++) {
        output[i * 4 + 0] = (state[i] >> 0) & 0xFF;
        output[i * 4 + 1] = (state[i] >> 8) & 0xFF;
        output[i * 4 + 2] = (state[i] >> 16) & 0xFF;
        output[i * 4 + 3] = (state[i] >> 24) & 0xFF;
    }
}
