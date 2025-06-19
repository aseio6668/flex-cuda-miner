/**
 * CUDA implementation of Shabal512
 * Based on the Shabal specification
 */

#include "cuda_shabal512.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define SHABAL_C_SIZE 16
#define SHABAL_B_SIZE 16
#define SHABAL_A_SIZE 12

__device__ __forceinline__ uint32_t shabal_rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// Shabal round function
__device__ void shabal_round(uint32_t A[SHABAL_A_SIZE], uint32_t B[SHABAL_B_SIZE], 
                            uint32_t C[SHABAL_C_SIZE], const uint32_t M[16], int r) {
    // Temporary variables
    uint32_t temp_A[SHABAL_A_SIZE];
    uint32_t temp_B[SHABAL_B_SIZE];
    
    // Copy A to temp
    for (int i = 0; i < SHABAL_A_SIZE; i++) {
        temp_A[i] = A[i];
    }
    
    // Update A
    for (int i = 0; i < SHABAL_A_SIZE; i++) {
        int j = (i + 1) % SHABAL_A_SIZE;
        int k = (i + 2) % SHABAL_A_SIZE;
        
        uint32_t t = temp_A[i] ^ temp_A[j] ^ temp_A[k];
        t = shabal_rotl32(t, r + i);
        A[i] = t ^ B[i] ^ (B[(i + 13) % 16] & ~B[(i + 6) % 16]) ^ M[i % 16];
    }
    
    // Update B
    for (int i = 0; i < SHABAL_B_SIZE; i++) {
        temp_B[i] = B[i];
    }
    
    for (int i = 0; i < SHABAL_B_SIZE; i++) {
        B[i] = shabal_rotl32(temp_B[(i + 1) % 16], 1) ^ A[i % SHABAL_A_SIZE];
    }
    
    // Update C
    for (int i = 0; i < SHABAL_C_SIZE; i++) {
        C[i] = shabal_rotl32(C[i], 11) ^ A[i % SHABAL_A_SIZE];
    }
}

__device__ void cuda_shabal512_impl(const uint8_t* input, uint32_t inputLen, uint8_t* output) {
    // Initialize state for Shabal-512
    uint32_t A[SHABAL_A_SIZE] = {
        0x20728DFD, 0x46C0BD53, 0xE782B699, 0x55304632,
        0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
        0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F
    };
    
    uint32_t B[SHABAL_B_SIZE] = {
        0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640,
        0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
        0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E,
        0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B
    };
    
    uint32_t C[SHABAL_C_SIZE] = {
        0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359,
        0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
        0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A,
        0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969
    };
    
    // Process input in 64-byte blocks
    uint32_t pos = 0;
    uint64_t totalBits = inputLen * 8;
    
    while (pos + 64 <= inputLen) {
        uint32_t M[16];
        
        // Load message block
        for (int i = 0; i < 16; i++) {
            M[i] = 0;
            for (int j = 0; j < 4; j++) {
                M[i] |= ((uint32_t)input[pos + i * 4 + j]) << (j * 8);
            }
        }
        
        // Apply 3 rounds
        for (int round = 0; round < 3; round++) {
            shabal_round(A, B, C, M, round);
        }
        
        pos += 64;
    }
    
    // Process remaining bytes with padding
    uint8_t padded_block[64] = {0};
    uint32_t remaining = inputLen - pos;
    
    // Copy remaining bytes
    for (uint32_t i = 0; i < remaining; i++) {
        padded_block[i] = input[pos + i];
    }
    
    // Add padding bit
    padded_block[remaining] = 0x80;
    
    // Add length in bits at the end
    *((uint64_t*)(padded_block + 56)) = totalBits;
    
    // Process final block
    uint32_t M[16];
    for (int i = 0; i < 16; i++) {
        M[i] = 0;
        for (int j = 0; j < 4; j++) {
            M[i] |= ((uint32_t)padded_block[i * 4 + j]) << (j * 8);
        }
    }
    
    // Apply final rounds
    for (int round = 0; round < 3; round++) {
        shabal_round(A, B, C, M, round);
    }
    
    // Additional blank rounds for finalization
    uint32_t blank[16] = {0};
    for (int round = 0; round < 4; round++) {
        shabal_round(A, B, C, blank, round);
    }
    
    // Extract 512-bit output (64 bytes)
    // Output comes from C array for Shabal-512
    for (int i = 0; i < 16; i++) {
        output[i * 4 + 0] = (C[i] >> 0) & 0xFF;
        output[i * 4 + 1] = (C[i] >> 8) & 0xFF;
        output[i * 4 + 2] = (C[i] >> 16) & 0xFF;
        output[i * 4 + 3] = (C[i] >> 24) & 0xFF;
    }
}
