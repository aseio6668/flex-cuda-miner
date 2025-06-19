/**
 * CUDA implementation of Whirlpool512
 * Based on the ISO standardized Whirlpool specification
 */

#include "cuda_whirlpool512.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define WHIRLPOOL_ROUNDS 10
#define WHIRLPOOL_BLOCK_SIZE 64

// Whirlpool S-box (8x8 substitution box)
__device__ const uint8_t whirlpool_sbox[256] = {
    0x18, 0x23, 0xC6, 0xE8, 0x87, 0xB8, 0x01, 0x4F, 0x36, 0xA6, 0xD2, 0xF5, 0x79, 0x6F, 0x91, 0x52,
    0x60, 0xBC, 0x9B, 0x8E, 0xA3, 0x0C, 0x7B, 0x35, 0x1D, 0xE0, 0xD7, 0xC2, 0x2E, 0x4B, 0xFE, 0x57,
    0x15, 0x77, 0x37, 0xE5, 0x9F, 0xF0, 0x4A, 0xDA, 0x58, 0xC9, 0x29, 0x0A, 0xB1, 0xA0, 0x6B, 0x85,
    0xBD, 0x5D, 0x10, 0xF4, 0xCB, 0x3E, 0x05, 0x67, 0xE4, 0x27, 0x41, 0x8B, 0xA7, 0x7D, 0x95, 0xD8,
    0xFB, 0xEE, 0x7C, 0x66, 0xDD, 0x17, 0x47, 0x9E, 0xCA, 0x2D, 0xBF, 0x07, 0xAD, 0x5A, 0x83, 0x33,
    0x63, 0x02, 0xAA, 0x71, 0xC8, 0x19, 0x49, 0xD9, 0xF2, 0xE3, 0x5B, 0x88, 0x9A, 0x26, 0x32, 0xB0,
    0xE9, 0x0F, 0xD5, 0x80, 0xBE, 0xCD, 0x34, 0x48, 0xFF, 0x7A, 0x90, 0x5F, 0x20, 0x68, 0x1A, 0xAE,
    0xB4, 0x54, 0x93, 0x22, 0x64, 0xF1, 0x73, 0x12, 0x40, 0x08, 0xC3, 0xEC, 0xDB, 0xA1, 0x8D, 0x3D,
    0x97, 0x00, 0xCF, 0x2B, 0x76, 0x82, 0xD6, 0x1B, 0xB5, 0xAF, 0x6A, 0x50, 0x45, 0xF3, 0x30, 0xEF,
    0x3F, 0x55, 0xA2, 0xEA, 0x65, 0xBA, 0x2F, 0xC0, 0xDE, 0x1C, 0xFD, 0x4D, 0x92, 0x75, 0x06, 0x8A,
    0xB2, 0xE6, 0x0E, 0x1F, 0x62, 0xD4, 0xA8, 0x96, 0xF9, 0xC5, 0x25, 0x59, 0x84, 0x72, 0x39, 0x4C,
    0x5E, 0x78, 0x38, 0x8C, 0xD1, 0xA5, 0xE2, 0x61, 0xB3, 0x21, 0x9C, 0x1E, 0x43, 0xC7, 0xFC, 0x04,
    0x51, 0x99, 0x6D, 0x0D, 0xFA, 0xDF, 0x7E, 0x24, 0x3B, 0xAB, 0xCE, 0x11, 0x8F, 0x4E, 0xB7, 0xEB,
    0x3C, 0x81, 0x94, 0xF7, 0xB9, 0x13, 0x2C, 0xD3, 0xE7, 0x6E, 0xC4, 0x03, 0x56, 0x44, 0x7F, 0xA9,
    0x2A, 0xBB, 0xC1, 0x53, 0xDC, 0x0B, 0x9D, 0x6C, 0x31, 0x74, 0xF6, 0x46, 0xAC, 0x89, 0x14, 0xE1,
    0x16, 0x3A, 0x69, 0x09, 0x70, 0xB6, 0xD0, 0xED, 0xCC, 0x42, 0x98, 0xA4, 0x28, 0x5C, 0xF8, 0x86
};

// Whirlpool round constants
__device__ const uint64_t whirlpool_round_constants[WHIRLPOOL_ROUNDS] = {
    0x1823C6E887B8014FULL, 0x36A6D2F5796F9152ULL, 0x60BC9B8EA30C7B35ULL, 0x1DE0D7C22E4BFEB7ULL,
    0x157737E59FF04ADAULL, 0x58C9290AB1A06B85ULL, 0xBD5D10F4CB3E0567ULL, 0xE427418BA77D95D8ULL,
    0xFBEE7C66DD17479EULL, 0xCA2DBF07AD5A8333ULL
};

__device__ __forceinline__ uint64_t whirlpool_rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// Whirlpool SubBytes transformation
__device__ void whirlpool_sub_bytes(uint64_t state[8]) {
    for (int i = 0; i < 8; i++) {
        uint8_t* bytes = (uint8_t*)&state[i];
        for (int j = 0; j < 8; j++) {
            bytes[j] = whirlpool_sbox[bytes[j]];
        }
    }
}

// Whirlpool ShiftColumns transformation
__device__ void whirlpool_shift_columns(uint64_t state[8]) {
    uint64_t temp[8];
    
    // Copy original state
    for (int i = 0; i < 8; i++) {
        temp[i] = state[i];
    }
    
    // Shift columns cyclically
    for (int row = 0; row < 8; row++) {
        uint8_t* state_bytes = (uint8_t*)&state[row];
        uint8_t* temp_bytes = (uint8_t*)&temp[row];
        
        for (int col = 0; col < 8; col++) {
            int src_col = (col + row) % 8;
            state_bytes[col] = temp_bytes[src_col];
        }
    }
}

// Whirlpool MixRows transformation using matrix multiplication in GF(2^8)
__device__ void whirlpool_mix_rows(uint64_t state[8]) {
    // Circulant matrix for Whirlpool MixRows
    const uint8_t mix_matrix[8] = {0x01, 0x01, 0x04, 0x01, 0x08, 0x05, 0x02, 0x09};
    
    for (int col = 0; col < 8; col++) {
        uint8_t column[8];
        uint8_t result[8] = {0};
        
        // Extract column
        for (int row = 0; row < 8; row++) {
            column[row] = ((uint8_t*)&state[row])[col];
        }
        
        // Matrix multiplication in GF(2^8)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                uint8_t a = mix_matrix[(i - j + 8) % 8];
                uint8_t b = column[j];
                
                // Simple GF(2^8) multiplication (simplified)
                if (a == 0x01) {
                    result[i] ^= b;
                } else if (a == 0x02) {
                    result[i] ^= (b << 1) ^ ((b & 0x80) ? 0x1B : 0);
                } else if (a == 0x04) {
                    uint8_t temp = (b << 1) ^ ((b & 0x80) ? 0x1B : 0);
                    result[i] ^= (temp << 1) ^ ((temp & 0x80) ? 0x1B : 0);
                } else if (a == 0x05) {
                    uint8_t temp = (b << 1) ^ ((b & 0x80) ? 0x1B : 0);
                    temp = (temp << 1) ^ ((temp & 0x80) ? 0x1B : 0);
                    result[i] ^= temp ^ b;
                } else if (a == 0x08) {
                    uint8_t temp = (b << 1) ^ ((b & 0x80) ? 0x1B : 0);
                    temp = (temp << 1) ^ ((temp & 0x80) ? 0x1B : 0);
                    result[i] ^= (temp << 1) ^ ((temp & 0x80) ? 0x1B : 0);
                } else if (a == 0x09) {
                    uint8_t temp = (b << 1) ^ ((b & 0x80) ? 0x1B : 0);
                    temp = (temp << 1) ^ ((temp & 0x80) ? 0x1B : 0);
                    temp = (temp << 1) ^ ((temp & 0x80) ? 0x1B : 0);
                    result[i] ^= temp ^ b;
                }
            }
        }
        
        // Store result back
        for (int row = 0; row < 8; row++) {
            ((uint8_t*)&state[row])[col] = result[row];
        }
    }
}

// Whirlpool AddRoundKey
__device__ void whirlpool_add_round_key(uint64_t state[8], const uint64_t round_key[8]) {
    for (int i = 0; i < 8; i++) {
        state[i] ^= round_key[i];
    }
}

// Whirlpool key schedule
__device__ void whirlpool_key_schedule(uint64_t master_key[8], uint64_t round_keys[WHIRLPOOL_ROUNDS + 1][8]) {
    // Copy master key as first round key
    for (int i = 0; i < 8; i++) {
        round_keys[0][i] = master_key[i];
    }
    
    // Generate round keys
    for (int round = 1; round <= WHIRLPOOL_ROUNDS; round++) {
        // Copy previous round key
        for (int i = 0; i < 8; i++) {
            round_keys[round][i] = round_keys[round - 1][i];
        }
        
        // Apply transformations
        whirlpool_sub_bytes(round_keys[round]);
        whirlpool_shift_columns(round_keys[round]);
        whirlpool_mix_rows(round_keys[round]);
        
        // Add round constant
        round_keys[round][0] ^= whirlpool_round_constants[round - 1];
    }
}

// Whirlpool round function
__device__ void whirlpool_round(uint64_t state[8], const uint64_t round_key[8]) {
    whirlpool_add_round_key(state, round_key);
    whirlpool_sub_bytes(state);
    whirlpool_shift_columns(state);
    whirlpool_mix_rows(state);
}

__device__ void cuda_whirlpool512_impl(const uint8_t* input, uint32_t inputLen, uint8_t* output) {
    uint64_t state[8] = {0}; // Initialize to zero
    uint64_t round_keys[WHIRLPOOL_ROUNDS + 1][8];
    
    // Process input in 64-byte blocks
    uint32_t pos = 0;
    uint64_t total_bits = inputLen * 8;
    
    while (pos + 64 <= inputLen) {
        uint64_t block[8];
        uint64_t key[8];
        
        // Load block as big-endian 64-bit words
        for (int i = 0; i < 8; i++) {
            block[i] = 0;
            for (int j = 0; j < 8; j++) {
                block[i] = (block[i] << 8) | input[pos + i * 8 + j];
            }
        }
        
        // Use current state as key for this block
        for (int i = 0; i < 8; i++) {
            key[i] = state[i];
        }
        
        // Generate round keys
        whirlpool_key_schedule(key, round_keys);
        
        // XOR block with state
        for (int i = 0; i < 8; i++) {
            state[i] ^= block[i];
        }
        
        // Apply Whirlpool rounds
        for (int round = 0; round < WHIRLPOOL_ROUNDS; round++) {
            whirlpool_round(state, round_keys[round]);
        }
        
        // Final round key addition
        whirlpool_add_round_key(state, round_keys[WHIRLPOOL_ROUNDS]);
        
        // XOR with original block (Miyaguchi-Preneel construction)
        for (int i = 0; i < 8; i++) {
            state[i] ^= block[i] ^ key[i];
        }
        
        pos += 64;
    }
    
    // Process final block with padding
    uint8_t final_block[64] = {0};
    uint32_t remaining = inputLen - pos;
    
    // Copy remaining bytes
    for (uint32_t i = 0; i < remaining; i++) {
        final_block[i] = input[pos + i];
    }
    
    // Add padding bit
    final_block[remaining] = 0x80;
    
    // If not enough space for length, process this block and create another
    if (remaining >= 56) {
        // Process current block
        uint64_t block[8];
        uint64_t key[8];
        
        for (int i = 0; i < 8; i++) {
            block[i] = 0;
            for (int j = 0; j < 8; j++) {
                block[i] = (block[i] << 8) | final_block[i * 8 + j];
            }
            key[i] = state[i];
        }
        
        whirlpool_key_schedule(key, round_keys);
        
        for (int i = 0; i < 8; i++) {
            state[i] ^= block[i];
        }
        
        for (int round = 0; round < WHIRLPOOL_ROUNDS; round++) {
            whirlpool_round(state, round_keys[round]);
        }
        
        whirlpool_add_round_key(state, round_keys[WHIRLPOOL_ROUNDS]);
        
        for (int i = 0; i < 8; i++) {
            state[i] ^= block[i] ^ key[i];
        }
        
        // Create new block with just length
        memset(final_block, 0, 64);
    }
    
    // Add length in bits (big-endian, last 8 bytes)
    for (int i = 0; i < 8; i++) {
        final_block[56 + i] = (total_bits >> (56 - i * 8)) & 0xFF;
    }
    
    // Process final block
    uint64_t block[8];
    uint64_t key[8];
    
    for (int i = 0; i < 8; i++) {
        block[i] = 0;
        for (int j = 0; j < 8; j++) {
            block[i] = (block[i] << 8) | final_block[i * 8 + j];
        }
        key[i] = state[i];
    }
    
    whirlpool_key_schedule(key, round_keys);
    
    for (int i = 0; i < 8; i++) {
        state[i] ^= block[i];
    }
    
    for (int round = 0; round < WHIRLPOOL_ROUNDS; round++) {
        whirlpool_round(state, round_keys[round]);
    }
    
    whirlpool_add_round_key(state, round_keys[WHIRLPOOL_ROUNDS]);
    
    for (int i = 0; i < 8; i++) {
        state[i] ^= block[i] ^ key[i];
    }
    
    // Extract 512-bit output (64 bytes) as big-endian
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (state[i] >> (56 - j * 8)) & 0xFF;
        }
    }
}
