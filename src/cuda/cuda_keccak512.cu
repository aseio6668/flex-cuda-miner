/**
 * CUDA implementation of Keccak (SHA-3) hash function
 * Optimized for GPU mining
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Keccak round constants
__constant__ uint64_t keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets
__constant__ int keccakf_rotc[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

// Lane positions
__constant__ int keccakf_piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

__device__ uint64_t rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__device__ void keccakf(uint64_t st[25]) {
    uint64_t t, bc[5];

    for (int r = 0; r < 24; r++) {
        // Theta
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        // Rho Pi
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = rotl64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        // Iota
        st[0] ^= keccakf_rndc[r];
    }
}

__device__ void cuda_keccak(const uint8_t* input, uint32_t inputLen, uint8_t* output, int mdlen) {
    uint64_t st[25];
    uint8_t temp[144];
    int rsiz = 200 - 2 * mdlen;
    int pt = 0;

    // Initialize state
    for (int i = 0; i < 25; i++) st[i] = 0;

    // Absorb input
    for (uint32_t i = 0; i < inputLen; i++) {
        temp[pt++] = input[i];
        if (pt >= rsiz) {
            for (int j = 0; j < rsiz; j++) {
                ((uint8_t*)st)[j] ^= temp[j];
            }
            keccakf(st);
            pt = 0;
        }
    }

    // Padding
    temp[pt++] = 0x01;
    while (pt < rsiz) temp[pt++] = 0;
    temp[rsiz - 1] |= 0x80;

    for (int j = 0; j < rsiz; j++) {
        ((uint8_t*)st)[j] ^= temp[j];
    }
    keccakf(st);

    // Squeeze
    for (int i = 0; i < mdlen; i++) {
        output[i] = ((uint8_t*)st)[i];
    }
}

__device__ void cuda_keccak512(const uint8_t* input, uint32_t inputLen, uint8_t* output) {
    cuda_keccak(input, inputLen, output, 64);
}

__device__ void cuda_keccak256(const uint8_t* input, uint32_t inputLen, uint8_t* output) {
    cuda_keccak(input, inputLen, output, 32);
}
