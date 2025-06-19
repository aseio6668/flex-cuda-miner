#ifndef CUDA_HASH_HEADERS_H
#define CUDA_HASH_HEADERS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for all hash functions
__device__ void cuda_bmw512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_groestl512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_skein512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_luffa512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_cubehash512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_shavite512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_simd512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_echo512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_hamsi512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_fugue512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_shabal512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_whirlpool512(const uint8_t* input, uint32_t inputLen, uint8_t* output);

// GhostRider algorithm functions
__device__ void cuda_ghostrider(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_jh512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_sha512(const uint8_t* input, uint32_t inputLen, uint8_t* output);

#ifdef __cplusplus
}
#endif

#endif // CUDA_HASH_HEADERS_H
