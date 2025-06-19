/**
 * CUDA GhostRider hash function header
 * GhostRider algorithm implementation for CUDA mining
 */

#ifndef CUDA_GHOSTRIDER_H
#define CUDA_GHOSTRIDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA device functions
__device__ void cuda_ghostrider(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_jh512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_sha512(const uint8_t* input, uint32_t inputLen, uint8_t* output);

// Host-callable functions
void ghostrider_hash_gpu(uint32_t threads, uint32_t startNonce, 
                        uint32_t* h_input, uint32_t* h_target, uint32_t* h_result);

#ifdef __cplusplus
}
#endif

#endif // CUDA_GHOSTRIDER_H
