#ifndef CUDA_KECCAK_H
#define CUDA_KECCAK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

__device__ void cuda_keccak512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
__device__ void cuda_keccak256(const uint8_t* input, uint32_t inputLen, uint8_t* output);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KECCAK_H
