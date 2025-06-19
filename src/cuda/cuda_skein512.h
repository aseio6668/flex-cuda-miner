#ifndef CUDA_SKEIN512_H
#define CUDA_SKEIN512_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__device__ void cuda_skein512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
#ifdef __cplusplus
}
#endif
#endif
