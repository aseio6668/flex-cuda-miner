#ifndef CUDA_FUGUE512_H
#define CUDA_FUGUE512_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__device__ void cuda_fugue512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
#ifdef __cplusplus
}
#endif
#endif
