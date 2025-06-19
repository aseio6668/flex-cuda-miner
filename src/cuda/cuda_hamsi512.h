#ifndef CUDA_HAMSI512_H
#define CUDA_HAMSI512_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__device__ void cuda_hamsi512(const uint8_t* input, uint32_t inputLen, uint8_t* output);
#ifdef __cplusplus
}
#endif
#endif
