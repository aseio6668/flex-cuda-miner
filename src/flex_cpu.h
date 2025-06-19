/**
 * Flex Hash Algorithm - CPU Implementation
 * Reference implementation for validation and testing
 */

#ifndef FLEX_CPU_H
#define FLEX_CPU_H

#include <cstdint>
#include <cstring>

// Function declaration for CPU-based Flex hash computation
void flex_hash(const char* input, int size, unsigned char* output);
void flex_hash(const uint8_t* input, int size, uint8_t* output);

// Helper function for algorithm selection based on input
uint8_t select_algorithm(const uint8_t* input, int size);

#endif // FLEX_CPU_H
