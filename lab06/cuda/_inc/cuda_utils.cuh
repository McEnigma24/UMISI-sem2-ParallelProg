#pragma once

#include <cuda_runtime.h>

#include "__preprocessor__.h"

#define CUDA_CHECK(x)                                                                                                                                \
    do                                                                                                                                               \
    {                                                                                                                                                \
        cudaError_t err = (x);                                                                                                                       \
        if (err != cudaSuccess) { FATAL_ERROR(std::string(cudaGetErrorString(err))); }                                                               \
    } while (0)