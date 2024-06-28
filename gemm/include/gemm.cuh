#pragma once
#include <cuda_runtime.h>

#include "common.h"

template <typename T>
__global__ void cuda_gemm_naive(const T* A, const T* B, const T* C, T* D, size_t m, size_t n,
                                size_t k);

template <typename T>
__global__ void cuda_gemm_tiled_shared_memory(const T* A, const T* B, const T* C, T* D, size_t m,
                                              size_t n, size_t k);

template <typename T>
void run_cuda_gemm_kernel(const T* A, const T* B, const T* C, T* D, size_t m, size_t n, size_t k,
                          const std::string& kernel_name);