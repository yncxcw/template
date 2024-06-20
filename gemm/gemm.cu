
#include <functional>

#include "include/gemm.cuh"

template <typename T>
__global__ void cuda_gemm_naive(const T* A, const T* B, const T* C, T* D, size_t m, size_t n, size_t k) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        float sum = 0;
        for (size_t l = 0; l < k; ++l) {
            sum += A[i * k + l] * B[l * n + j];
        }
        D[i * n + j] = sum + C[i * n + j];
    }
}

template <typename T>
void run_cuda_gemm_kernel(
    const T* A,
    const T* B,
    const T* C,
    T* D,
    size_t m,
    size_t n,
    size_t k,
    const std::string& kernel_name
) {
    dim3 block_size(32, 32);
    dim3 grid_size((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    if(kernel_name == "cuda_gemm_naive") {
        cuda_gemm_naive<T><<<grid_size, block_size>>>(A, B, C, D, m, n, k);
    } else {
        throw std::runtime_error("Unknown kernel name: " + kernel_name);
    }
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaGetLastError() );
}


template __global__ void cuda_gemm_naive<float>(const float* A, const float* B, const float* C, float* D, size_t m, size_t n, size_t k);
template __global__ void cuda_gemm_naive<double>(const double* A, const double* B, const double* C, double* D, size_t m, size_t n, size_t k);


template void run_cuda_gemm_kernel<float>(const float* A, const float* B, const float* C, float* D, size_t m, size_t n, size_t k, const std::string& kernel_name);
template void run_cuda_gemm_kernel<double>(const double* A, const double* B, const double* C, double* D, size_t m, size_t n, size_t k, const std::string& kernel_name);
