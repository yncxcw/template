
#include <functional>

#include "cuda_fp16.h"
#include "include/gemm.cuh"

static constexpr size_t BLOCK_SIZE = 32;

template <typename T>
__global__ void cuda_gemm_naive(const T* A, const T* B, const T* C, T* D, size_t m, size_t n,
                                size_t k) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        T sum = 0;
        for (size_t l = 0; l < k; ++l) {
            sum += A[i * k + l] * B[l * n + j];
        }
        D[i * n + j] = sum + C[i * n + j];
    }
}

template <typename T>
__global__ void cuda_gemm_tiled_shared_memory(const T* A, const T* B, const T* C, T* D, size_t m,
                                              size_t n, size_t k) {
    // shared by all threads in the block.
    __shared__ T shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // The tiling size should be the same as the block size.
    // assert(blockDim.x == blockDim.y);
    size_t tile_size = blockDim.x;

    size_t block_x = blockIdx.x;
    size_t block_y = blockIdx.y;

    size_t thread_x = threadIdx.x;
    size_t thread_y = threadIdx.y;

    // The coordinates of the element in the matix D
    // The `x` is the row index of both A and D
    // The `y` is the column index of both B and D
    size_t x = block_x * blockDim.x + thread_x;
    size_t y = block_y * blockDim.y + thread_y;

    if (x < m && y < n) {
        T sum = 0;
        // Loop over the tiles
        for (int i = 0; i < k / tile_size; i++) {
            // Load the tiles into shared memory
            // The x * k: find the row index of the tile in A
            // The i * tile_size: find the index of the tile in A at row x * k
            // The thread_x: find the column index of the element in the tile
            shared_A[thread_x][thread_y] = A[x * k + i * tile_size + thread_y];
            shared_B[thread_x][thread_y] = B[(i * tile_size + thread_x) * n + y];
            __syncthreads();

            // Compute the tile
            for (int j = 0; j < tile_size; j++) {
                sum += shared_A[thread_x][j] * shared_B[j][thread_y];
            }
            __syncthreads();
        }
        D[x * n + y] = C[x * n + y] + sum;
    }
}

template <typename T>
void run_cuda_gemm_kernel(const T* A, const T* B, const T* C, T* D, size_t m, size_t n, size_t k,
                          const std::string& kernel_name) {
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    if (kernel_name == "cuda_gemm_naive") {
        cuda_gemm_naive<T><<<grid_size, block_size>>>(A, B, C, D, m, n, k);
    } else if (kernel_name == "cuda_gemm_tiled_shared_memory") {
        cuda_gemm_tiled_shared_memory<T><<<grid_size, block_size>>>(A, B, C, D, m, n, k);
    } else {
        throw std::runtime_error("Unknown kernel name: " + kernel_name);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());
}

template __global__ void cuda_gemm_naive<float>(const float* A, const float* B, const float* C,
                                                float* D, size_t m, size_t n, size_t k);
template __global__ void cuda_gemm_naive<double>(const double* A, const double* B, const double* C,
                                                 double* D, size_t m, size_t n, size_t k);
template __global__ void cuda_gemm_naive<half>(const half* A, const half* B, const half* C, half* D,
                                               size_t m, size_t n, size_t k);

template __global__ void cuda_gemm_tiled_shared_memory<float>(const float* A, const float* B,
                                                              const float* C, float* D, size_t m,
                                                              size_t n, size_t k);
template __global__ void cuda_gemm_tiled_shared_memory<double>(const double* A, const double* B,
                                                               const double* C, double* D, size_t m,
                                                               size_t n, size_t k);
template __global__ void cuda_gemm_tiled_shared_memory<half>(const half* A, const half* B,
                                                             const half* C, half* D, size_t m,
                                                             size_t n, size_t k);

template void run_cuda_gemm_kernel<float>(const float* A, const float* B, const float* C, float* D,
                                          size_t m, size_t n, size_t k,
                                          const std::string& kernel_name);
template void run_cuda_gemm_kernel<double>(const double* A, const double* B, const double* C,
                                           double* D, size_t m, size_t n, size_t k,
                                           const std::string& kernel_name);
template void run_cuda_gemm_kernel<half>(const half* A, const half* B, const half* C, half* D,
                                         size_t m, size_t n, size_t k,
                                         const std::string& kernel_name);