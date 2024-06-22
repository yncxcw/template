#include "include/common.h"
#include "include/kernel.h"

template <class T>
__global__ void add(const T* a, const T* b, T* c, const int N) {
    int tid = blockIdx.x;  // this thread handles the data at its thread id
    if (tid < N) c[tid] = a[tid] + b[tid];
}

template <class T>
void gpu_add_loop(const T* ptr_a, const T* ptr_b, T* ptr_c, const int N) {
    T *dev_a, *dev_b, *dev_c;
    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(T)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(T)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(T)));

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy((void*)dev_a, (void*)ptr_a, N * sizeof(T), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy((void*)dev_b, (void*)ptr_b, N * sizeof(T), cudaMemcpyHostToDevice));

    add<T><<<N, 1>>>(dev_a, dev_b, dev_c, N);

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy((void*)ptr_c, (void*)dev_c, N * sizeof(T), cudaMemcpyDeviceToHost));

    // free the memory allocated on the GPU
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
}

template void gpu_add_loop<double>(const double* a, const double* b, double* c, int N);
template void gpu_add_loop<int>(const int* a, const int* b, int* c, const int N);
