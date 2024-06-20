#pragma once

#include <iostream>

#include "common.h"
#include "gemm.cuh"

template<typename T>
class GemmBenchmarkCPU : public GemmBenchmarkBase<T> {
    public:
        GemmBenchmarkCPU(size_t m, size_t n, size_t k) : GemmBenchmarkBase<T>(m, n, k) {}

        void run() override {
            Timer timer;
            std::cout << "Running CPU GEMM" << std::endl;
            cpu_gemm(
                GemmBenchmarkBase<T>::A.get(),
                GemmBenchmarkBase<T>::B.get(),
                GemmBenchmarkBase<T>::C.get(),
                GemmBenchmarkBase<T>::D.get(),
                GemmBenchmarkBase<T>::m,
                GemmBenchmarkBase<T>::n,
                GemmBenchmarkBase<T>::k
            );
        }

};

template<typename T>
class GemmBenchmarkCUDA : public GemmBenchmarkBase<T> {
    public:
        GemmBenchmarkCUDA(size_t m, size_t n, size_t k, const std::string& kernel_name) : GemmBenchmarkBase<T>(m, n, k), kernel_name(kernel_name) {}

        void run() override {
            std::cout << "Running GPU CUDA naive GEMM" << std::endl; 
            Timer timer;
            run_cuda_gemm_kernel<T>(
                GemmBenchmarkBase<T>::A_d.get(),
                GemmBenchmarkBase<T>::B_d.get(),
                GemmBenchmarkBase<T>::C_d.get(),
                GemmBenchmarkBase<T>::D_d.get(),
                GemmBenchmarkBase<T>::m,
                GemmBenchmarkBase<T>::n,
                GemmBenchmarkBase<T>::k,
                kernel_name
            );
        }
    
    private:
        std::string kernel_name;

};
