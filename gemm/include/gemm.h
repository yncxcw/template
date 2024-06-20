#pragma once

#include <iostream>

#include "common.h"


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