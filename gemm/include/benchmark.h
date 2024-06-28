#pragma once

#include <iostream>

#include "common.h"
#include "gemm.cuh"

enum class GemmType { CPU, CUDA };

// Benchamrk for GEMM D = A*B + C
template <typename T>
class GemmBenchmarkBase {
   public:
    GemmBenchmarkBase(size_t m, size_t n, size_t k, GemmType type) : m(m), n(n), k(k), type(type) {
        // CPU buffer
        A = std::make_unique<T[]>(m * k);
        B = std::make_unique<T[]>(k * n);
        C = std::make_unique<T[]>(m * n);
        D = std::make_unique<T[]>(m * n);
        randomize_matrix<T>(A, m * k);
        randomize_matrix<T>(B, k * n);
        randomize_matrix<T>(C, m * n);

        // GPU buffer, these are called with move assignment.
        A_d = DevicePtr<T>(m * k);
        B_d = DevicePtr<T>(k * n);
        C_d = DevicePtr<T>(m * n);
        D_d = DevicePtr<T>(m * n);

        // Copy data from CPU to GPU
        HANDLE_ERROR(cudaMemcpy(A_d.get(), A.get(), m * k * sizeof(T), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(B_d.get(), B.get(), k * n * sizeof(T), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(C_d.get(), C.get(), m * n * sizeof(T), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(D_d.get(), D.get(), m * n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~GemmBenchmarkBase() {}

    virtual void run() = 0;

    void validation() {
        if (type == GemmType::CPU) {
            validation_cpu();
        } else {
            validation_gpu();
        }
        std::cout << "Validation done" << std::endl;
    }

    void validation_cpu() {
        auto D_golden = std::make_unique<T[]>(m * n);
        cpu_gemm(A.get(), B.get(), C.get(), D_golden.get(), m, n, k);
        for (size_t i = 0; i < m * n; i++) {
            auto diff = fabs(D_golden[i] - D[i]);
            if (diff > 1e-5) {
                std::cout << "Validation failed at index " << i << " diff " << diff << std::endl;
            }
        }
    }

    void validation_gpu() {
        auto D_cpu = std::make_unique<T[]>(m * n);
        HANDLE_ERROR(cudaMemcpy(D_cpu.get(), D_d.get(), m * n * sizeof(T), cudaMemcpyDeviceToHost));
        auto D_golden = std::make_unique<T[]>(m * n);
        cpu_gemm(A.get(), B.get(), C.get(), D_golden.get(), m, n, k);
        for (size_t i = 0; i < m * n; i++) {
            auto diff = fabs(D_golden[i] - D_cpu[i]);
            if (diff > 1e-5) {
                std::cout << "Validation failed at index " << i << " CPU: " << D_golden[i]
                          << " GPU: " << D_cpu.get()[i] << std::endl;
            }
        }
    }

   protected:
    std::unique_ptr<T[]> A{nullptr};
    DevicePtr<T> A_d;
    std::unique_ptr<T[]> B{nullptr};
    DevicePtr<T> B_d;
    std::unique_ptr<T[]> C{nullptr};
    DevicePtr<T> C_d;
    std::unique_ptr<T[]> D{nullptr};
    DevicePtr<T> D_d;
    size_t m;
    size_t n;
    size_t k;

    GemmType type;
};

template <typename T>
class GemmBenchmarkCPU : public GemmBenchmarkBase<T> {
   public:
    GemmBenchmarkCPU(size_t m, size_t n, size_t k) : GemmBenchmarkBase<T>(m, n, k, GemmType::CPU) {}

    void run() override {
        Timer timer;
        std::cout << "Running CPU GEMM" << std::endl;
        cpu_gemm(GemmBenchmarkBase<T>::A.get(), GemmBenchmarkBase<T>::B.get(),
                 GemmBenchmarkBase<T>::C.get(), GemmBenchmarkBase<T>::D.get(),
                 GemmBenchmarkBase<T>::m, GemmBenchmarkBase<T>::n, GemmBenchmarkBase<T>::k);
    }
};

template <typename T>
class GemmBenchmarkCUDA : public GemmBenchmarkBase<T> {
   public:
    GemmBenchmarkCUDA(size_t m, size_t n, size_t k, const std::string &kernel_name)
        : GemmBenchmarkBase<T>(m, n, k, GemmType::CUDA), kernel_name(kernel_name) {}

    void run() override {
        std::cout << "Running GPU CUDA" << kernel_name << " GEMM" << std::endl;
        Timer timer;
        run_cuda_gemm_kernel<T>(GemmBenchmarkBase<T>::A_d.get(), GemmBenchmarkBase<T>::B_d.get(),
                                GemmBenchmarkBase<T>::C_d.get(), GemmBenchmarkBase<T>::D_d.get(),
                                GemmBenchmarkBase<T>::m, GemmBenchmarkBase<T>::n,
                                GemmBenchmarkBase<T>::k, kernel_name);
    }

   private:
    std::string kernel_name;
};
