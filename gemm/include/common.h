#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <math.h>
#include <memory>
#include <random>
#include <stdio.h>
#include <iostream>

class Timer {
    public:
        Timer() {
            start = std::chrono::high_resolution_clock::now(); 
        }
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Elapsed time " << duration << " milliseconds" << std::endl;
        }
    private:
        std::chrono::high_resolution_clock::time_point start;
};


template<typename T>
void randomize_matrix(std::unique_ptr<T[]>& x, const size_t len) {
    // Rmadomize matrix A to between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0, 10);
    for (size_t i = 0; i < len; i++) {
        x[i] = static_cast<T>(dis(gen));
    }
}

// Compute D = A*B + C 
template<typename T>
void cpu_gemm(const T *A, const T *B, const T *C, T *D, const size_t m, const size_t n, const size_t k) {
    // Check len of A, B, C, D
    /*
     A = m x k
     B = k x n
     C = m x n
     D = m x n
    */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            T sum{0};
            for (size_t l = 0; l < k; l++) {
                // A[i, l] * B[l, j]
                sum += A[i*k + l] * B[l*n + j];
            }
            D[i*n + j] = sum + C[i*n + j];
        }
    }
}


// Benchamrk for GEMM D = A*B + C
template<typename T>
class GemmBenchmarkBase {

    public:
        GemmBenchmarkBase(size_t m, size_t n, size_t k): m(m), n(n), k(k){
            A = std::make_unique<T[]>(m*k);
            B = std::make_unique<T[]>(k*n);
            C = std::make_unique<T[]>(m*n);
            D = std::make_unique<T[]>(m*n);
            randomize_matrix<T>(A, m*k);
            randomize_matrix<T>(B, k*n);
            randomize_matrix<T>(C, m*n);

        }
        ~GemmBenchmarkBase(){}
        virtual void run() = 0;
    
    protected:
        std::unique_ptr<T[]> A{nullptr};
        std::unique_ptr<T[]> B{nullptr};
        std::unique_ptr<T[]> C{nullptr};
        std::unique_ptr<T[]> D{nullptr};
        size_t m;
        size_t n;
        size_t k;
};



static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))