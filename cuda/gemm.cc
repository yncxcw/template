// My implementation and benchmark of the GEMM algorithm using CUDA
#include "include/gemm.h"

int main() {

    // Benchmark for GEMM
    GemmBenchmarkCPU<double> benchmark(1000, 1000, 1000);
    benchmark.run();

    return 0;
}