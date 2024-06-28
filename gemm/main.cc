#include "include/benchmark.h"

int main() {
    size_t m = 1;
    size_t n = 1;
    size_t k = 1;
    // Benchmark for GEMM
    {
        GemmBenchmarkCPU<double> benchmark(m, n, k);
        benchmark.run();
        benchmark.validation();
    }

    {
        GemmBenchmarkCUDA<double> benchmark(m, n, k, "cuda_gemm_naive");
        benchmark.run();
        benchmark.validation();
    }
    {
        GemmBenchmarkCUDA<double> benchmark(m, n, k, "cuda_gemm_tiled_shared_memory");
        benchmark.run();
        benchmark.validation();
    }
    return 0;
}
