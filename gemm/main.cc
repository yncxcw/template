#include "include/benchmark.h"

int main() {
    size_t m = 2000;
    size_t n = 2000;
    size_t k = 2000;
    // Benchmark for GEMM
    {
        GemmBenchmarkCPU<double> benchmark(m, n, k);
        benchmark.run();
    }

    {
        GemmBenchmarkCUDA<double> benchmark(m, n, k, "cuda_gemm_naive");
        benchmark.run();
    }
    {
        GemmBenchmarkCUDA<double> benchmark(m, n, k, "cuda_gemm_tiled_shared_memory");
        benchmark.run();
    }
    return 0;
}
