// clang-format off
#include <benchmark/benchmark.h> // must be first
// clang-format on
#include "cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{

__global__ void saxpy_kernel(float alpha, const float* x, float* y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { y[i] = alpha * x[i] + y[i]; }
}

void saxpy_cpu(float alpha, const float* x, float* y, int n)
{
    for (int i = 0; i < n; ++i) { y[i] = alpha * x[i] + y[i]; }
}

double percentile_90(const std::vector<double>& samples)
{
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const size_t idx = static_cast<size_t>(0.9 * static_cast<double>(sorted.size() - 1) + 0.5);
    return sorted[idx];
}

} // namespace

static void BM_saxpy_cpu(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    std::vector<float> x(static_cast<size_t>(n), 1.5f);
    std::vector<float> y(static_cast<size_t>(n), 2.0f);
    const float alpha = 0.5f;

    for (auto _ : state)
    {
        saxpy_cpu(alpha, x.data(), y.data(), n);
        benchmark::DoNotOptimize(y.data());
    }

    state.SetLabel("CPU: SAXPY na host");
}

static void BM_saxpy_gpu(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float alpha = 0.5f;

    float *d_x = null, *d_y = null;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    std::vector<float> hx(static_cast<size_t>(n), 1.5f);
    std::vector<float> hy(static_cast<size_t>(n), 2.0f);
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, hy.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    cudaEvent_t start = null, stop = null;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state)
    {
        CUDA_CHECK(cudaEventRecord(start));
        saxpy_kernel<<<blocks, threads>>>(alpha, d_x, d_y, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(static_cast<double>(ms) * 1e-3); // cudaEventElapsedTime jest w ms; manual time w sekundach
        benchmark::ClobberMemory();
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    state.SetLabel("GPU: SAXPY na urzadzeniu");
}

#define REGISTER_SAXPY_BENCH_COMMON(fn)                                                                                                                \
    BENCHMARK(fn)                                                                                                                                      \
        ->RangeMultiplier(4)                                                                                                                         \
        ->Range(1 << 12, 1 << 22)                                                                                                                    \
        ->Repetitions(10)                                                                                                                            \
        ->DisplayAggregatesOnly(true)                                                                                                                \
        ->ComputeStatistics("p90", percentile_90)                                                                                                    \
        ->MinTime(0.05)                                                                                                                              \
        ->Unit(benchmark::kNanosecond)

REGISTER_SAXPY_BENCH_COMMON(BM_saxpy_cpu);
REGISTER_SAXPY_BENCH_COMMON(BM_saxpy_gpu)->UseManualTime();

BENCHMARK_MAIN();