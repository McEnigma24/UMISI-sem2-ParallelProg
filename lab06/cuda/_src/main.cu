#include "cuda_utils.cuh"

class test
{
    int tab[100];
};

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { c[i] = a[i] + b[i]; }
}

static void vector_add_gpu(const float* h_a, const float* h_b, float* h_c, int n)
{
    float *d_a = null, *d_b = null, *d_c = null;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
  line("CUDA devices: " + std::to_string(device_count));

    if (device_count > 0)
    {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        line(std::string("GPU[0]: ") + prop.name);
    }

    srand(time(NULL));
    line("It just works");
    time_stamp("It just works - timestamped");

    CORE::str::split_string("Hello World!", ' ');
    var(CORE::str::to_lower_case("Hello, World!"));

    show_sizeof(test);
    show_sizeof_many(test, 100);

    int num = 123456789;
    double num2 = 1234567.89123;

    cout << CORE::format_number(num) << endl;
    cout << CORE::format_number(num2) << endl;

    if (device_count > 0)
    {
        constexpr int n = 8;
        float ha[n] = {0, 1, 2, 3, 4, 5, 6, 7};
        float hb[n] = {7, 6, 5, 4, 3, 2, 1, 0};
        float hc[n]{};

        vector_add_gpu(ha, hb, hc, n);
        for (int i = 0; i < n; ++i) { var(hc[i]); }
    }

    for (int i = 0; i < 10; i++)
    {
        var(rand());
        cout.flush();
        sleep(300);
    }

    return 0;
}
#endif