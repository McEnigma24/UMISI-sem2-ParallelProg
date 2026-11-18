// clang-format off
#include <gtest/gtest.h> // must be first
// clang-format on
#include "cuda_utils.cuh"

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { c[i] = a[i] + b[i]; }
}

static void vector_add_gpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c)
{
    const int n = static_cast<int>(a.size());
    ASSERT_EQ(n, static_cast<int>(b.size()));
    c.assign(static_cast<size_t>(n), 0.0f);

    float *d_a = null, *d_b = null, *d_c = null;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// 1) Prosty test niezależny
TEST(SampleTest, Sanity_2) { EXPECT_EQ(1, 1); }

// 2) TEST_F – testy oparte na fixtures
class MyFixture : public ::testing::Test
{
protected:
    std::vector<int> data;

    void SetUp() override
    {
        data = {0, 1, 2, 3, 4};
    }

    void TearDown() override { data.clear(); }
};

TEST_F(MyFixture, SumIsCorrect)
{
    int sum = 0;
    for (int v : data)
        sum += v;
    EXPECT_EQ(sum, 0 + 1 + 2 + 3 + 4);
}

TEST_F(MyFixture, FirstElementIsZero)
{
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(data.front(), 0);
}

// 3) TEST_P + INSTANTIATE_TEST_CASE_P – testy parametryzowane
typedef std::tuple<int, bool> testParams;
class ParityTest : public ::testing::TestWithParam<testParams>
{
};

// clang-format off
INSTANTIATE_TEST_CASE_P(ParityChecks, ParityTest, 
::testing::Values(
                    testParams(0, true)
                    , testParams(1, false)
                    , testParams(42, true)
                    , testParams(17, false)
));
// clang-format on

TEST_P(ParityTest, CheckEven)
{
    auto [value, isEven] = GetParam();
    EXPECT_EQ((value % 2 == 0), isEven);
}

TEST(CudaTest, VectorAdd)
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        GTEST_SKIP() << "Brak GPU — pomijam test CUDA";
    }

    const std::vector<float> a = {0.f, 1.f, 2.f, 3.f};
    const std::vector<float> b = {10.f, 20.f, 30.f, 40.f};
    std::vector<float> c;

    vector_add_gpu(a, b, c);

    for (size_t i = 0; i < a.size(); ++i) { EXPECT_FLOAT_EQ(c[i], a[i] + b[i]); }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}